import argparse
import angr
import random
import sys
import os
import pickle
import re
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import glob

# Import iced-x86 related modules
from iced_x86 import (
    Decoder,
    DecoderOptions,
    Instruction,
    InstructionInfoFactory,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
    RflagsBits,
    OpKind,
    OpAccess,
)

# Set constants
MAX_SEQ_LEN = 32 * 1024 - 2  # Maximum sequence length
MAX_WALK_ATTEMPTS = 20  # Maximum random walk attempts
MIN_SEQ_LEN = 100  # New: minimum sequence length requirement


class AsmVocab:
    """Simplified vocabulary class"""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = 0
        self.seq_id = 1
        self.cls_id = 2
        self.unk_id = 3  # Assume [UNK] ID is 3

    def load(self, vocab_path):
        """Load vocabulary from file"""
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                self.token_to_id[token] = i
                self.id_to_token[i] = token

    def get_id(self, token):
        """Get token ID, return UNK ID if not exists"""
        return self.token_to_id.get(token, self.unk_id)


class AsmTokenizer:
    """Instruction tokenizer"""

    def __init__(self, vocab_paths):
        self.token_vocab = AsmVocab()
        self.mnemonic_vocab = AsmVocab()
        self.op_kind_vocab = AsmVocab()
        self.op_id_vocab = AsmVocab()
        self.reg_r_vocab = AsmVocab()
        self.reg_w_vocab = AsmVocab()
        self.eflags_vocab = AsmVocab()

        # Load vocabulary
        self.token_vocab.load(vocab_paths["assemble_tokens"])
        self.mnemonic_vocab.load(vocab_paths["mnemonic_tokens"])
        self.op_kind_vocab.load(vocab_paths["op_kind_tokens"])
        self.op_id_vocab.load(vocab_paths["op_id_tokens"])
        self.reg_r_vocab.load(vocab_paths["reg_r_tokens"])
        self.reg_w_vocab.load(vocab_paths["reg_w_tokens"])
        self.eflags_vocab.load(vocab_paths["eflags_tokens"])

        # Create instruction info factory and formatter
        self.info_factory = InstructionInfoFactory()
        self.formatter = Formatter(FormatterSyntax.NASM)
        self.formatter.gas_show_mnemonic_size_suffix = True

    def normalize_insn(self, asm):
        """Normalize instruction format"""
        if "\t" not in asm:
            return asm, []

        opcode, op_str = asm.split("\t")

        op_str = op_str.replace(" + ", "+")
        op_str = op_str.replace(" - ", "-")
        op_str = op_str.replace(" * ", "*")
        op_str = op_str.replace(" : ", ":")

        # Define regex pattern for matching various numeric formats
        pattern = r"0x[0-9a-fA-F]+h?|\b[0-9a-fA-F]+h\b|\b[0-9]\b|(?<=[+\-*/])\d+"

        # Replace numeric values in operand strings
        def repl(match):
            start = match.start()
            preceding = op_str[max(0, start - 15) : start].lower()

            if "ptr" in preceding:
                return "PTR_ADDR"
            elif "rel" in preceding:
                return "REL_ADDR"
            else:
                return "IMM"

        op_str = re.sub(pattern, repl, op_str)

        if op_str:
            opnd_strs = [x.strip() for x in op_str.split(",")]
        else:
            opnd_strs = []

        # Iterate and replace numeric values in each operand
        opnd_strs = [re.sub(pattern, repl, opnd) for opnd in opnd_strs]

        return opcode, opnd_strs

    def get_eflag_bit_name(self, bit):
        """Map RflagsBits value to descriptive string"""
        flag_map = {
            RflagsBits.OF: "OF",
            RflagsBits.SF: "SF",
            RflagsBits.ZF: "ZF",
            RflagsBits.AF: "AF",
            RflagsBits.PF: "PF",
            RflagsBits.CF: "CF",
            RflagsBits.DF: "DF",
            RflagsBits.IF: "IF",
            RflagsBits.AC: "AC",
            RflagsBits.UIF: "UIF",
            RflagsBits.C0: "C0",
            RflagsBits.C1: "C1",
            RflagsBits.C2: "C2",
            RflagsBits.C3: "C3",
        }
        return flag_map.get(bit, "NULL")

    def get_eflags_string(self, instruction: Instruction) -> str:
        """Analyze instruction read and write to eflags"""
        if not (instruction.rflags_read or instruction.rflags_modified):
            return "NULL"

        updated_flags = []

        rflags_group = [
            (instruction.rflags_read, "READ"),
            (instruction.rflags_written, "WRITTEN"),
            (instruction.rflags_cleared, "CLEARED"),
            (instruction.rflags_set, "SET"),
            (instruction.rflags_undefined, "UNDEFINED"),
        ]

        for bits, class_name in rflags_group:
            for test_bit in [
                RflagsBits.OF,
                RflagsBits.SF,
                RflagsBits.ZF,
                RflagsBits.AF,
                RflagsBits.PF,
                RflagsBits.CF,
                RflagsBits.DF,
                RflagsBits.IF,
                RflagsBits.AC,
                RflagsBits.UIF,
                RflagsBits.C0,
                RflagsBits.C1,
                RflagsBits.C2,
                RflagsBits.C3,
            ]:
                if bits & test_bit:
                    name = self.get_eflag_bit_name(test_bit)
                    if name != "NULL":
                        updated_flags.append(f"{class_name}_{name}")

        if not updated_flags:
            return "NULL"

        return ",".join(updated_flags)

    def parse_operands(self, instruction: Instruction) -> List:
        """Parse instruction operand information"""
        results = []
        instruction_info = self.info_factory.info(instruction)

        for op_index in range(instruction.op_count):
            op_kind = instruction.op_kind(op_index)
            access = instruction_info.op_access(op_index)
            is_read = (
                1
                if access
                in [OpAccess.READ, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE]
                else 0
            )
            is_write = (
                1
                if access
                in [OpAccess.WRITE, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE]
                else 0
            )

            if op_kind == OpKind.REGISTER:
                op_id = instruction.op_register(op_index)
            else:
                op_id = 0

            results.append([op_kind, op_id, is_read, is_write])

        return results

    def decode_instruction(self, raw_bytes, ip):
        """Decode single instruction"""
        decoder = Decoder(64, raw_bytes, ip=ip, options=DecoderOptions.NONE)
        instruction = decoder.decode()

        if instruction is None or instruction.is_invalid:
            return None, None, None, None, 0

        # Get disassembly representation of instruction
        mnemonic_str = self.formatter.format_mnemonic(
            instruction, FormatMnemonicOptions.NO_PREFIXES
        )
        operands_str = self.formatter.format_all_operands(instruction)
        disasm = f"{mnemonic_str}\t{operands_str}"

        # Parse instruction information
        opcode, opnd_strs = self.normalize_insn(disasm)
        mnemonic = instruction.mnemonic
        op_info = self.parse_operands(instruction)
        eflags = self.get_eflags_string(instruction)

        return opcode, opnd_strs, mnemonic, op_info, eflags, instruction.len

    def encode_instruction(self, opcode, opnd_strs, mnemonic, op_info, eflags):
        """Encode instruction as token IDs"""
        # Encode opcode
        opcode_id = self.token_vocab.get_id(opcode)

        # Encode operands
        opnd_ids = [self.token_vocab.get_id(opnd) for opnd in opnd_strs]

        # Encode mnemonic
        mnemonic_id = self.mnemonic_vocab.get_id(str(mnemonic))

        # Encode operand information
        op_kind_ids = []
        op_id_ids = []
        reg_r_ids = []
        reg_w_ids = []

        for op in op_info:
            op_kind_ids.append(self.op_kind_vocab.get_id(str(op[0])))
            op_id_ids.append(self.op_id_vocab.get_id(str(op[1])))
            reg_r_ids.append(self.reg_r_vocab.get_id(str(op[2])))
            reg_w_ids.append(self.reg_w_vocab.get_id(str(op[3])))

        # Encode eflags
        eflags_id = self.eflags_vocab.get_id(eflags)

        return {
            "opcode_id": opcode_id,
            "opnd_ids": opnd_ids,
            "mnemonic_id": mnemonic_id,
            "op_kind_ids": op_kind_ids,
            "op_id_ids": op_id_ids,
            "reg_r_ids": reg_r_ids,
            "reg_w_ids": reg_w_ids,
            "eflags_id": eflags_id,
        }, len(opnd_ids) + 1


def load_token_dicts(tokens_dir):
    """Load token dictionary"""
    vocab_paths = {
        "assemble_tokens": os.path.join(tokens_dir, "assemble_tokens.txt"),
        "mnemonic_tokens": os.path.join(tokens_dir, "mnemonic_tokens.txt"),
        "op_kind_tokens": os.path.join(tokens_dir, "op_kind_tokens.txt"),
        "op_id_tokens": os.path.join(tokens_dir, "op_id_tokens.txt"),
        "reg_r_tokens": os.path.join(tokens_dir, "reg_r_tokens.txt"),
        "reg_w_tokens": os.path.join(tokens_dir, "reg_w_tokens.txt"),
        "eflags_tokens": os.path.join(tokens_dir, "eflags_tokens.txt"),
    }

    # Check if all files exist
    for path in vocab_paths.values():
        if not os.path.exists(path):
            print(f"Error: Vocabulary file not found {path}")
            return None

    return vocab_paths


def random_walk_cfg(cfg, tokenizer, max_seq_len=MAX_SEQ_LEN):
    """Random walk on CFG to generate instruction sequence"""
    sequences = []
    sequence_token_count = []

    # Get all basic blocks
    all_nodes = list(cfg.model.nodes())
    if not all_nodes:
        return sequences

    # Filter out nodes without blocks
    valid_nodes = [
        node for node in all_nodes if hasattr(node, "block") and node.block is not None
    ]
    if not valid_nodes:
        print("Warning: No valid basic block nodes found")
        return sequences

    # Try random walk multiple times
    for attempt in range(MAX_WALK_ATTEMPTS):
        # Randomly select starting node - prioritize nodes with multiple successors
        nodes_with_successors = [
            node
            for node in valid_nodes
            if hasattr(node, "successors") and len(node.successors) >= 1
        ]
        if nodes_with_successors:
            current_node = random.choice(nodes_with_successors)
        else:
            current_node = random.choice(valid_nodes)

        # Initialize sequence
        current_sequence = []
        current_length = 0

        # Random walk until reaching maximum length or unable to continue
        while current_length < max_seq_len - 2:
            # Get instructions of current basic block
            try:
                block = current_node.block
                if block is None:
                    break

                # Get raw bytes of basic block
                block_bytes = block.bytes
                block_addr = block.addr

                # Parse all instructions in basic block
                block_instructions = []
                offset = 0
                block_tokens = 0

                while offset < len(block_bytes):
                    opcode, opnd_strs, mnemonic, op_info, eflags, insn_len = (
                        tokenizer.decode_instruction(
                            block_bytes[offset:], block_addr + offset
                        )
                    )

                    if insn_len == 0:
                        break

                    if opcode is not None:
                        encoded_insn, length = tokenizer.encode_instruction(
                            opcode, opnd_strs, mnemonic, op_info, eflags
                        )
                        block_instructions.append(encoded_insn)
                        block_tokens += length

                    offset += insn_len

                # If adding this basic block will exceed maximum length, stop
                if current_length + block_tokens > max_seq_len:
                    break

                # Add instructions of basic block to sequence
                current_sequence.extend(block_instructions)
                current_length += block_tokens

                # Get successor nodes
                try:
                    successors = list(current_node.successors)
                    if not successors:
                        # If no successor nodes, try to continue from other nodes
                        break
                    # Assign selection probability based on number of successors
                    successor_weights = []
                    for succ in successors:
                        # Calculate number of successors for each successor node
                        succ_count = (
                            len(list(succ.successors))
                            if hasattr(succ, "successors")
                            else 0
                        )
                        # Weight = 1 + number of successors, ensure each node has base weight 1
                        successor_weights.append(1 + succ_count)

                    # Randomly select successor node based on weights
                    total_weight = sum(successor_weights)
                    if total_weight > 0:
                        # Normalize weight as probability
                        probabilities = [w / total_weight for w in successor_weights]
                        # Select by probability
                        current_node = random.choices(
                            successors, weights=probabilities, k=1
                        )[0]
                    else:
                        # If all weights are 0, select with equal probability
                        current_node = random.choice(successors)

                except Exception as e:
                    print(f"Error getting successor nodes: {e}")
                    # Try to jump to new node to continue
                    break

            except Exception as e:
                print(f"Error processing basic block: {e}")
                break

        # If sequence is not empty and length is sufficient, add to result
        if current_sequence and len(current_sequence) >= MIN_SEQ_LEN:
            sequences.append(current_sequence)
            sequence_token_count.append(current_length)
            print(
                f"Walk {attempt+1}: Generated sequence length = {len(current_sequence)}, token count = {current_length}"
            )

    return sequences, sequence_token_count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Receive binary file path from command line, use Angr to statically parse binary file to restore control flow graph, "
            "and use random walker method to collect instruction sequence. "
            "\nNote: Each returned sequence is aligned to basic block boundaries."
        )
    )
    parser.add_argument("binary", help="Binary file path to analyze")
    parser.add_argument("--tokens-dir", "-t", help="Path to token vocabulary directory", required=True)
    parser.add_argument("--output", "-o", help="Output file path", required=True)
    args = parser.parse_args()

    # Load token vocabulary
    vocab_paths = load_token_dicts(args.tokens_dir)
    print(vocab_paths)
    if vocab_paths is None:
        sys.exit(1)

    # Initialize tokenizer
    tokenizer = AsmTokenizer(vocab_paths)

    try:
        # Use angr to load binary file
        project = angr.Project(args.binary, auto_load_libs=False)

        # Get CFG
        cfg = project.analyses.CFGFast()
        print(f"Successfully generated control flow graph: {cfg.graph}")

        # Execute random walk to get instruction sequence
        sequences, token_count = random_walk_cfg(cfg, tokenizer)
        print(f"Successfully generated {len(sequences)} instruction sequences")
        average_length = np.mean(token_count)
        print(f"Average token sequence length: {average_length}")

        if not sequences:
            print("Failed to generate valid instruction sequence")
            sys.exit(1)

        # Construct output file path
        binary_name = os.path.splitext(os.path.basename(args.binary))[0]
        output_path = os.path.join(args.output, f"{binary_name}.pkl")

        # Save result to pickle file
        with open(output_path, "wb") as f:
            pickle.dump(sequences, f)

        print(f"Successfully saved instruction sequence to: {output_path}")

    except Exception as e:
        print(f"Error processing binary file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
