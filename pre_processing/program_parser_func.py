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

MAX_SEQ_LEN = 32 * 1024 - 2
MAX_WALK_ATTEMPTS = 20
MIN_SEQ_LEN = 0


class AsmVocab:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = 0
        self.seq_id = 1
        self.cls_id = 2
        self.unk_id = 3

    def load(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                self.token_to_id[token] = i
                self.id_to_token[i] = token

    def get_id(self, token):
        return self.token_to_id.get(token, self.unk_id)


class AsmTokenizer:
    def __init__(self, vocab_paths):
        self.token_vocab = AsmVocab()
        self.mnemonic_vocab = AsmVocab()
        self.op_kind_vocab = AsmVocab()
        self.op_id_vocab = AsmVocab()
        self.reg_r_vocab = AsmVocab()
        self.reg_w_vocab = AsmVocab()
        self.eflags_vocab = AsmVocab()

        self.token_vocab.load(vocab_paths["assemble_tokens"])
        self.mnemonic_vocab.load(vocab_paths["mnemonic_tokens"])
        self.op_kind_vocab.load(vocab_paths["op_kind_tokens"])
        self.op_id_vocab.load(vocab_paths["op_id_tokens"])
        self.reg_r_vocab.load(vocab_paths["reg_r_tokens"])
        self.reg_w_vocab.load(vocab_paths["reg_w_tokens"])
        self.eflags_vocab.load(vocab_paths["eflags_tokens"])

        self.info_factory = InstructionInfoFactory()
        self.formatter = Formatter(FormatterSyntax.NASM)
        self.formatter.gas_show_mnemonic_size_suffix = True

    def normalize_insn(self, asm):
        if "\t" not in asm:
            return asm, []

        opcode, op_str = asm.split("\t")
        op_str = (
            op_str.replace(" + ", "+")
            .replace(" - ", "-")
            .replace(" * ", "*")
            .replace(" : ", ":")
        )

        pattern = r"0x[0-9a-fA-F]+h?|\b[0-9a-fA-F]+h\b|\b[0-9]\b|(?<=[+\-*/])\d+"

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
        opnd_strs = (
            [re.sub(pattern, repl, opnd.strip()) for opnd in op_str.split(",")]
            if op_str
            else []
        )
        return opcode, opnd_strs

    def get_eflag_bit_name(self, bit):
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

        return ",".join(updated_flags) if updated_flags else "NULL"

    def parse_operands(self, instruction: Instruction) -> List:
        results = []
        instruction_info = self.info_factory.info(instruction)
        for op_index in range(instruction.op_count):
            op_kind = instruction.op_kind(op_index)
            access = instruction_info.op_access(op_index)
            is_read = int(
                access in [OpAccess.READ, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE]
            )
            is_write = int(
                access
                in [OpAccess.WRITE, OpAccess.READ_WRITE, OpAccess.READ_COND_WRITE]
            )
            op_id = (
                instruction.op_register(op_index) if op_kind == OpKind.REGISTER else 0
            )
            results.append([op_kind, op_id, is_read, is_write])
        return results

    def decode_instruction(self, raw_bytes, ip):
        decoder = Decoder(64, raw_bytes, ip=ip, options=DecoderOptions.NONE)
        instruction = decoder.decode()
        if instruction is None or instruction.is_invalid:
            return None, None, None, None, 0
        mnemonic_str = self.formatter.format_mnemonic(
            instruction, FormatMnemonicOptions.NO_PREFIXES
        )
        operands_str = self.formatter.format_all_operands(instruction)
        disasm = f"{mnemonic_str}\t{operands_str}"
        opcode, opnd_strs = self.normalize_insn(disasm)
        mnemonic = instruction.mnemonic
        op_info = self.parse_operands(instruction)
        eflags = self.get_eflags_string(instruction)
        return opcode, opnd_strs, mnemonic, op_info, eflags, instruction.len

    def encode_instruction(self, opcode, opnd_strs, mnemonic, op_info, eflags):
        opcode_id = self.token_vocab.get_id(opcode)
        opnd_ids = [self.token_vocab.get_id(opnd) for opnd in opnd_strs]
        mnemonic_id = self.mnemonic_vocab.get_id(str(mnemonic))
        op_kind_ids = [self.op_kind_vocab.get_id(str(op[0])) for op in op_info]
        op_id_ids = [self.op_id_vocab.get_id(str(op[1])) for op in op_info]
        reg_r_ids = [self.reg_r_vocab.get_id(str(op[2])) for op in op_info]
        reg_w_ids = [self.reg_w_vocab.get_id(str(op[3])) for op in op_info]
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
    vocab_paths = {
        "assemble_tokens": os.path.join(tokens_dir, "assemble_tokens.txt"),
        "mnemonic_tokens": os.path.join(tokens_dir, "mnemonic_tokens.txt"),
        "op_kind_tokens": os.path.join(tokens_dir, "op_kind_tokens.txt"),
        "op_id_tokens": os.path.join(tokens_dir, "op_id_tokens.txt"),
        "reg_r_tokens": os.path.join(tokens_dir, "reg_r_tokens.txt"),
        "reg_w_tokens": os.path.join(tokens_dir, "reg_w_tokens.txt"),
        "eflags_tokens": os.path.join(tokens_dir, "eflags_tokens.txt"),
    }
    for path in vocab_paths.values():
        if not os.path.exists(path):
            print(f"Error: Vocabulary file not found {path}")
            return None
    return vocab_paths


def random_walk_cfg(cfg, tokenizer, start_node=None, max_seq_len=MAX_SEQ_LEN):
    sequences = []
    sequence_token_count = []
    all_nodes = list(cfg.model.nodes())
    valid_nodes = [
        node for node in all_nodes if hasattr(node, "block") and node.block is not None
    ]
    if not valid_nodes:
        return sequences, sequence_token_count

    for attempt in range(MAX_WALK_ATTEMPTS):
        current_node = start_node if start_node else random.choice(valid_nodes)
        current_sequence = []
        current_length = 0

        while current_length < max_seq_len - 2:
            try:
                block = current_node.block
                if block is None:
                    break
                block_bytes = block.bytes
                block_addr = block.addr

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

                if current_length + block_tokens > max_seq_len:
                    break

                current_sequence.extend(block_instructions)
                current_length += block_tokens

                successors = list(current_node.successors)
                if not successors:
                    break
                weights = [1 + len(list(succ.successors)) for succ in successors]
                current_node = random.choices(successors, weights=weights, k=1)[0]

            except Exception as e:
                print(f"Error processing basic block: {e}")
                break

        if current_sequence and len(current_sequence) >= MIN_SEQ_LEN:
            sequences.append(current_sequence)
            sequence_token_count.append(current_length)

    return sequences, sequence_token_count


def collect_sequences_per_function(project, tokenizer) -> Dict[str, List[List[Dict]]]:
    results = {}
    cfg = project.analyses.CFGFast()

    # Filter out non-external functions
    internal_functions = {}
    for func_addr, func in cfg.kb.functions.items():
        # Check if the function is an external function
        if not func.is_simprocedure and not func.is_plt and not func.is_syscall:
            internal_functions[func_addr] = func

    print(f"Total number of functions detected: {len(cfg.kb.functions)}")
    print(f"Number of internal functions: {len(internal_functions)}")

    for func_addr, func in internal_functions.items():
        func_name = func.name or f"sub_{func_addr:x}"
        print(f"\nProcessing function: {func_name} @ 0x{func_addr:x}")
        try:
            local_cfg = project.analyses.CFGFast(
                normalize=True, function_starts=[func_addr]
            )
            # Use model.get_any_node instead of local_cfg.get_any_node
            start_node = local_cfg.model.get_any_node(func_addr)
            if start_node is None:
                print(f"Function {func_name} has no start node, skipping")
                continue
            sequences, token_counts = random_walk_cfg(
                local_cfg, tokenizer, start_node=start_node
            )
            if sequences:
                results[func_name] = sequences
                print(f"Function {func_name} successfully generated {len(sequences)} sequences")
            else:
                print(f"Function {func_name} did not generate valid sequences")
        except Exception as e:
            print(f"Function {func_name} processing failed: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate instruction sequences for each function for similarity analysis")
    parser.add_argument("binary", help="Path to the binary file to be analyzed")
    parser.add_argument("--tokens-dir", "-t", help="Path to the token dictionary directory", required=True)
    parser.add_argument("--output", "-o", help="Output file path", required=True)
    args = parser.parse_args()

    vocab_paths = load_token_dicts(args.tokens_dir)
    if vocab_paths is None:
        sys.exit(1)

    tokenizer = AsmTokenizer(vocab_paths)

    try:
        project = angr.Project(args.binary, auto_load_libs=False)
        function_sequences = collect_sequences_per_function(project, tokenizer)

        if not function_sequences:
            print("Failed to collect instruction sequences for any functions")
            sys.exit(1)

        binary_name = os.path.splitext(os.path.basename(args.binary))[0]
        output_path = os.path.join(args.output, f"{binary_name}.pkl")

        with open(output_path, "wb") as f:
            pickle.dump(function_sequences, f)

        print(f"Successfully saved function instruction sequences to: {output_path}")

    except Exception as e:
        print(f"Error processing binary file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
