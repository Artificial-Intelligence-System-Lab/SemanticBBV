import os
import pickle
import argparse
from typing import Set, Dict, List
import sys
from tqdm import tqdm

from iced_x86 import (
    Decoder,
    DecoderOptions,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
)

# Import custom tokenizer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pre_processing.tokenizer import tokenize_binary_instruction

# Define special token IDs
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
UNK_ID = 3
MASK_ID = 4

class AsmVocab:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = 0
        self.sep_id = 1
        self.cls_id = 2
        self.unk_id = 3

    def load(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if not token:
                    continue
                self.token_to_id[token] = i
                self.id_to_token[i] = token

    def get_id(self, token):
        return self.token_to_id.get(token, self.unk_id)
    
    def length(self):
        return len(self.token_to_id)


def collect_asm_instructions(directory_path: str, vocabs: {str:AsmVocab}) -> tuple:
    """
    Traverse all saved_index.pkl files in the given directory, extract all unique assembly instructions and tokenize them
    
    Args:
        directory_path: directory path to traverse
        vocabs: vocabularies
        
    Returns:
        A tuple containing a set of all unique assembly instructions and a dictionary of corresponding encoding information
    """
    unique_instructions = set()
    instruction_encodings = {}  # Store instructions and their encoding information
    instruction_tokens = {}
    processed_files = 0
    
    # Create iced_x86 formatter
    formatter = Formatter(FormatterSyntax.NASM)
    formatter.gas_show_mnemonic_size_suffix = True
    
    # Traverse directory and subdirectories
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file == "saved_index.pkl":
                file_path = os.path.join(root, file)
                try:
                    # Load pickle file
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Traverse all functions
                    for func_name, func_data_list in data.items():
                        for func_data in func_data_list:
                            # Check if there is binary data
                            if len(func_data) > 2 and isinstance(func_data[2], bytes):
                                ip = func_data[0]
                                binary_data = func_data[2]

                                # Use iced_x86 to decode binary data
                                decoder = Decoder(64, binary_data, ip=ip, options=DecoderOptions.NONE)
                                offset = 0
                                
                                # Decode all instructions
                                while offset < len(binary_data):
                                    old_position = decoder.position
                                    instruction = decoder.decode()
                                    
                                    # Check if the decoder has advanced, if not, break out of the loop
                                    if decoder.position == old_position:
                                        offset += 1
                                        decoder.position = offset
                                        continue
                                    
                                    # Format the instruction
                                    mnemonic_str = formatter.format_mnemonic(instruction, FormatMnemonicOptions.NO_PREFIXES)
                                    operands_str = formatter.format_all_operands(instruction)
                                    formatted_instruction = f"{mnemonic_str}\t{operands_str}".strip()
                                    mnemonic_id = instruction.mnemonic
                                    
                                    # Add to unique instruction set
                                    unique_instructions.add(formatted_instruction)
                                    
                                    # Use tokenizer to tokenize the instruction
                                    if formatted_instruction not in instruction_encodings:
                                        try:
                                            insn_tokens = tokenize_binary_instruction(binary_data[old_position:decoder.position], instruction.ip)[0]
                                            
                                            tokens = {}
                                            tokens["asm"] = [vocabs["asm"].get_id(tok) for tok in insn_tokens["asm"]]
                                            tokens["mne"] = [vocabs["mne"].get_id(insn_tokens["mne"])] * len(insn_tokens["asm"])
                                            tokens["type"] = [
                                                vocabs["type"].get_id(tok) for tok, count in insn_tokens["type"] for _ in range(count)
                                            ]
                                            tokens["reg"] = [vocabs["reg"].get_id(tok) for tok in insn_tokens["reg"]]
                                            tokens["rw"] = [
                                                vocabs["rw"].get_id(tok) for tok, count in insn_tokens["rw"] for _ in range(count)
                                            ]
                                            tokens["eflag"] = [vocabs["eflag"].get_id(insn_tokens["eflag"])] * len(insn_tokens["asm"])

                                            instruction_encodings[formatted_instruction] = tokens
                                            instruction_tokens[formatted_instruction] = insn_tokens
                                        except Exception as e:
                                            import traceback
                                            print(f"Error encoding instruction '{formatted_instruction}': {e}")
                                            print(f"Exception details: {traceback.format_exc()}")
                                    
                                    print(formatted_instruction)
                                    print(instruction_tokens[formatted_instruction])
                                    # Move to next instruction
                                    offset = decoder.position

                    processed_files += 1
                    print(f"Processing file: {file_path} (processed: {processed_files} files, current instruction count: {len(unique_instructions)})")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    return unique_instructions, instruction_encodings


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect and process assembly instructions using tokenizer.py')
    parser.add_argument('directory', help='Directory path containing saved_index.pkl files')
    parser.add_argument('-o', '--output', default='unique_instructions_tokenized.txt', 
                        help='Output filename (default: unique_instructions_tokenized.txt)')
    parser.add_argument('--vocabs_dir', required=True, help='Vocabulary path')
    parser.add_argument('--encoding-output', default=None, 
                        help='Instruction encoding output filename (default: same as output file but with .pkl extension)')
    args = parser.parse_args()

    vocabs = {}

    vocab_config = {
        "asm": "asm_tokens.txt",
        "mne": "mne_tokens.txt",
        "type": "type_tokens.txt",
        "reg": "reg_tokens.txt",
        "rw": "rw_tokens.txt",
        "eflag": "eflag_tokens.txt"
    }

    for key, filename in vocab_config.items():
        vocab_path = os.path.join(args.vocabs_dir, filename)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please ensure all vocabulary files exist in the '{args.vocabs_dir}' directory.")
        vocab = AsmVocab()
        vocab.load(vocab_path)
        vocabs[key] = vocab
    
    # Collect all unique assembly instructions and encoding information
    print(f"Starting to collect assembly instructions...")
    unique_instructions, instruction_encodings = collect_asm_instructions(
        args.directory, vocabs
    )
    
    # Sort instructions
    sorted_instructions = sorted(unique_instructions)
    
    # Write results to file
    with open(args.output, 'w') as f:
        for instruction in sorted_instructions:
            f.write(f"{instruction}\n")
    
    print(f"Found {len(sorted_instructions)} unique assembly instructions, written to {args.output}")
    
    # If there is encoding information, save as pickle file
    if instruction_encodings:
        # If encoding output file is not specified, use default name
        encoding_output = args.encoding_output
        if not encoding_output:
            base_name = os.path.splitext(args.output)[0]
            encoding_output = f"{base_name}_encodings_fine.pkl"
        
        # Save encoding information
        with open(encoding_output, 'wb') as f:
            pickle.dump(instruction_encodings, f)
        
        # print information
        print(f"Encoding file write completed, written to {encoding_output}")

if __name__ == "__main__":
    main()