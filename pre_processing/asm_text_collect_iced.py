import os
import pickle
import argparse
from typing import Set, List, Dict

from iced_x86 import (
    Decoder,
    DecoderOptions,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
)

# Import AsmTokenizer class
from program_parser_func import AsmTokenizer, load_token_dicts

def collect_asm_instructions(directory_path: str, tokenizer=None, filter_unencoded=False) -> tuple:
    """
    Traverse all saved_index.pkl files in the given directory, extract all unique assembly instructions
    
    Args:
        directory_path: directory path to traverse
        tokenizer: optional AsmTokenizer instance for encoding instructions
        filter_unencoded: whether to filter instructions that cannot be encoded
        
    Returns:
        A tuple containing a set of all unique assembly instructions and a dictionary of corresponding encoding information
    """
    unique_instructions = set()
    instruction_encodings = {}  # Store instructions and their encoding information
    failed_encodings = set()    # Store instructions that failed to encode
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
                                
                                # Decode binary data using iced_x86
                                decoder = Decoder(64, binary_data, ip=ip, options=DecoderOptions.NONE)
                                offset = 0
                                
                                # Decode all instructions
                                while offset < len(binary_data):
                                    instruction = decoder.decode()
                                    if instruction.code == 0:  # Invalid instruction
                                        offset += 1
                                        decoder.position = offset
                                        continue
                                    
                                    # Format instruction
                                    formatted_instruction = formatter.format(instruction)
                                    
                                    # If tokenizer is provided, try to encode the instruction
                                    if tokenizer and formatted_instruction not in instruction_encodings and formatted_instruction not in failed_encodings:
                                        try:
                                            # Decode and encode instruction using tokenizer
                                            opcode, opnd_strs, mnemonic, op_info, eflags, _ = tokenizer.decode_instruction(
                                                binary_data[offset:offset+instruction.len], ip + offset
                                            )
                                            if opcode is not None:
                                                encoded_insn, _ = tokenizer.encode_instruction(
                                                    opcode, opnd_strs, mnemonic, op_info, eflags
                                                )
                                                instruction_encodings[formatted_instruction] = encoded_insn
                                                # Only successfully encoded instructions are added to the unique instructions set
                                                unique_instructions.add(formatted_instruction)
                                            else:
                                                # opcode is None, record as encoding failure
                                                failed_encodings.add(formatted_instruction)
                                                if not filter_unencoded:
                                                    unique_instructions.add(formatted_instruction)
                                        except Exception as e:
                                            print(f"Error encoding instruction '{formatted_instruction}': {e}")
                                            failed_encodings.add(formatted_instruction)
                                            if not filter_unencoded:
                                                unique_instructions.add(formatted_instruction)
                                    elif formatted_instruction not in failed_encodings:
                                        # If encoding is not needed or already encoded, add directly to unique instructions set
                                        if not tokenizer or formatted_instruction in instruction_encodings or not filter_unencoded:
                                            unique_instructions.add(formatted_instruction)
                                    
                                    # Move to next instruction
                                    offset = decoder.position
                            
                            # Also process the existing assembly instructions list
                            if len(func_data) > 1 and isinstance(func_data[1], list):
                                asm_instructions = func_data[1]
                                for instruction in asm_instructions:
                                    # Only add if not filtering or successfully encoded
                                    if not tokenizer or not filter_unencoded or instruction in instruction_encodings:
                                        unique_instructions.add(instruction)
                    
                    processed_files += 1
                    print(f"Processing file: {file_path} (Processed: {processed_files} files, current instructions: {len(unique_instructions)})")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    # If filtering unencodable instructions is needed, ensure unique instructions set only contains successfully encodable instructions
    if filter_unencoded and tokenizer:
        unique_instructions = set(instruction_encodings.keys())
        print(f"Filtered instructions count: {len(unique_instructions)}")
        print(f"Encoding failed instructions count: {len(failed_encodings)}")
    
    return unique_instructions, instruction_encodings


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect unique assembly instructions from pickle files')
    parser.add_argument('directory', help='Directory path containing saved_index.pkl files')
    parser.add_argument('-o', '--output', default='unique_instructions_iced.txt', 
                        help='Output filename (default: unique_instructions_iced.txt)')
    parser.add_argument('--tokens-dir', '-t', help='Token dictionary directory path for instruction encoding')
    parser.add_argument('--encoding-output', default=None, 
                        help='Instruction encoding output filename (default: same as output file but with .pkl extension)')
    parser.add_argument('--filter-unencoded', action='store_true',
                        help='Whether to filter unencodable instructions to make unique instructions set and encoding info count consistent')
    
    args = parser.parse_args()
    
    # Initialize tokenizer (if tokens-dir is provided)
    tokenizer = None
    if args.tokens_dir:
        vocab_paths = load_token_dicts(args.tokens_dir)
        if vocab_paths:
            try:
                tokenizer = AsmTokenizer(vocab_paths)
                print(f"Token dictionary loaded, will collect instruction encoding information")
            except Exception as e:
                print(f"Error initializing AsmTokenizer: {e}")
                tokenizer = None
    
    # Collect all unique assembly instructions and encoding information
    unique_instructions, instruction_encodings = collect_asm_instructions(
        args.directory, tokenizer, filter_unencoded=args.filter_unencoded
    )
    
    # Sort the instructions
    sorted_instructions = sorted(unique_instructions)
    
    # Write results to file
    with open(args.output, 'w') as f:
        for instruction in sorted_instructions:
            f.write(f"{instruction}\n")
    
    print(f"Found {len(sorted_instructions)} unique assembly instructions, written to {args.output}")
    
    # If there is encoding information, save as pickle file
    if instruction_encodings and tokenizer:
        # If encoding output file is not specified, use default name
        encoding_output = args.encoding_output
        if not encoding_output:
            base_name = os.path.splitext(args.output)[0]
            encoding_output = f"{base_name}_encodings.pkl"
        # Save encoding information
        with open(encoding_output, 'wb') as f:
            pickle.dump(instruction_encodings, f)
        
        print(f"Saved encoding information for {len(instruction_encodings.keys())} instructions to {encoding_output}")

if __name__ == "__main__":
    main()
