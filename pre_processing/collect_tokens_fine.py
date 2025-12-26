#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import re
from time import time
import sys
import multiprocessing as mp
from tqdm import tqdm

import angr

# iced-x86 related modules
from iced_x86 import (
    Decoder,
    DecoderOptions,
    Instruction,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
)

# Import custom tokenizer module
from tokenizer import tokenize_binary_instruction
import traceback


def load_token_sets(output_dir):
    # Define dict format file paths
    asm_tokens_dict_path = os.path.join(output_dir, "asm_tokens_dict.pkl")
    mne_tokens_dict_path = os.path.join(output_dir, "mne_tokens_dict.pkl")
    type_tokens_dict_path = os.path.join(output_dir, "type_tokens_dict.pkl")
    reg_tokens_dict_path = os.path.join(output_dir, "reg_tokens_dict.pkl")
    rw_tokens_dict_path = os.path.join(output_dir, "rw_tokens_dict.pkl")
    eflag_tokens_dict_path = os.path.join(output_dir, "eflag_tokens_dict.pkl")

    # Initialize token dictionaries
    # Initialize various token dictionaries
    asm_tokens = {}
    mne_tokens = {}
    type_tokens = {}
    reg_tokens = {}
    rw_tokens = {}
    eflag_tokens = {}
    
    # Try to load existing token sets
    try:
        # Prefer to load dict format first
        if os.path.exists(asm_tokens_dict_path):
            with open(asm_tokens_dict_path, "rb") as f:
                asm_tokens = pickle.load(f)
            print(f"Loaded assembly tokens dictionary: {len(asm_tokens)} entries")
        if os.path.exists(mne_tokens_dict_path):
            with open(mne_tokens_dict_path, "rb") as f:
                mne_tokens = pickle.load(f)
            print(f"Loaded mnemonic tokens dictionary: {len(mne_tokens)} entries")
        if os.path.exists(type_tokens_dict_path):
            with open(type_tokens_dict_path, "rb") as f:
                type_tokens = pickle.load(f)
            print(f"Loaded type tokens dictionary: {len(type_tokens)} entries")
        if os.path.exists(reg_tokens_dict_path):
            with open(reg_tokens_dict_path, "rb") as f:
                reg_tokens = pickle.load(f)
            print(f"Loaded register tokens dictionary: {len(reg_tokens)} entries")
        if os.path.exists(rw_tokens_dict_path):
            with open(rw_tokens_dict_path, "rb") as f:
                rw_tokens = pickle.load(f)
            print(f"Loaded read/write tokens dictionary: {len(rw_tokens)} entries")
        if os.path.exists(eflag_tokens_dict_path):
            with open(eflag_tokens_dict_path, "rb") as f:
                eflag_tokens = pickle.load(f)
            print(f"Loaded flag tokens dictionary: {len(eflag_tokens)} entries")
    except Exception as e:
        print(f"Error loading token sets: {e}")
        sys.exit(1)
    
    return asm_tokens, mne_tokens, type_tokens, reg_tokens, rw_tokens, eflag_tokens


def save_token_sets(output_dir, asm_tokens, mne_tokens, type_tokens, reg_tokens, rw_tokens, eflag_tokens):
    """
    Save token dictionaries to specified path, saving both set format and dict format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dict format token sets
    with open(os.path.join(output_dir, "asm_tokens_dict.pkl"), "wb") as f:
        pickle.dump(asm_tokens, f)
    with open(os.path.join(output_dir, "mne_tokens_dict.pkl"), "wb") as f:
        pickle.dump(mne_tokens, f)
    with open(os.path.join(output_dir, "type_tokens_dict.pkl"), "wb") as f:
        pickle.dump(type_tokens, f)
    with open(os.path.join(output_dir, "reg_tokens_dict.pkl"), "wb") as f:
        pickle.dump(reg_tokens, f)
    with open(os.path.join(output_dir, "rw_tokens_dict.pkl"), "wb") as f:
        pickle.dump(rw_tokens, f)
    with open(os.path.join(output_dir, "eflag_tokens_dict.pkl"), "wb") as f:
        pickle.dump(eflag_tokens, f)
    
    # Also save set format token sets (for backward compatibility)
    with open(os.path.join(output_dir, "asm_tokens.pkl"), "wb") as f:
        pickle.dump(set(asm_tokens.keys()), f)
    with open(os.path.join(output_dir, "mne_tokens.pkl"), "wb") as f:
        pickle.dump(set(mne_tokens.keys()), f)
    with open(os.path.join(output_dir, "type_tokens.pkl"), "wb") as f:
        pickle.dump(set(type_tokens.keys()), f)
    with open(os.path.join(output_dir, "reg_tokens.pkl"), "wb") as f:
        pickle.dump(set(reg_tokens.keys()), f)
    with open(os.path.join(output_dir, "rw_tokens.pkl"), "wb") as f:
        pickle.dump(set(rw_tokens.keys()), f)
    with open(os.path.join(output_dir, "eflag_tokens.pkl"), "wb") as f:
        pickle.dump(set(eflag_tokens.keys()), f)
    
    print(f"Assembly tokens dictionary: {len(asm_tokens)} entries")
    print(f"Mnemonic tokens dictionary: {len(mne_tokens)} entries")
    print(f"Type tokens dictionary: {len(type_tokens)} entries") 
    print(f"Register tokens dictionary: {len(reg_tokens)} entries")
    print(f"Read/write tokens dictionary: {len(rw_tokens)} entries")
    print(f"Flag tokens dictionary: {len(eflag_tokens)} entries")


def process_file(file_path, debug=False):
    """
    Process a single file and return extracted tokens
    
    Args:
        file_path: File path to process
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing counts of various token types
    """
    # Initialize local token dictionaries
    local_asm_tokens = {}
    local_mne_tokens = {}
    local_type_tokens = {}
    local_reg_tokens = {}
    local_rw_tokens = {}
    local_eflag_tokens = {}
    instruction_count = 0
    
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
                    
                    # Use tokenize_binary_instruction function to process binary data
                    tokens_list = tokenize_binary_instruction(binary_data, ip)
                    
                    if debug and tokens_list:
                        print(f"Parsed {len(tokens_list)} instructions")
                    
                    # Update counts in token dictionaries
                    for token_results in tokens_list:
                        # Process assembly layer tokens
                        for token in token_results["asm"]:
                            local_asm_tokens[token] = local_asm_tokens.get(token, 0) + 1
                        
                        mne_str = token_results["mne"]
                        local_mne_tokens[mne_str] = local_mne_tokens.get(mne_str, 0)
                        
                        # Process type layer tokens
                        for token_type, count in token_results["type"]:
                            local_type_tokens[token_type] = local_type_tokens.get(token_type, 0) + 1
                        
                        # Process register layer tokens
                        for token in token_results["reg"]:
                            local_reg_tokens[token] = local_reg_tokens.get(token, 0) + 1
                        
                        # Process read/write layer tokens
                        for token_rw, count in token_results["rw"]:
                            local_rw_tokens[token_rw] = local_rw_tokens.get(token_rw, 0) + 1
                        
                        # Process flag layer tokens
                        eflag_str = token_results["eflag"]
                        local_eflag_tokens[eflag_str] = local_eflag_tokens.get(eflag_str, 0) + 1
                        
                        instruction_count += 1
        
        return {
            "asm_tokens": local_asm_tokens,
            "mne_tokens": local_mne_tokens,
            "type_tokens": local_type_tokens,
            "reg_tokens": local_reg_tokens,
            "rw_tokens": local_rw_tokens,
            "eflag_tokens": local_eflag_tokens,
            "instruction_count": instruction_count,
            "file_path": file_path
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        print("Error traceback information:")
        traceback.print_exc()
        return {
            "asm_tokens": {},
            "mne_tokens": {},
            "type_tokens": {},
            "reg_tokens": {},
            "rw_tokens": {},
            "eflag_tokens": {},
            "instruction_count": 0,
            "file_path": file_path,
            "error": str(e)
        }


# Create a wrapper function for imap_unordered
def _process_file_wrapper(args_tuple):
    """
    Helper function to unpack arguments and call process_file.
    
    Args:
        args_tuple: Tuple containing (file_path, debug_flag)
        
    Returns:
        Return value of process_file function
    """
    file_path, debug_flag = args_tuple
    return process_file(file_path, debug_flag)


def main(binary_path: str, output_dir: str, debug: bool = False, num_processes: int = None) -> None:
    """
    1) Traverse all saved_index.pkl files in the directory;
    2) Use multiprocessing to process each file in parallel and extract tokens;
    3) Dynamically merge processing results;
    4) Save tokens to pickle files.
    
    Args:
        binary_path: Binary file directory path
        output_dir: Output directory path
        debug: Whether to print debug information
        num_processes: Number of processes, defaults to CPU core count
    """
    # If number of processes is not specified, use CPU core count
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Load existing token sets
    asm_tokens, mne_tokens, type_tokens, reg_tokens, rw_tokens, eflag_tokens = load_token_sets(output_dir)
    
    # Collect all files that need to be processed
    file_paths = []
    for root, _, files in os.walk(binary_path):
        for file in files:
            if file == "saved_index.pkl":
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    total_files = len(file_paths)
    print(f"Found {total_files} files to process, using {num_processes} processes")
    
    # Create process pool
    pool = mp.Pool(processes=num_processes)
    
    # Initialize counters
    total_instruction_count = 0
    processed_files = 0
    
    # Prepare argument list to pass to imap_unordered
    process_args = [(file_path, debug) for file_path in file_paths]
    
    # Use process pool to process files and display progress bar
    # Add dynamic_ncols=True to make progress bar adapt better to terminal width changes
    with tqdm(total=total_files, desc="Processing files", leave=True, disable=False, dynamic_ncols=True) as pbar:
        # Use imap_unordered to process files, so progress bar can be updated in completion order
        for result in pool.imap_unordered(_process_file_wrapper, process_args):
            if result:
                # Dynamically merge results
                if "error" not in result:
                    # Update assembly tokens
                    for token, count in result["asm_tokens"].items():
                        asm_tokens[token] = asm_tokens.get(token, 0) + count
                    
                    for token, count in result["mne_tokens"].items():
                        mne_tokens[token] = mne_tokens.get(token, 0) + count
                    
                    # Update type tokens
                    for token, count in result["type_tokens"].items():
                        type_tokens[token] = type_tokens.get(token, 0) + count
                    
                    # Update register tokens
                    for token, count in result["reg_tokens"].items():
                        reg_tokens[token] = reg_tokens.get(token, 0) + count
                    
                    # Update read/write tokens
                    for token, count in result["rw_tokens"].items():
                        rw_tokens[token] = rw_tokens.get(token, 0) + count
                    
                    # Update flag tokens
                    for token, count in result["eflag_tokens"].items():
                        eflag_tokens[token] = eflag_tokens.get(token, 0) + count
                    
                    # Update instruction count
                    instruction_count = result["instruction_count"]
                    total_instruction_count += instruction_count
                    
                    # Update progress bar information
                    pbar.set_postfix({
                        "Instructions": total_instruction_count,
                        "Current file instructions": instruction_count
                    })
                
                processed_files += 1
                # Periodically display processing progress
                if processed_files % 10 == 0:
                    print(f"Processed {processed_files}/{total_files} files, currently {total_instruction_count} instructions total")
            
            pbar.update(1)
    
    # Close process pool
    pool.close()
    pool.join()
    
    # Save tokens to pickle files
    os.makedirs(output_dir, exist_ok=True)
    save_token_sets(output_dir, asm_tokens, mne_tokens, type_tokens, reg_tokens, rw_tokens, eflag_tokens)
    print(f"Processing completed, processed {total_files} files, extracted {total_instruction_count} instructions")
    print(f"Token sets saved to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage: python collect_tokens_fine.py <path_to_binary> <output_dir> [num_processes]\n")
        sys.exit(1)

    bin_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    # If process count parameter is provided, use the specified number
    num_processes = None
    if len(sys.argv) > 3:
        try:
            num_processes = int(sys.argv[3])
        except ValueError:
            print("Process count must be an integer, will use default value (CPU core count)")
    
    # Set debug True/False as needed
    main(bin_path, out_dir, debug=False, num_processes=num_processes)