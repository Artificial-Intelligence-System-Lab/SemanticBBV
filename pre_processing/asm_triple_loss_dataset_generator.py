import os
import pickle
import argparse
import time
from typing import Set, List, Dict
import networkx as nx
import random
import multiprocessing as mp
from tqdm import tqdm
import sys
import gc  # Garbage collection module
from collections import defaultdict


# Ensure tokenizer can be imported correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import tokenize_binary_instruction
from asm_dataset_preprocessed_fine import AsmVocab

from iced_x86 import (
    Decoder,
    DecoderOptions,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
)

vocabs = {}
vocab_config = {
    "asm": "asm_tokens.txt",
    "mne": "mne_tokens.txt",
    "type": "type_tokens.txt",
    "reg": "reg_tokens.txt",
    "rw": "rw_tokens.txt",
    "eflag": "eflag_tokens.txt"
}

def process_file_with_iters(args):
    """
    Wrapper function to process a single file and save the extracted dataset
    
    Args:
        args: Tuple containing file path and number of iterations
    """
    file_path = args
    return process_file(file_path)

def process_file(file_path):
    """
    Process a single file and save the extracted dataset
    
    Args:
        file_path: File path to process
    """
    global vocabs
    local_dataset = []
    functions_count = 0
    
    try:
        # Directly use pickle.load to load data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Immediately perform garbage collection to release temporary objects that pickle.load may produce
        gc.collect()

        result = defaultdict(list)
        mapped_result = defaultdict(list)
        
        # Iterate over all functions
        for func_name, func_data_list in data.items():
            result[func_name] = []
            for func_data in func_data_list:
                functions_count += 1
                binary_code = func_data[2]
                ip = func_data[0]
                sequence = tokenize_binary_instruction(binary_code, ip)
                mapped_dict = defaultdict(list)
                for inst in sequence:
                    mapped_dict["asm"].extend([vocabs["asm"].get_id(tok) for tok in inst["asm"]])
                    mapped_dict["mne"].extend([vocabs["mne"].get_id(inst["mne"])] * len(inst["asm"]))
                    mapped_dict["type"].extend([
                        vocabs["type"].get_id(tok) for tok, count in inst["type"] for _ in range(count)
                    ])
                    mapped_dict["reg"].extend([vocabs["reg"].get_id(tok) for tok in inst["reg"]])
                    mapped_dict["rw"].extend([
                        vocabs["rw"].get_id(tok) for tok, count in inst["rw"] for _ in range(count)
                    ])
                    mapped_dict["eflag"].extend([vocabs["eflag"].get_id(inst["eflag"])] * len(inst["asm"]))
                result[func_name].append(sequence)
                length = len(mapped_dict["asm"])
                for key in mapped_dict:
                    assert len(mapped_dict[key]) == length
                mapped_result[func_name].append(mapped_dict)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return
    
    # Perform garbage collection again after processing
    gc.collect()
    
    # Save the processing results as separate saved_tokens.pkl files
    directory = os.path.dirname(file_path)
    tokens_file = os.path.join(directory, "saved_func_tokens.pkl")
    mapped_file = os.path.join(directory, "saved_func_tokens_cached.pkl")
    
    try:
        with open(tokens_file, 'wb') as f:
            # Use highest protocol version for efficiency
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {functions_count} training data entries to {tokens_file}")
        with open(mapped_file, "wb") as f:
            pickle.dump(mapped_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {functions_count} training data entries to {mapped_file}")
        return True
    except Exception as e:
        print(f"Error saving files {tokens_file}, {mapped_file}: {e}")
        return False

def collect_asm_instructions(directory_path: str, num_processes=None):
    """
    Use multiprocessing to traverse all saved_index.pkl files in the given directory, extract all unique assembly instructions,
    and create corresponding saved_tokens.pkl files for each saved_index.pkl file
    
    Args:
        directory_path: Directory path to traverse
        num_processes: Number of processes to use, defaults to CPU core count
    """
    # If number of processes not specified, use CPU core count
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Collect all files that need to be processed
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file == "saved_index.pkl":
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    total_files = len(file_paths)
    print(f"Found {total_files} files to process, using {num_processes} processes")
    
    # Create process pool
    pool = mp.Pool(processes=num_processes)
    
    # Prepare parameter list
    args_list = [(file_path) for file_path in file_paths]
    
    # Process files using the process pool and display progress bar
    with tqdm(total=total_files, desc="Processing files") as pbar:
        # Use imap_unordered to process files, so the progress bar can be updated in completion order
        for _ in pool.imap_unordered(process_file_with_iters, args_list):
            pbar.update(1)
    
    # Close process pool
    pool.close()
    pool.join()
    
    print(f"Processing completed, total {total_files} files processed")
    
    return total_files


def main():
    global vocabs
    global vocab_config
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect assembly instructions from pickle files')
    parser.add_argument('directory', help='Directory path containing saved_index.pkl files')
    parser.add_argument('--vocabs', type=str, required=True, help="Dictionary file directory path")
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of parallel processes, defaults to CPU core count')
    
    args = parser.parse_args()


    for key, filename in vocab_config.items():
        vocab_path = os.path.join(args.vocabs, filename)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please ensure all vocabulary files exist in the '{args.vocabs}' directory.")
        vocab = AsmVocab()
        vocab.load(vocab_path)
        vocabs[key] = vocab
    
    for key in vocabs:
        print(f"Vocab {key}, Length: {vocabs[key].length()}")
    
    # Set garbage collection threshold, more aggressive memory reclamation
    gc.set_threshold(100, 5, 5)  # Default values are (700, 10, 10)
    
    # Collect dataset and save as separate files
    start_time = time.time()
    collect_asm_instructions(args.directory, args.jobs)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
