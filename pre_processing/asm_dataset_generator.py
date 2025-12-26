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

# Ensure tokenizer can be imported correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import tokenize_binary_instruction

from iced_x86 import (
    Decoder,
    DecoderOptions,
    Formatter,
    FormatterSyntax,
    FormatMnemonicOptions,
)

def random_walk_cfg(cfg):
    all_nodes = list(cfg.nodes())
    valid_nodes = [
        node for node in all_nodes if "raw" in cfg.nodes[node]
    ]
    if not valid_nodes:
        return [], 0
    
    start_node = random.choice(valid_nodes)

    current_node = start_node
    current_sequence = []
    current_length = 0
    
    # Add node counter, limit to traverse at most 10 nodes
    nodes_visited = 0
    max_nodes = 5

    while nodes_visited < max_nodes:  # Modify loop condition to node count limit
        try:
            # Increase node count
            nodes_visited += 1
            
            block_bytes = cfg.nodes[current_node]["raw"]
            block_addr = current_node

            block_instructions = tokenize_binary_instruction(block_bytes, block_addr)

            # Add all instructions of the current block, regardless of length limit
            current_sequence.extend(block_instructions)
            for instruction_tokens in block_instructions:
                current_length += len(instruction_tokens)
            
            if current_length > 10000:
                break

            # If maximum number of nodes has been visited, exit the loop
            if nodes_visited >= max_nodes:
                break
                
            # Choose next node
            successors = list(cfg.successors(current_node))
            if not successors:
                break
            
            weights = [1 + max(0, cfg.out_degree(succ)) for succ in successors]
            current_node = random.choices(successors, weights=weights, k=1)[0]

        except Exception as e:
            print(f"Error processing basic block: {e}")
            break

    return current_sequence, current_length

def process_file_with_iters(args):
    """
    Wrapper function to process a single file and save the extracted dataset
    
    Args:
        args: Tuple containing file path and number of iterations
    """
    file_path, iters_per_function = args
    return process_file(file_path, iters_per_function)

def process_file(file_path, iters_per_function=1):
    """
    Process a single file and save the extracted dataset
    
    Args:
        file_path: The file path to process
        iters_per_function: The number of random walk iterations per function
    """
    local_dataset = []
    functions_count = 0
    
    try:
        # Directly use pickle.load to load data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Immediately perform garbage collection to release temporary objects that pickle.load may produce
        gc.collect()
        
        # Traverse all functions
        functions_count = len(data)
        for func_name, func_data_list in data.items():
            for func_data in func_data_list:
                # Check graph
                if len(func_data) > 4 and isinstance(func_data[3], nx.Graph):
                    cfg = func_data[3]
                    for _ in range(iters_per_function):
                        # Random walk CFG
                        sequence, sequence_length = random_walk_cfg(cfg)
                        if sequence:  # Only add non-empty sequences
                            local_dataset.append(sequence)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return
    
    # Perform garbage collection again after processing
    gc.collect()
    
    # Save the processing result as a separate saved_tokens.pkl file
    directory = os.path.dirname(file_path)
    tokens_file = os.path.join(directory, "saved_tokens.pkl")
    
    try:
        with open(tokens_file, 'wb') as f:
            # Use the highest protocol version to improve efficiency
            pickle.dump(local_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(local_dataset)} training data entries to {tokens_file}")
        return True
    except Exception as e:
        print(f"Error saving file {tokens_file}: {e}")
        return False

def collect_asm_instructions(directory_path: str, iters_per_function=1, num_processes=None):
    """
    Use multiprocessing to traverse all saved_index.pkl files in the given directory, extract all unique assembly instructions,
    and create corresponding saved_tokens.pkl files for each saved_index.pkl file
    
    Args:
        directory_path: Directory path to traverse
        iters_per_function: Number of random walk iterations per function
        num_processes: Number of processes to use, defaults to CPU core count
    """
    # If number of processes is not specified, use CPU core count
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
    args_list = [(file_path, iters_per_function) for file_path in file_paths]
    
    # Use process pool to process files and display progress bar
    with tqdm(total=total_files, desc="Processing files") as pbar:
        # Use imap_unordered to process files, so the progress bar can be updated in completion order
        for _ in pool.imap_unordered(process_file_with_iters, args_list):
            pbar.update(1)
    
    # Close process pool
    pool.close()
    pool.join()
    
    print(f"Processing completed, processed {total_files} files in total")
    
    return total_files


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect assembly instructions from pickle files')
    parser.add_argument('directory', help='Directory path containing saved_index.pkl files')
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of parallel processes, defaults to CPU core count')
    parser.add_argument('-i', '--iters', type=int, default=1, help='Number of random walk iterations per function')
    
    args = parser.parse_args()
    
    # Set garbage collection threshold to recycle memory more aggressively
    gc.set_threshold(100, 5, 5)  # Default values are (700, 10, 10)
    
    # Collect dataset and save as separate files
    start_time = time.time()
    collect_asm_instructions(args.directory, args.iters, args.jobs)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
