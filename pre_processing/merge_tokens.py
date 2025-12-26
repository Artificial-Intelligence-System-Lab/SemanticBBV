import os
import pickle
import argparse
import time
from tqdm import tqdm
import gc
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import random

def load_file(file_path, result_queue, error_queue, min_seq_len, max_seq_len, sample_ratio=0.2):
    """
    Load a single file, filter its sequences by length, and randomly sample a percentage of them.

    Args:
        file_path (str): Path to the file to load.
        result_queue (queue.Queue): Queue to store results.
        error_queue (queue.Queue): Queue to store errors.
        min_seq_len (int or None): Minimum sequence length to keep.
        max_seq_len (int or None): Maximum sequence length to keep.
        sample_ratio (float): Percentage of sequences to keep after filtering (0.0-1.0).
    """
    try:
        with open(file_path, 'rb') as f:
            sequences_in_file = pickle.load(f) # This is a list of sequence data
        
        if isinstance(sequences_in_file, list):
            filtered_sequences = []
            if sequences_in_file: # Check if the list is not empty
                for sequence_data in sequences_in_file:
                    if isinstance(sequence_data, list):
                        seq_len = 0
                        for inst in sequence_data:
                            seq_len += len(inst["asm"])
                        if (min_seq_len is None or seq_len >= min_seq_len) and \
                           (max_seq_len is None or seq_len < max_seq_len):
                            filtered_sequences.append(sequence_data)
                    else:
                        error_msg = f"Warning: A sequence in file {file_path} has incorrect format, skipped this sequence."
                        error_queue.put((file_path, error_msg))
                        # Print out the problematic sequence data
                        print(f"{error_msg}\nProblem data: {sequence_data}") 
                
                # Randomly sample sequences that meet the conditions
                if filtered_sequences and sample_ratio < 1.0:
                    # Set random seed to ensure reproducibility
                    sampled_count = max(1, int(len(filtered_sequences) * sample_ratio))
                    filtered_sequences = random.sample(filtered_sequences, sampled_count)
            
            result_queue.put((file_path, filtered_sequences))
        else:
            error_queue.put((file_path, f"Warning: {file_path} does not contain list data, skipped"))
    except Exception as e:
        error_queue.put((file_path, f"Error processing file {file_path}: {e}"))

def merge_token_files(directory_path: str, output_file: str, num_threads: int = None, min_length: int = None, max_length: int = None, sample_ratio: float = 1.0):
    """
    Traverse all saved_tokens.pkl files in the given directory using multiple threads,
    merge them in chunks, filter sequences by length, and save them to numbered pkl files.
    
    Args:
        directory_path (str): Directory path to traverse.
        output_file (str): Base name for the output merged files (e.g., merged_tokens.pkl).
        num_threads (int, optional): Number of threads, defaults to 2x CPU cores.
        min_length (int, optional): Minimum length of sequences to include.
        max_length (int, optional): Maximum length of sequences to include.
        sample_ratio (float, optional): Percentage of sequences to keep after filtering (0.0-1.0).
    """
    # If not specified, use CPU cores times 2 (IO intensive task)
    if num_threads is None:
        import multiprocessing
        num_threads = multiprocessing.cpu_count() * 2
    
    # Collect all files to process
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file == "saved_tokens.pkl":
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    total_input_files = len(file_paths)
    if total_input_files == 0:
        print(f"No 'saved_tokens.pkl' files found in directory {directory_path}.")
        return False
        
    print(f"Found {total_input_files} 'saved_tokens.pkl' files to merge.")
    print(f"Will randomly sample {sample_ratio * 100:.1f}% of sequences that meet length conditions for merging.")

    # Define files per chunk
    files_per_chunk = 1800
    output_file_base, output_file_ext = os.path.splitext(output_file)
    
    processed_chunks_count = 0
    total_tokens_saved_across_chunks = 0

    for i in range(0, total_input_files, files_per_chunk):
        chunk_file_paths = file_paths[i:i + files_per_chunk]
        current_chunk_total_files = len(chunk_file_paths)
        chunk_index = i // files_per_chunk
        
        print(f"\nProcessing chunk {chunk_index}, containing {current_chunk_total_files} files...")
        
        # Create result queue and error queue
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        # Use thread pool to load files in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = [executor.submit(load_file, file_path, result_queue, error_queue, min_length, max_length, sample_ratio) 
                      for file_path in chunk_file_paths]
            
            # Use tqdm to create progress bar
            with tqdm(total=current_chunk_total_files, desc=f"Loading files (chunk {chunk_index})") as pbar:
                # Wait for all tasks to complete
                while True:
                    completed = sum(future.done() for future in futures)
                    pbar.n = completed
                    pbar.refresh()
                    
                    if completed == current_chunk_total_files:
                        break
                    
                    time.sleep(0.1)
        
        # Process errors
        error_count = error_queue.qsize()
        if error_count > 0:
            print(f"Chunk {chunk_index} has {error_count} files that failed to process:")
            while not error_queue.empty():
                file_path, error_msg = error_queue.get()
                print(f"  - {error_msg}")
        
        # Merge current chunk's data
        print(f"Merging data for chunk {chunk_index}...")
        all_tokens_in_chunk = []
        processed_files_in_chunk = 0
        
        # Get all data from result queue
        while not result_queue.empty():
            file_path, tokens = result_queue.get()
            all_tokens_in_chunk.extend(tokens)
            processed_files_in_chunk += 1
        
        if not all_tokens_in_chunk:
            print(f"Chunk {chunk_index} did not successfully load any data, skipping save.")
            continue

        # Build current chunk's output filename
        current_output_file = f"{output_file_base}_{chunk_index}{output_file_ext}"
        
        # Save current chunk's merged data
        try:
            print(f"Saving {len(all_tokens_in_chunk)} merged training data entries for chunk {chunk_index} to {current_output_file}")
            with open(current_output_file, 'wb') as f:
                pickle.dump(all_tokens_in_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Chunk {chunk_index} merge completed! Successfully processed {processed_files_in_chunk}/{current_chunk_total_files} files.")
            total_tokens_saved_across_chunks += len(all_tokens_in_chunk)
            processed_chunks_count += 1
        except Exception as e:
            print(f"Error saving merged file {current_output_file} for chunk {chunk_index}: {e}")
        
        # Clean up memory
        del all_tokens_in_chunk
        gc.collect()

    if processed_chunks_count > 0:
        print(f"\nAll chunks processed! Generated {processed_chunks_count} merged files.")
        print(f"Total {total_tokens_saved_across_chunks} token data entries saved.")
        return True
    else:
        print("\nNo data chunks were successfully processed.")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge all saved_tokens.pkl files and filter sequences by length')
    parser.add_argument('directory', help='Directory path containing saved_tokens.pkl files')
    parser.add_argument('-o', '--output', default='merged_tokens.pkl', 
                        help='Output merged file path, defaults to merged_tokens.pkl in current directory')
    parser.add_argument('-j', '--jobs', type=int, default=None, 
                        help='Number of parallel processing threads, defaults to 2x CPU cores')
    parser.add_argument('--min-length', type=int, default=64,
                        help='Minimum length of sequences when merging (inclusive)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum length of sequences when merging (inclusive)')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='Proportion of sequences meeting length conditions to be retained (0.0-1.0), defaults to 1.0 (all retained)')
    
    args = parser.parse_args()
    
    # Validate sample ratio parameter
    if args.sample_ratio <= 0.0 or args.sample_ratio > 1.0:
        print(f"Error: Sample ratio must be in range (0.0, 1.0], current value: {args.sample_ratio}")
        sys.exit(1)
    
    # Set garbage collection threshold to recycle memory more aggressively
    gc.set_threshold(100, 5, 5)  # Default values are (700, 10, 10)
    
    # Merge files
    start_time = time.time()
    merge_token_files(args.directory, args.output, args.jobs, args.min_length, args.max_length, args.sample_ratio)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()