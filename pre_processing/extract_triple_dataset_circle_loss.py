import os
import pickle
import argparse
import time
from tqdm import tqdm
import gc
import random

def merge_pickle_files(directory_path, output_file, file_pattern="*.pkl"):
    """
    Merge pickle files matching a specific pattern in the specified directory
    
    Args:
        directory_path (str): Directory path containing pickle files
        output_file (str): Output merged file path
        file_pattern (str): File name matching pattern, defaults to "train_dataset_*.pkl"
        
    Returns:
        bool: Whether the operation was successful
    """
    import glob
    
    # Get all matching files
    file_pattern_path = os.path.join(directory_path, file_pattern)
    file_paths = glob.glob(file_pattern_path)
    
    if not file_paths:
        print(f"No files matching '{file_pattern}' found in directory {directory_path}.")
        return False
    
    file_paths.sort()  # Ensure files are sorted by name
    print(f"Found {len(file_paths)} files to merge:")
    for file_path in file_paths:
        print(f"  - {os.path.basename(file_path)}")
    
    # Merge all data
    all_data = []
    total_sequences = 0
    
    for file_path in tqdm(file_paths, desc="Merging files"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # If data is dict type, only take its values list
                if isinstance(data, dict):
                    data = list(data.values())
                
                if isinstance(data, list):
                    # Randomly select 2000 indices from data
                    sampled_length = int(len(data) * 0.10)
                    indices = random.sample(range(len(data)), sampled_length)
                    for idx in indices:
                        positives = data[idx]
                        negative_indices = random.sample(range(len(data)), len(positives)*2 + 1)
                        negative_indices = [i for i in negative_indices if i != idx]
                        if len(negative_indices) > len(positives) * 2:
                            negative_indices.pop()
                        negatives = [random.choice(data[i]) for i in negative_indices]
                        all_data.append((positives, negatives))
                        total_sequences += 1
                else:
                    print(f"Warning: {file_path} does not contain list data, skipped")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if not all_data:
        print("No data was successfully loaded, cannot save merged file.")
        return False
    
    # Save merged data
    try:
        print(f"Saving {len(all_data)} merged data entries to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Merge completed! Total {total_sequences} sequences merged.")
        return True
    except Exception as e:
        print(f"Error saving merged file {output_file}: {e}")
        return False

def main():
    """
    Main function, parse command line arguments and execute merge operation
    """
    parser = argparse.ArgumentParser(description='Merge pickle files in the specified directory')
    parser.add_argument('directory', help='Directory path containing pickle files')
    parser.add_argument('-o', '--output', default='merged_dataset.pkl', 
                        help='Output merged file path, defaults to merged_dataset.pkl in current directory')
    parser.add_argument('-p', '--pattern', default='*.pkl',
                        help='File name matching pattern, defaults to "*.pkl"')
    
    args = parser.parse_args()
    
    # Set garbage collection threshold, more aggressive memory reclamation
    gc.set_threshold(100, 5, 5)  # Default values are (700, 10, 10)
    
    # Merge files
    start_time = time.time()
    merge_pickle_files(args.directory, args.output, args.pattern)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()