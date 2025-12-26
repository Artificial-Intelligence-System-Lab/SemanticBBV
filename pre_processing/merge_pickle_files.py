import os
import pickle
import argparse
import time
from tqdm import tqdm
import gc

def read_file_paths_from_txt(txt_file_path):
    """
    Read file path list from txt file
    
    Args:
        txt_file_path (str): Path to txt file containing file paths
        
    Returns:
        list: List of file paths
    """
    file_paths = []
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comment lines
                    # Handle relative paths, convert to absolute paths
                    if not os.path.isabs(line):
                        # Path relative to the directory where txt file is located
                        txt_dir = os.path.dirname(os.path.abspath(txt_file_path))
                        line = os.path.join(txt_dir, line)
                    file_paths.append(line)
        print(f"Read {len(file_paths)} file paths from {txt_file_path}")
        return file_paths
    except Exception as e:
        print(f"Error reading file path list {txt_file_path}: {e}")
        return []

def merge_pickle_files(directory_path=None, output_file="merged_dataset.pkl", file_pattern="*.pkl", merge_type="list", file_list_txt=None):
    """
    Merge pickle files matching a specific pattern in the specified directory, or merge pickle files specified in a txt file
    
    Args:
        directory_path (str): Directory path containing pickle files (used when file_list_txt is None)
        output_file (str): Output merged file path
        file_pattern (str): Filename matching pattern, defaults to "*.pkl" (only valid when using directory_path)
        merge_type (str): Merge type, options are "list" or "dict", defaults to "list"
        file_list_txt (str): Path to txt file containing file path list
        
    Returns:
        bool: Whether the operation was successful
    """
    import glob
    
    # Get file path list
    if file_list_txt:
        # Read file paths from txt file
        file_paths = read_file_paths_from_txt(file_list_txt)
        if not file_paths:
            return False
    else:
        # Get matching files from directory
        if not directory_path:
            print("Error: Must specify directory path or file list txt file")
            return False
            
        file_pattern_path = os.path.join(directory_path, file_pattern)
        file_paths = glob.glob(file_pattern_path)
        
        if not file_paths:
            print(f"No files matching '{file_pattern}' found in directory {directory_path}.")
            return False
    
    # Validate file existence
    valid_file_paths = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            valid_file_paths.append(file_path)
        else:
            print(f"Warning: File does not exist, skipped: {file_path}")
    
    if not valid_file_paths:
        print("No valid files found")
        return False
    
    valid_file_paths.sort()  # Ensure files are sorted by name
    print(f"Found {len(valid_file_paths)} valid files to merge:")
    for file_path in valid_file_paths:
        print(f"  - {os.path.basename(file_path)}")
    
    # Initialize data structure based on merge type
    if merge_type == "list":
        all_data = []
    elif merge_type == "dict":
        all_data = {}
    else:
        print(f"Unsupported merge type: {merge_type}, supported types are 'list' or 'dict'")
        return False
    
    total_items = 0
    
    for file_path in tqdm(valid_file_paths, desc="Merging files"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                if merge_type == "list":
                    if isinstance(data, list):
                        all_data.extend(data)
                        total_items += len(data)
                    else:
                        print(f"Warning: {file_path} does not contain list data, skipped")
                elif merge_type == "dict":
                    if isinstance(data, dict):
                        all_data.update(data)
                        total_items += len(data)
                    else:
                        print(f"Warning: {file_path} does not contain dict data, skipped")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if (merge_type == "list" and not all_data) or (merge_type == "dict" and not all_data):
        print("No data was successfully loaded, cannot save merged file.")
        return False
    
    # Save merged data
    try:
        print(f"Saving {len(all_data)} merged data entries to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Merge completed! Total {total_items} items merged.")
        return True
    except Exception as e:
        print(f"Error saving merged file {output_file}: {e}")
        return False

def main():
    """
    Main function, parse command line arguments and execute merge operation
    """
    parser = argparse.ArgumentParser(description='Merge pickle files in specified directory or pickle files specified in txt file')
    
    # Create mutually exclusive group to ensure only one input method can be selected
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('directory', nargs='?', help='Directory path containing pickle files')
    input_group.add_argument('-f', '--file-list', dest='file_list_txt', 
                           help='Path to txt file containing file path list, one file path per line')
    
    parser.add_argument('-o', '--output', default='merged_dataset.pkl', 
                        help='Output merged file path, defaults to merged_dataset.pkl in current directory')
    parser.add_argument('-p', '--pattern', default='*.pkl',
                        help='Filename matching pattern, defaults to "*.pkl" (only valid in directory mode)')
    parser.add_argument('-t', '--type', default='list', choices=['list', 'dict'],
                        help='Merge type, options are "list" or "dict", defaults to "list"')
    
    args = parser.parse_args()
    
    # Set garbage collection threshold to recycle memory more aggressively
    gc.set_threshold(100, 5, 5)  # Default values are (700, 10, 10)
    
    # Merge files
    start_time = time.time()
    
    if args.file_list_txt:
        # Use file list from txt file
        success = merge_pickle_files(
            output_file=args.output, 
            merge_type=args.type,
            file_list_txt=args.file_list_txt
        )
    else:
        # Use directory mode
        success = merge_pickle_files(
            directory_path=args.directory, 
            output_file=args.output, 
            file_pattern=args.pattern, 
            merge_type=args.type
        )
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    if success:
        print("Merge operation completed successfully!")
    else:
        print("Merge operation failed!")
        exit(1)

if __name__ == "__main__":
    main()