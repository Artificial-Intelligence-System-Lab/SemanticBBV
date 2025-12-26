#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
from typing import Set, List

special_tokens = ["[PAD]"]

def load_pickle_file(file_path: str) -> Set:
    """
    Load pickle file and return its content
    
    Args:
        file_path: pickle file path
        
    Returns:
        Loaded pickle content
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def export_tokens_to_txt(tokens: Set, output_path: str) -> None:
    """
    Export token set to sorted txt file, and remove [PAD] token
    
    Args:
        tokens: token set
        output_path: output file path
    """
    # Convert to list for sorting
    token_list = list(tokens)
    
    # Remove [PAD] token
    if "PAD" in token_list:
        token_list.remove("PAD")
    
    if "[PAD]" in token_list:
        token_list.remove("[PAD]")
    
    # Sort
    token_list.sort()
    token_list = special_tokens + token_list
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in token_list:
            f.write(f"{token}\n")
    
    print(f"Exported {len(token_list)} tokens to {output_path}")

def process_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Process all non *_dict.pkl files in the directory
    
    Args:
        input_dir: input directory path
        output_dir: output directory path, defaults to same as input directory
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pkl files
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl') and not f.endswith('_dict.pkl')]
    
    if not pkl_files:
        print(f"No non *_dict.pkl files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} pkl files to process")
    
    # Process each file
    for pkl_file in pkl_files:
        input_path = os.path.join(input_dir, pkl_file)
        
        # Build output file path, replace .pkl with .txt
        base_name = os.path.splitext(pkl_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        try:
            # Load token set
            tokens = load_pickle_file(input_path)
            
            # Export to txt
            export_tokens_to_txt(tokens, output_path)
        except Exception as e:
            print(f"Error processing file {pkl_file}: {e}")

def main():
    """
    Main function, parse command line arguments and execute processing
    """
    parser = argparse.ArgumentParser(description='Export token sets to sorted txt files')
    parser.add_argument('input_dir', help='Input directory path containing pkl files')
    parser.add_argument('--output-dir', '-o', help='Output directory path, defaults to same as input directory')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()