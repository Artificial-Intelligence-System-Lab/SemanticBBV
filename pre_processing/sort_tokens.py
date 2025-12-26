#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import sys

def load_and_sort_tokens(input_file, output_file):
    """
    Load tokens dictionary file, extract keys and sort, then write to text file
    
    Parameters:
        input_file: Input pickle file path
        output_file: Output text file path
    """
    try:
        # Load pickle file
        with open(input_file, 'rb') as f:
            tokens_dict = pickle.load(f)
        
        # Extract keys and sort
        sorted_keys = sorted(tokens_dict.keys())
        
        # Write sorted keys to text file
        with open(output_file, 'w', encoding='utf-8') as f:
            for key in sorted_keys:
                f.write(f"{key}\n")
        
        print(f"Successfully extracted and sorted {len(sorted_keys)} tokens from {input_file}")
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage: python sort_tokens.py <input_pickle_file> <output_text_file>\n")
        print("Example: python sort_tokens.py tokens_fine/assemble_tokens_dict.pkl sorted_tokens.txt\n")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    load_and_sort_tokens(input_file, output_file)