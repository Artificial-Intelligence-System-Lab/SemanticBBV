import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob


def count_sequence_length(sequence):
    """Calculate the length of a single sequence (number of opcode + operands)"""
    length = 0
    for instruction in sequence:
        # Length of each instruction = 1(opcode) + len(operands)
        length += 1 + len(instruction.get("opnd_ids", []))
    return length


def analyze_pkl_file(file_path):
    """Analyze sequence lengths in a single pkl file"""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        lengths = []
        
        # Process data structure
        for function_data in data:
            for opt_level, sequences in function_data.items():
                for sequence in sequences:
                    seq_length = count_sequence_length(sequence)
                    if seq_length > 0:  # Ignore empty sequences
                        lengths.append(seq_length)
        
        return lengths
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def find_blame_pkl_files(directory):
    """Recursively find all *_blame.pkl files in the directory"""
    pattern = os.path.join(directory, "**", "*_blame.pkl")
    return glob.glob(pattern, recursive=True)


def plot_length_distribution(lengths, output_path=None):
    """Plot sequence length distribution"""
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    plt.hist(lengths, bins=50, alpha=0.7, color='blue')
    
    # Add title and label
    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length (opcode + operands)")
    plt.ylabel("Frequency")
    
    # Add statistical information
    if lengths:
        plt.axvline(np.mean(lengths), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(lengths):.2f}')
        plt.axvline(np.median(lengths), color='g', linestyle='dashed', linewidth=1, label=f'Median: {np.median(lengths):.2f}')
        plt.legend()
    
    # Save or display chart
    if output_path:
        plt.savefig(output_path)
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()


def print_statistics(lengths):
    """Print sequence length statistics"""
    if not lengths:
        print("No valid sequences found")
        return
    
    print("\nSequence Length Statistics:")
    print(f"Total sequences: {len(lengths)}")
    print(f"Minimum length: {min(lengths)}")
    print(f"Maximum length: {max(lengths)}")
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"Standard deviation: {np.std(lengths):.2f}")
    
    # Calculate quantiles
    percentiles = [5, 25, 50, 75, 95, 99]
    for p in percentiles:
        print(f"{p} percentile: {np.percentile(lengths, p):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sequence length distribution in *_blame.pkl files")
    parser.add_argument("directory", help="Directory path to search")
    parser.add_argument("--plot", "-p", help="Path to save chart", default=None)
    args = parser.parse_args()
    
    # Find all matching files
    pkl_files = find_blame_pkl_files(args.directory)
    
    if not pkl_files:
        print(f"No *_blame.pkl files found in {args.directory} directory")
        sys.exit(1)
    
    print(f"Found {len(pkl_files)} *_blame.pkl files")
    
    # Analyze all files
    all_lengths = []
    for file_path in pkl_files:
        print(f"Analyzing: {file_path}")
        lengths = analyze_pkl_file(file_path)
        all_lengths.extend(lengths)
    
    if not all_lengths:
        print("Failed to extract any valid sequence lengths from file")
        sys.exit(1)
    
    # Print statistics
    print_statistics(all_lengths)
    
    # Plot distribution
    plot_length_distribution(all_lengths, args.plot)


if __name__ == "__main__":
    main()