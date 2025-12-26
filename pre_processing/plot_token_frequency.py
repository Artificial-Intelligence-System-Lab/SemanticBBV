#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import Counter

def load_token_dict(file_path):
    """
    Load token dictionary file
    """
    try:
        with open(file_path, 'rb') as f:
            token_dict = pickle.load(f)
        return token_dict
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return {}

def plot_frequency_distribution(token_dict, title, output_path=None, log_scale=True, top_n=None):
    """
    Plot token frequency distribution
    
    Parameters:
        token_dict: token dictionary, key is token, value is occurrence count
        title: chart title
        output_path: output image path, if None then display image
        log_scale: whether to use log scale
        top_n: only show top N most common tokens
    """
    if not token_dict:
        print(f"Warning: {title} token dictionary is empty")
        return
    
    # Get frequency list
    frequencies = list(token_dict.values())
    
    # If top_n is specified, only take the top N most common tokens
    if top_n is not None and top_n < len(frequencies):
        frequencies = sorted(frequencies, reverse=True)[:top_n]
    
    # Create frequency counter
    freq_counter = Counter(frequencies)
    
    # Sort for plotting
    sorted_freqs = sorted(freq_counter.keys())
    counts = [freq_counter[f] for f in sorted_freqs]
    
    # Create chart
    plt.figure(figsize=(12, 8))
    
    # Plot frequency distribution
    plt.bar(range(len(sorted_freqs)), counts, align='center')
    plt.xticks(range(len(sorted_freqs)), sorted_freqs, rotation=90)
    
    # Set log scale (if needed)
    if log_scale:
        plt.yscale('log')
    
    # Set title and labels
    plt.title(f"{title} - Frequency Distribution")
    plt.xlabel("Occurrence Count")
    plt.ylabel("Number of Tokens")
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the chart
    if output_path:
        plt.savefig(output_path, format='pdf', dpi=300)
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_cumulative_distribution(token_dict, title, output_path=None):
    """
    Plot cumulative distribution chart, showing token coverage
    
    Parameters:
        token_dict: token dictionary, key is token, value is occurrence count
        title: chart title
        output_path: output image path, if None then display image
    """
    if not token_dict:
        print(f"Warning: {title} token dictionary is empty")
        return
    
    # Get frequency list and sort
    frequencies = sorted(token_dict.values(), reverse=True)
    total_occurrences = sum(frequencies)
    
    # Calculate cumulative coverage
    cumulative = np.cumsum(frequencies)
    coverage = cumulative / total_occurrences * 100
    
    # Create chart
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative distribution
    plt.plot(range(1, len(frequencies) + 1), coverage, 'b-')
    
    # Add reference lines
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% Coverage')
    plt.axhline(y=95, color='g', linestyle='--', alpha=0.7, label='95% Coverage')
    plt.axhline(y=99, color='y', linestyle='--', alpha=0.7, label='99% Coverage')
    plt.axhline(y=99.9, color='m', linestyle='--', alpha=0.7, label='99.9% Coverage')
    
    # Find the number of tokens required to reach specific coverage
    tokens_90 = np.searchsorted(coverage, 90) + 1
    tokens_95 = np.searchsorted(coverage, 95) + 1
    tokens_99 = np.searchsorted(coverage, 99) + 1
    tokens_999 = np.searchsorted(coverage, 99.9) + 1
    
    # Add annotations
    plt.annotate(f'90%: {tokens_90} tokens', 
                xy=(tokens_90, 90), 
                xytext=(tokens_90 + len(frequencies) * 0.05, 90 - 5),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'95%: {tokens_95} tokens', 
                xy=(tokens_95, 95), 
                xytext=(tokens_95 + len(frequencies) * 0.05, 95 - 5),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'99%: {tokens_99} tokens', 
                xy=(tokens_99, 99), 
                xytext=(tokens_99 + len(frequencies) * 0.05, 99 - 5),
                arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'99.9%: {tokens_999} tokens', 
                xy=(tokens_999, 99.9), 
                xytext=(tokens_999 + len(frequencies) * 0.05, 99.9 - 5),
                arrowprops=dict(arrowstyle='->'))
    
    # Set title and labels
    plt.title(f"{title} - Cumulative Coverage")
    plt.xlabel("Number of Tokens (Sorted by Frequency)")
    plt.ylabel("Cumulative Coverage (%)")
    plt.legend()
    
    # Set x-axis formatter to display scientific notation for large numbers
    def format_fn(x, pos):
        if x >= 1000:
            return f'{x/1000:.0f}k'
        return str(int(x))
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_fn))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the chart
    if output_path:
        plt.savefig(output_path, format='pdf', dpi=300)
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_token_stats(token_dict, title):
    """
    Print token statistics
    """
    if not token_dict:
        print(f"Warning: {title} token dictionary is empty")
        return
    
    # Get frequency list and sort
    frequencies = sorted(token_dict.values(), reverse=True)
    total_tokens = len(token_dict)
    total_occurrences = sum(frequencies)
    
    # Calculate statistics
    min_freq = min(frequencies)
    max_freq = max(frequencies)
    avg_freq = total_occurrences / total_tokens
    
    # Calculate the number of tokens under different frequency thresholds
    tokens_freq_1 = sum(1 for f in frequencies if f >= 1)
    tokens_freq_5 = sum(1 for f in frequencies if f >= 5)
    tokens_freq_10 = sum(1 for f in frequencies if f >= 10)
    tokens_freq_100 = sum(1 for f in frequencies if f >= 100)
    
    # Print statistics
    print(f"\n===== {title} Statistics =====")
    print(f"Total token count: {total_tokens}")
    print(f"Total occurrences: {total_occurrences}")
    print(f"Minimum frequency: {min_freq}")
    print(f"Maximum frequency: {max_freq}")
    print(f"Average frequency: {avg_freq:.2f}")
    print(f"Tokens with frequency >= 1: {tokens_freq_1} ({tokens_freq_1/total_tokens*100:.2f}%)")
    print(f"Tokens with frequency >= 5: {tokens_freq_5} ({tokens_freq_5/total_tokens*100:.2f}%)")
    print(f"Tokens with frequency >= 10: {tokens_freq_10} ({tokens_freq_10/total_tokens*100:.2f}%)")
    print(f"Tokens with frequency >= 100: {tokens_freq_100} ({tokens_freq_100/total_tokens*100:.2f}%)")
    
    # Calculate cumulative coverage
    cumulative = np.cumsum(frequencies)
    coverage = cumulative / total_occurrences * 100
    
    # Find the number of tokens required to reach specific coverage
    tokens_90 = np.searchsorted(coverage, 90) + 1
    tokens_95 = np.searchsorted(coverage, 95) + 1
    tokens_99 = np.searchsorted(coverage, 99) + 1
    tokens_999 = np.searchsorted(coverage, 99.9) + 1
    
    print(f"\nCoverage statistics:")
    print(f"Tokens required for 90% coverage: {tokens_90} ({tokens_90/total_tokens*100:.2f}%)")
    print(f"Tokens required for 95% coverage: {tokens_95} ({tokens_95/total_tokens*100:.2f}%)")
    print(f"Tokens required for 99% coverage: {tokens_99} ({tokens_99/total_tokens*100:.2f}%)")
    print(f"Tokens required for 99.9% coverage: {tokens_999} ({tokens_999/total_tokens*100:.2f}%)")
    
    # Print the 10 tokens with the least occurrences
    print(f"\nThe 10 tokens with the least occurrences:")
    # Sort tokens by occurrence count
    sorted_tokens = sorted(token_dict.items(), key=lambda x: x[1])
    # Take the first 10 (or all, if less than 10)
    least_common = sorted_tokens[:min(10, len(sorted_tokens))]
    for i, (token, count) in enumerate(least_common):
        # For string type tokens, print their content; for other types, print directly
        if isinstance(token, str):
            token_str = token
        else:
            token_str = str(token)
        print(f"{i+1}. Token: {token_str} - Occurrences: {count}")

def main():
    parser = argparse.ArgumentParser(description='Plot token frequency distribution chart')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory where token dictionary files are located')
    parser.add_argument('--output_dir', type=str, default=None, help='Output image directory, default is input_dir/plots')
    parser.add_argument('--log_scale', action='store_true', default=True, help='Use logarithmic scale (enabled by default)')
    parser.add_argument('--top_n', type=int, default=None, help='Only show the top N most common tokens')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'plots')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths to load
    token_files = {
        "Assembly tokens": os.path.join(args.input_dir, "assemble_tokens_dict.pkl"),
        "Mnemonic tokens": os.path.join(args.input_dir, "mnemonic_tokens_dict.pkl"),
        "Operand type tokens": os.path.join(args.input_dir, "op_kind_tokens_dict.pkl"),
        "Operand ID tokens": os.path.join(args.input_dir, "op_id_tokens_dict.pkl"),
        "eflags tokens": os.path.join(args.input_dir, "eflags_tokens_dict.pkl")
    }
    
    # Process each token dictionary
    for title, file_path in token_files.items():
        if os.path.exists(file_path):
            print(f"Processing {title}...")
            token_dict = load_token_dict(file_path)
            
            # Print statistics
            print_token_stats(token_dict, title)
            
            # English title mapping
            english_titles = {
                "Assembly tokens": "Assembly Tokens",
                "Mnemonic tokens": "Mnemonic Tokens",
                "Operand type tokens": "Operand Type Tokens",
                "Operand ID tokens": "Operand ID Tokens",
                "eflags tokens": "EFLAGS Tokens"
            }
            
            # Use English title
            english_title = english_titles.get(title, title)
            
            # Plot frequency distribution chart
            freq_output_path = os.path.join(args.output_dir, f"{title.replace(' ', '_')}_frequency_distribution.pdf")
            plot_frequency_distribution(token_dict, english_title, freq_output_path, args.log_scale, args.top_n)
            
            # Plot cumulative distribution chart
            cum_output_path = os.path.join(args.output_dir, f"{title.replace(' ', '_')}_cumulative_coverage.pdf")
            plot_cumulative_distribution(token_dict, english_title, cum_output_path)
        else:
            print(f"Warning: File {file_path} does not exist")

if __name__ == "__main__":
    main()