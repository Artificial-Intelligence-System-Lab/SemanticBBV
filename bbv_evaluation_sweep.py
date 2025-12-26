#!/usr/bin/env python3
import os
import subprocess
import re
import argparse
from tqdm import tqdm
from collections import defaultdict

def parse_metric(output, pattern, metric_name):
    """
    Use regular expressions to parse individual metrics from output text.

    Args:
        output (str): The model's standard output.
        pattern (str): The regular expression for matching.
        metric_name (str): The name of the metric, used for printing warnings.

    Returns:
        float or None: The parsed float value, or None if not found.
    """
    match = re.search(pattern, output)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            print(f"Warning: Found match for '{metric_name}' but could not convert to float.")
            return None
    return None

def run_evaluation(checkpoint_path, data_path, with_cpi=True):
    """
    Run a single evaluation command and parse the output of all key metrics.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        data_path (str): Path to the evaluation dataset.
        with_cpi (bool): Whether to enable CPI task.

    Returns:
        dict or None: Dictionary containing all parsed metrics, or None if execution fails.
    """
    command = [
        "python", "evaluate_bbv_model.py",
        "--model_path", checkpoint_path,
        "--data_path", data_path
    ]
    if with_cpi:
        command.append("--with_cpi")
        command.append("True")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        output = result.stdout
        metrics = {}
        
        # Define all metrics that need to be parsed and their regular expressions
        metric_patterns = {
            'avg_triplet_loss': r"Average triplet loss:\s*([0-9.]+)",
            'avg_pos_distance': r"Average positive distance:\s*([0-9.]+)",
            'avg_neg_distance': r"Average negative distance:\s*([0-9.]+)",
            'margin_violation_rate': r"Margin violation rate:\s*([0-9.]+)",
            'distance_separation': r"Distance separation:\s*([0-9.]+)",
            'avg_cpi_loss': r"Average CPI loss:\s*([0-9.]+)",
        }
        
        for name, pattern in metric_patterns.items():
            value = parse_metric(output, pattern, name)
            if value is not None:
                metrics[name] = value

        # If no metrics could be parsed, consider this run problematic
        if not metrics:
            print(f"Warning: Unable to parse any metrics from the output of {checkpoint_path}.")
            return None

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"\nError: An error occurred while running {checkpoint_path}.")
        print("--- STDERR ---")
        print(e.stderr)
        print("--------------")
        return None
    except Exception as e:
        print(f"\nUnknown error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Automatically evaluate a series of model checkpoints and sort by multiple metrics.")
    parser.add_argument("--checkpoints_dir", default="./bbv_checkpoints", help="Root directory containing all checkpoint-* directories.")
    parser.add_argument("--start_step", type=int, default=1000, help="Starting step for evaluation.")
    parser.add_argument("--end_step", type=int, default=3200, help="Ending step for evaluation.")
    parser.add_argument("--step_interval", type=int, default=100, help="Step interval between checkpoints.")
    parser.add_argument("--data_path", default="./dataset/bbv_test_cpi_dataset.pkl", help="Path to the evaluation dataset.")
    parser.add_argument("--with_cpi", action='store_true', help="Include --with_cpi True parameter in the evaluation command.")
    
    args = parser.parse_args()

    # Find all eligible checkpoints
    checkpoints_to_eval = []
    for step in range(args.start_step, args.end_step + 1, args.step_interval):
        checkpoint_dir = os.path.join(args.checkpoints_dir, f"checkpoint-{step}")
        model_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_file):
            checkpoints_to_eval.append(model_file)

    if not checkpoints_to_eval:
        print("Error: No checkpoints found in the specified range. Please check the path and step range.")
        return

    print(f"Found {len(checkpoints_to_eval)} checkpoints for evaluation, from {args.start_step} to {args.end_step}.")

    # Run evaluation and collect all results
    # Structure: [(ckpt_path, {'metric1': val1, 'metric2': val2, ...}), ...]
    all_results = []
    for ckpt_path in tqdm(checkpoints_to_eval, desc="Evaluating all checkpoints"):
        print(f"\n--- Evaluating: {ckpt_path} ---")
        metrics = run_evaluation(ckpt_path, args.data_path, args.with_cpi)
        if metrics:
            all_results.append((ckpt_path, metrics))

    # Define metrics and their sorting methods
    # 'asc' = lower is better, 'desc' = higher is better
    metrics_to_report = {
        "Average triplet loss (avg_triplet_loss)": ('avg_triplet_loss', 'asc'),
        "Average positive distance (avg_pos_distance)": ('avg_pos_distance', 'asc'),
        "Margin violation rate (margin_violation_rate)": ('margin_violation_rate', 'asc'),
        "Average CPI loss (avg_cpi_loss)": ('avg_cpi_loss', 'asc'),
        "Average negative distance (avg_neg_distance)": ('avg_neg_distance', 'desc'),
        "Distance separation (distance_separation)": ('distance_separation', 'desc'),
    }

    print("\n\n" + "="*80)
    print("           EVALUATION RESULTS SUMMARY")
    print("="*80)

    # Iterate through each metric, sort and report
    for display_name, (metric_key, sort_order) in metrics_to_report.items():
        
        # Filter out results that do not have this metric
        filtered_results = [(path, metrics) for path, metrics in all_results if metric_key in metrics]
        
        if not filtered_results:
            print(f"\n--- No evaluation results found for metric '{display_name}' ---")
            continue

        # Sort based on current metric and sort order
        reverse = (sort_order == 'desc')
        filtered_results.sort(key=lambda x: x[1][metric_key], reverse=reverse)

        print(f"\n--- Top 3 Best Models (sorted by {display_name}) ---")
        for i, (path, metrics) in enumerate(filtered_results[:3]):
            print(f"  --- Top {i+1} ---")
            print(f"    Model path: {path}")
            print(f"    Metric value: {metrics[metric_key]:.4f}")
            # Print other relevant metrics for reference
            print("    Other metrics:")
            for name, val in metrics.items():
                if name != metric_key:
                    print(f"      - {name}: {val:.4f}")
    
    if not all_results:
        print("Failed to successfully evaluate any models.")

if __name__ == "__main__":
    main()