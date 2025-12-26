#!/usr/bin/env python3
"""
CPI prediction evaluation script
Use Set-Transformer model to predict CPI and evaluate accuracy
Combine data reading from simpoint_experiment.py and model prediction functionality from evaluate_bbv_model.py
"""

import argparse
import gzip
import os
import pickle
import random
import re
import sys
import json
import traceback
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, 'Set-Transformer'))

from evaluate_bbv_model import SetTransformerModel

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_seed(seed):
    """
    Set global random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global random seed set to: {seed}")

def load_map(path: str):
    """
    Load mapping file
    
    Args:
        path: Mapping file path
        
    Returns:
        Mapping dictionary
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_reference_stats(ref_json_path: str):
    """
    Load reference simulation statistics data
    
    Args:
        ref_json_path: Reference JSON file path
        
    Returns:
        Dictionary containing simulation statistics data
    """
    with open(ref_json_path, 'r') as f:
        return json.load(f)

def vectors_mapping(input_gz: str, map_file: str):
    """
    Extract vectors and embedding mappings from compressed BBV file
    
    Args:
        input_gz: Path to compressed BBV input file
        map_file: Mapping file path
        
    Returns:
        tuple: (vectors, vector_embs) - Vector array and embedding mapping list
    """
    pat = re.compile(r":(\d+):(\d+)")
    bbvs = []
    
    with gzip.open(input_gz, 'rt', encoding='utf-8', errors='replace') as inp:
        for line in inp:
            line = line.strip()
            if not line or not line.startswith('T'):
                continue
            d = {int(k): int(v) for k, v in pat.findall(line)}
            bbvs.append(d)
    
    if len(bbvs) < 1:
        raise RuntimeError("Too few samples")
    
    # Create random projection matrix
    max_id = max(k for d in bbvs for k in d)
    proj = np.random.randn(max_id + 1, 100)
    
    # Generate vector representation
    vectors = np.vstack([
        np.array([d.get(i, 0) for i in range(max_id + 1)]).dot(proj) 
        for d in bbvs
    ])
    
    # Load ID mapping
    id_map = load_map(map_file)
    
    # Generate vector embeddings
    vector_embs = []
    for d in bbvs:
        items = sorted(d.items(), key=lambda kv: kv[1])
        keys, ws = zip(*items) if items else ([], [])
        
        # Ensure index is in range
        embs = []
        weights = []
        for k, w in zip(keys, ws):
            if k - 1 < len(id_map):
                embs.append(id_map[k - 1])
                weights.append(w)
        
        if embs:  # Only add if valid embeddings exist
            vector_embs.append((embs, weights))
        else:
            # If no valid embeddings, add default zero vector
            vector_embs.append(([np.zeros(128)], [1]))
    
    return vectors, vector_embs

def build_and_load_model(args):
    """
    Build and load Set-Transformer model
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (model, device) - Loaded model and device
    """
    device = torch.device(args.device)
    model = SetTransformerModel(args)
    model = model.to(device)
    
    # Load model weights
    ckpt = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    
    # If not using CPI, filter out CPI-related weights
    if not args.with_cpi:
        sd = {k: v for k, v in sd.items() if not k.startswith('cpi_head.')}
    
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    return model, device

def predict_cpi_for_intervals(model, device, vector_embs, batch_size=32):
    """
    Use Set-Transformer model to predict CPI for each interval
    
    Args:
        model: Trained Set-Transformer model
        device: Computing device
        vector_embs: List of vector embeddings
        batch_size: Batch size
        
    Returns:
        numpy.ndarray: Predicted CPI value array
    """
    model.eval()
    predicted_cpis = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(vector_embs), batch_size), desc="Predict CPI"):
            batch_embs = vector_embs[i:i + batch_size]
            
            # Prepare batch data
            batch_embeddings = []
            batch_weights = []
            
            for embs, weights in batch_embs:
                # Ensure embedding is numpy array
                if isinstance(embs[0], list):
                    embs = [np.array(emb) for emb in embs]
                elif not isinstance(embs[0], np.ndarray):
                    embs = [np.array(emb) for emb in embs]
                
                batch_embeddings.append(torch.tensor(np.array(embs), dtype=torch.float32))
                batch_weights.append(torch.tensor(weights, dtype=torch.float32))
            
            # Move to device
            batch_embeddings = [emb.to(device) for emb in batch_embeddings]
            batch_weights = [w.to(device) for w in batch_weights]
            
            # Model prediction
            try:
                batch_cpis = model.predict_cpi(batch_embeddings, batch_weights)
                if batch_cpis.dim() > 1:
                    batch_cpis = batch_cpis.squeeze()
                
                predicted_cpis.extend(batch_cpis.cpu().numpy())
                
            except Exception as e:
                print(f"Error predicting batch {i//batch_size + 1}: {e}")
                # If prediction fails, use default values
                predicted_cpis.extend([1.0] * len(batch_embs))
    
    return np.array(predicted_cpis)

def calculate_true_cpi(ref_stats):
    """
    Calculate true CPI from reference statistics data
    
    Args:
        ref_stats: Reference statistics data dictionary
        
    Returns:
        numpy.ndarray: True CPI value array
    """
    insts = np.array(ref_stats["simInsts"])
    cycles = np.array(ref_stats["board.processor.start.core.numCycles"])
    
    # Calculate number of instructions for each interval (difference)
    if len(insts) > 1:
        inst_counts = np.diff(insts, prepend=0)
    else:
        inst_counts = insts
    
    # Avoid division by zero
    inst_counts[inst_counts == 0] = 1
    
    # Calculate CPI
    cpi_per_interval = cycles / inst_counts
    
    return cpi_per_interval

def evaluate_prediction_accuracy(true_cpi, predicted_cpi):
    """
    Evaluate CPI prediction accuracy
    
    Args:
        true_cpi: True CPI values
        predicted_cpi: Predicted CPI values
        
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    # Ensure array length consistency
    min_len = min(len(true_cpi), len(predicted_cpi))
    true_cpi = true_cpi[:min_len]
    predicted_cpi = predicted_cpi[:min_len]
    
    # Calculate various evaluation metrics
    mae = mean_absolute_error(true_cpi, predicted_cpi)
    mse = mean_squared_error(true_cpi, predicted_cpi)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_cpi, predicted_cpi)
    
    # Calculate relative error
    relative_errors = np.abs(predicted_cpi - true_cpi) / (true_cpi + 1e-8)
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)
    
    # Calculate accuracy (proportion of predictions within certain threshold)
    accuracy_5pct = np.mean(relative_errors < 0.05)  # 5% threshold
    accuracy_10pct = np.mean(relative_errors < 0.10)  # 10% threshold
    accuracy_20pct = np.mean(relative_errors < 0.20)  # 20% threshold
    
    # Calculate Pearson correlation coefficient
    correlation, p_value = stats.pearsonr(true_cpi, predicted_cpi)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error,
        'accuracy_5pct': accuracy_5pct,
        'accuracy_10pct': accuracy_10pct,
        'accuracy_20pct': accuracy_20pct,
        'correlation': correlation,
        'correlation_p_value': p_value,
        'relative_errors': relative_errors
    }

def plot_prediction_results(true_cpi, predicted_cpi, metrics, output_dir, program_name=None):
    """
    Plot visualization charts for prediction results
    
    Args:
        true_cpi: True CPI values
        predicted_cpi: Predicted CPI values
        metrics: Dictionary of evaluation metrics
        output_dir: Output directory
        program_name: Program name, for generating file names
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: predicted vs true values
    axes[0, 0].scatter(true_cpi, predicted_cpi, alpha=0.6, s=20)
    axes[0, 0].plot([true_cpi.min(), true_cpi.max()], [true_cpi.min(), true_cpi.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True CPI')
    axes[0, 0].set_ylabel('Predicted CPI')
    axes[0, 0].set_title(f'Predicted vs True CPI\nR² = {metrics["r2_score"]:.4f}, Correlation = {metrics["correlation"]:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series plot: predicted and true values comparison
    sample_indices = np.arange(len(true_cpi))
    axes[0, 1].plot(sample_indices, true_cpi, label='True CPI', alpha=0.8, linewidth=1)
    axes[0, 1].plot(sample_indices, predicted_cpi, label='Predicted CPI', alpha=0.8, linewidth=1)
    axes[0, 1].set_xlabel('Interval Index')
    axes[0, 1].set_ylabel('CPI')
    axes[0, 1].set_title('CPI Prediction Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Relative error distribution histogram
    axes[1, 0].hist(metrics['relative_errors'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(metrics['mean_relative_error'], color='red', linestyle='--', 
                      label=f'Mean: {metrics["mean_relative_error"]:.4f}')
    axes[1, 0].axvline(metrics['median_relative_error'], color='orange', linestyle='--', 
                      label=f'Median: {metrics["median_relative_error"]:.4f}')
    axes[1, 0].set_xlabel('Relative Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Relative Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy bar chart
    accuracy_labels = ['5%', '10%', '20%']
    accuracy_values = [metrics['accuracy_5pct'], metrics['accuracy_10pct'], metrics['accuracy_20pct']]
    bars = axes[1, 1].bar(accuracy_labels, accuracy_values, alpha=0.7, color=['green', 'blue', 'orange'])
    axes[1, 1].set_xlabel('Error Threshold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Prediction Accuracy at Different Thresholds')
    axes[1, 1].set_ylim(0, 1)
    
    # Add numeric labels on bar chart
    for bar, value in zip(bars, accuracy_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart - generate file name using program name
    if program_name:
        output_path = os.path.join(output_dir, f'{program_name}_cpi_prediction_evaluation.pdf')
    else:
        output_path = os.path.join(output_dir, 'cpi_prediction_evaluation.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization results saved to: {output_path}")
    
    plt.show()

def save_results(metrics, true_cpi, predicted_cpi, output_dir, program_name):
    """
    Save evaluation results to file
    
    Args:
        metrics: Dictionary of evaluation metrics
        true_cpi: True CPI values
        predicted_cpi: Predicted CPI values
        output_dir: Output directory
        program_name: Program name
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results = {
        'program_name': program_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mae': float(metrics['mae']),
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'r2_score': float(metrics['r2_score']),
            'mean_relative_error': float(metrics['mean_relative_error']),
            'median_relative_error': float(metrics['median_relative_error']),
            'accuracy_5pct': float(metrics['accuracy_5pct']),
            'accuracy_10pct': float(metrics['accuracy_10pct']),
            'accuracy_20pct': float(metrics['accuracy_20pct']),
            'correlation': float(metrics['correlation']),
            'correlation_p_value': float(metrics['correlation_p_value'])
        },
        'data': {
            'true_cpi': true_cpi.tolist(),
            'predicted_cpi': predicted_cpi.tolist(),
            'relative_errors': metrics['relative_errors'].tolist()
        }
    }
    
    # Save JSON results
    json_path = os.path.join(output_dir, f'{program_name}_cpi_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {json_path}")
    
    # Save brief report
    report_path = os.path.join(output_dir, f'{program_name}_cpi_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"CPI Prediction Evaluation Report\n")
        f.write(f"Program name: {program_name}\n")
        f.write(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample count: {len(true_cpi)}\n\n")
        
        f.write(f"Evaluation Metrics:\n")
        f.write(f"  Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}\n")
        f.write(f"  R² Coefficient of Determination: {metrics['r2_score']:.6f}\n")
        f.write(f"  Pearson Correlation Coefficient: {metrics['correlation']:.6f}\n")
        f.write(f"  Mean Relative Error: {metrics['mean_relative_error']:.6f} ({metrics['mean_relative_error']*100:.2f}%)\n")
        f.write(f"  Median Relative Error: {metrics['median_relative_error']:.6f} ({metrics['median_relative_error']*100:.2f}%)\n\n")
        
        f.write(f"Accuracy Analysis:\n")
        f.write(f"  Accuracy within 5% threshold: {metrics['accuracy_5pct']:.4f} ({metrics['accuracy_5pct']*100:.2f}%)\n")
        f.write(f"  Accuracy within 10% threshold: {metrics['accuracy_10pct']:.4f} ({metrics['accuracy_10pct']*100:.2f}%)\n")
        f.write(f"  Accuracy within 20% threshold: {metrics['accuracy_20pct']:.4f} ({metrics['accuracy_20pct']*100:.2f}%)\n\n")
        
        f.write(f"Statistical Summary:\n")
        f.write(f"  True CPI - Mean: {np.mean(true_cpi):.6f}, Standard Deviation: {np.std(true_cpi):.6f}\n")
        f.write(f"  Predicted CPI - Mean: {np.mean(predicted_cpi):.6f}, Standard Deviation: {np.std(predicted_cpi):.6f}\n")
    
    print(f"Evaluation report saved to: {report_path}")

def main():
    """
    Main function: Execute CPI prediction evaluation process
    """
    parser = argparse.ArgumentParser(description='CPI prediction evaluation script')
    
    # Input file parameters
    parser.add_argument('--input', '-i', required=True, help='BBV input file path (.bb format)')
    parser.add_argument('--map', '-m', default="", help='Mapping file path')
    parser.add_argument('--ref-json', required=True, help='Reference simulation data JSON file path')
    
    # Model parameters
    parser.add_argument('--model_path', required=True, help='Path to trained Set-Transformer model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Computing device')
    parser.add_argument('--with_cpi', action='store_true', default=True, help='Whether model contains CPI prediction head')
    # SetTransformer model parameters
    parser.add_argument('--st_dim', type=int, default=256, help='SetTransformer dimension D')
    parser.add_argument('--st_inducing_points', type=int, default=12, help='SetTransformer inducing points count m')
    parser.add_argument('--st_heads', type=int, default=4, help='SetTransformer attention heads count h')
    parser.add_argument('--st_k', type=int, default=4, help='SetTransformer seed vectors count k')
    
    # General model parameters
    parser.add_argument('--encoding_dim', type=int, default=512, help='Encoding dimension')
    parser.add_argument('--margin', type=float, default=0.3, help='Triplet loss margin')
    parser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'simclr'], 
                        help='Loss function type: triplet or simclr')
    parser.add_argument('--temperature', type=float, default=0.1, 
                        help='Temperature parameter for SimCLR loss')
    # Output parameters
    parser.add_argument('--output_dir', default='./cpi_evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Prediction batch size')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(SEED)

    if args.map == "":
        args.map = os.path.splitext(args.input)[0] + '.vectors.pkl'
    
    print("=== CPI Prediction Evaluation Started ===")
    print(f"Input file: {args.input}")
    print(f"Mapping file: {args.map}")
    print(f"Reference JSON: {args.ref_json}")
    print(f"Model path: {args.model_path}")
    print(f"Computing device: {args.device}")
    
    try:
        # 1. Load data
        print("\n=== Load Data ===")
        vectors, vector_embs = vectors_mapping(args.input, args.map)
        ref_stats = load_reference_stats(args.ref_json)
        
        print(f"Loaded {len(vector_embs)} BBV samples")
        print(f"Vector dimension: {vectors.shape}")
        
        # 2. Load model
        print("\n=== Load Model ===")
        model, device = build_and_load_model(args)
        print(f"Model loaded to device: {device}")
        
        # 3. Calculate true CPI
        print("\n=== Calculate True CPI ===")
        true_cpi = calculate_true_cpi(ref_stats)

        min_len = min(len(true_cpi), len(vector_embs))
        true_cpi = true_cpi[:min_len]
        print(f"Calculated true CPI for {len(true_cpi)} intervals")
        print(f"True CPI statistics - Mean: {np.mean(true_cpi):.4f}, Standard Deviation: {np.std(true_cpi):.4f}")
        
        # 4. Predict CPI
        print("\n=== Predict CPI ===")
        predicted_cpi = predict_cpi_for_intervals(model, device, vector_embs, args.batch_size)
        min_len = min(len(true_cpi), len(predicted_cpi))
        true_cpi = true_cpi[:min_len]
        predicted_cpi = predicted_cpi[:min_len]
        print(f"Predicted CPI for {len(predicted_cpi)} intervals")
        print(f"Predicted CPI statistics - Mean: {np.mean(predicted_cpi):.4f}, Standard Deviation: {np.std(predicted_cpi):.4f}")
        
        # 5. Evaluate prediction accuracy
        print("\n=== Evaluate Prediction Accuracy ===")
        metrics = evaluate_prediction_accuracy(true_cpi, predicted_cpi)
        
        # Print evaluation results
        print(f"\nEvaluation Results:")
        print(f"  Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        print(f"  R² Coefficient of Determination: {metrics['r2_score']:.6f}")
        print(f"  Pearson Correlation Coefficient: {metrics['correlation']:.6f}")
        print(f"  Mean Relative Error: {metrics['mean_relative_error']:.4f} ({metrics['mean_relative_error']*100:.2f}%)")
        print(f"  Accuracy within 5% threshold: {metrics['accuracy_5pct']:.4f} ({metrics['accuracy_5pct']*100:.2f}%)")
        print(f"  Accuracy within 10% threshold: {metrics['accuracy_10pct']:.4f} ({metrics['accuracy_10pct']*100:.2f}%)")
        print(f"  Accuracy within 20% threshold: {metrics['accuracy_20pct']:.4f} ({metrics['accuracy_20pct']*100:.2f}%)")
        
        # 6. Generate visualization and save results
        print("\n=== Generate Visualization and Save Results ===")
        program_name = os.path.splitext(os.path.basename(args.input))[0]
        

        plot_prediction_results(true_cpi, predicted_cpi, metrics, args.output_dir, program_name)
        
        # Save results
        save_results(metrics, true_cpi, predicted_cpi, args.output_dir, program_name)
        
        print("\n=== CPI Prediction Evaluation Completed ===")
        
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()