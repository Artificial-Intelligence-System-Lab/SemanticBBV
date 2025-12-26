import os
import pickle
import torch
import numpy as np
import argparse
import json
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Import model loading function
from asm_siamese_model_infer import load_model_from_checkpoint

def load_dataset(dataset_path: str) -> Dict[str, List[Dict]]:
    """
    Load pickle format dataset file
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        Dictionary containing functions at different optimization levels {func_name: [O0, O1, O2, O3, Os]}
    """
    print(f"Load dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Successfully loaded data, containing {len(data)} functions")
    return data

def prepare_sample_for_model(sample: Dict) -> Dict[str, torch.Tensor]:
    """
    Convert samples to model input format
    
    Args:
        sample: Original sample dictionary
        
    Returns:
        Processed feature dictionary, suitable for model input
    """
    processed = {}
    
    for key, values in sample.items():
        # Convert to tensor
        processed[key] = torch.tensor(values, dtype=torch.long)
    
    return processed

def encode_samples(model, samples: List[Dict], device: str = 'cuda', batch_size: int = 32) -> torch.Tensor:
    """
    Batch encode samples
    
    Args:
        model: Siamese model
        samples: Sample list
        device: Device type
        batch_size: Batch size
        
    Returns:
        Encoded vectors
    """
    model.eval()
    encodings = []
    
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_dict = {}
            
            # Prepare batch input
            for key in batch_samples[0].keys():
                batch_dict[key] = torch.stack([sample[key] for sample in batch_samples]).to(device)
            
            # Encode
            batch_encodings = model._encode_single_sample(batch_dict)
            encodings.append(batch_encodings.cpu())
    
    # Merge all encodings
    return torch.cat(encodings, dim=0)

def calculate_metrics(query_encodings: torch.Tensor, candidate_encodings: torch.Tensor, 
                     true_indices: List[int], metric: str = 'euclidean') -> Dict[str, float]:
    """
    Calculate MRR and Recall@k metrics
    
    Args:
        query_encodings: Query sample encoding
        candidate_encodings: Candidate sample encoding
        true_indices: List of indices for true matches
        metric: Similarity measurement method
        
    Returns:
        Dictionary containing MRR and Recall@k
    """
    batch_size = query_encodings.shape[0]
    num_candidates = candidate_encodings.shape[0]
    
    # Calculate distance/similarity matrix
    if metric == 'euclidean':
        # Calculate Euclidean distance
        distances = torch.cdist(query_encodings, candidate_encodings, p=2)
        # Sort by distance in ascending order (smaller distance means more similar)
        _, sorted_indices = torch.sort(distances, dim=1)
    else:  # cosine
        # Calculate cosine similarity
        similarities = torch.matmul(
            query_encodings / torch.norm(query_encodings, dim=1, keepdim=True),
            (candidate_encodings / torch.norm(candidate_encodings, dim=1, keepdim=True)).t()
        )
        # Sort by similarity in descending order (larger similarity means more similar)
        _, sorted_indices = torch.sort(similarities, dim=1, descending=True)
    
    # Calculate position of true match in ranking for each query
    ranks = []
    for i, true_idx in enumerate(true_indices):
        # Find position of true match in ranking
        rank = (sorted_indices[i] == true_idx).nonzero().item() + 1
        ranks.append(rank)
    
    # Calculate MRR
    mrr = np.mean([1.0 / rank for rank in ranks])
    
    # Calculate Recall@k
    recall_at_1 = np.mean([1.0 if rank <= 1 else 0.0 for rank in ranks])
    
    return {
        'mrr': float(mrr),
        'recall@1': float(recall_at_1),
        'avg_rank': float(np.mean(ranks))
    }

def run_bcsd_experiment(model, dataset: Dict[str, List[Dict]], device: str, 
                       combinations: List[Tuple[int, int]], num_experiments: int = 1000,
                       candidate_pool_size: int = 32, batch_size: int = 32, 
                       metric: str = 'euclidean') -> Dict[str, Dict[str, float]]:
    """
    Run BCSD experiment
    
    Args:
        model: Siamese model
        dataset: Dataset
        device: Device type
        combinations: List of evaluation combinations, each tuple contains (source_idx, target_idx)
        num_experiments: Number of experiments for each combination
        candidate_pool_size: Candidate pool size
        batch_size: Batch size, will process in batches when candidate pool size exceeds this value
        metric: Similarity measurement method
        
    Returns:
        Dictionary containing evaluation results for each combination
    """
    # Optimization level name mapping
    opt_levels = ['O0', 'O1', 'O2', 'O3', 'Os']
    
    # Results dictionary
    results = {}
    
    # Get all function names
    func_names = list(dataset.keys())
    
    # Run experiments for each evaluation combination
    print(f"candidate_pool_size: {candidate_pool_size}")
    print(f"batch_size: {batch_size}")
    for source_idx, target_idx in combinations:
        source_level = opt_levels[source_idx]
        target_level = opt_levels[target_idx]
        combination_name = f"{source_level} vs {target_level}"
        
        print(f"\nRun experiment: {combination_name}")
        
        # Filter valid functions (containing all optimization levels)
        valid_funcs = []
        for func_name in func_names:
            if len(dataset[func_name]) == 5:  # Ensure function has all 5 optimization levels
                valid_funcs.append(func_name)
        
        if len(valid_funcs) < candidate_pool_size:
            print(f"Warning: Number of valid functions ({len(valid_funcs)}) is less than candidate pool size ({candidate_pool_size}), skip this combination")
            continue
        
        # Run multiple experiments
        experiment_results = []
        for i in tqdm(range(num_experiments), desc=f"Experiment progress ({combination_name})"):
            # Randomly select 32 functions, one as query
            selected_funcs = random.sample(valid_funcs, candidate_pool_size)
            query_func = random.choice(selected_funcs)
            
            # Prepare query sample
            query_sample = prepare_sample_for_model(dataset[query_func][source_idx])
            
            # Prepare candidate pool (containing query function)
            candidate_funcs = [f for f in selected_funcs if f != query_func]
            # Randomly insert query function into candidate pool
            true_idx = random.randint(0, candidate_pool_size - 1)
            candidate_funcs.insert(true_idx, query_func)
            
            # Prepare candidate samples
            candidate_samples = []
            for func_name in candidate_funcs:
                candidate_sample = prepare_sample_for_model(dataset[func_name][target_idx])
                candidate_samples.append(candidate_sample)
            
            # Align sequence length to multiple of CHUNK_LEN
            CHUNK_LEN = 16  # From rwkv7_cuda.py
            
            # Encode query sample
            query_batch_dict = {}
            for key in query_sample:
                tensor = query_sample[key]
                # Ensure length is multiple of CHUNK_LEN
                if tensor.shape[0] % CHUNK_LEN != 0:
                    padding_len = ((tensor.shape[0] // CHUNK_LEN) + 1) * CHUNK_LEN - tensor.shape[0]
                    padding = torch.zeros(padding_len, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding])
                query_batch_dict[key] = tensor.unsqueeze(0).to(device)  # Add batch dimension
            
            # Use model to encode query sample
            with torch.no_grad():
                query_encoding = model._encode_single_sample(query_batch_dict)
            
            # Batch process candidate samples, avoid GPU memory overflow
            candidate_encodings = []
            for j in range(0, len(candidate_samples), batch_size):
                batch_candidates = candidate_samples[j:j+batch_size]
                
                # Find the longest sequence in this batch
                max_len = 0
                for sample in batch_candidates:
                    for key in sample:
                        max_len = max(max_len, sample[key].shape[0])
                
                # If longest sequence is not multiple of CHUNK_LEN, increase length to align
                if max_len % CHUNK_LEN != 0:
                    max_len = ((max_len // CHUNK_LEN) + 1) * CHUNK_LEN
                
                # Pad all samples to same length
                batch_dict = {}
                for key in batch_candidates[0].keys():
                    padded_tensors = []
                    for sample in batch_candidates:
                        tensor = sample[key]
                        if tensor.shape[0] < max_len:
                            padding = torch.zeros(max_len - tensor.shape[0], dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, padding])
                        else:
                            padded_tensor = tensor
                        padded_tensors.append(padded_tensor)
                    batch_dict[key] = torch.stack(padded_tensors).to(device)
                
                # Batch encode candidate samples
                with torch.no_grad():
                    batch_encodings = model._encode_single_sample(batch_dict)
                    candidate_encodings.append(batch_encodings)
            
            # Merge all batch candidate encodings
            candidate_encodings = torch.cat(candidate_encodings, dim=0)
            
            # Calculate metrics
            metrics = calculate_metrics(
                query_encoding, 
                candidate_encodings,
                [true_idx],
                metric
            )
            
            experiment_results.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in experiment_results[0].keys():
            avg_metrics[metric_name] = np.mean([result[metric_name] for result in experiment_results])
        
        results[combination_name] = avg_metrics
        
        print(f"{combination_name} evaluation results:")
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Binary code similarity detection (BCSD) experiment")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Dataset file path")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Model checkpoint path")
    parser.add_argument("--vocabs_dir", type=str, required=True,
                        help="Vocabulary directory")
    parser.add_argument("--output_dir", type=str, default="./bcsd_results",
                        help="Output directory")
    parser.add_argument("--num_experiments", type=int, default=1000,
                        help="Number of experiments for each combination")
    parser.add_argument("--candidate_pool_size", type=int, default=32,
                        help="Candidate pool size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, will process in batches when candidate pool size exceeds this value")
    parser.add_argument("--similarity_metric", type=str, default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Similarity measurement method")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device type")
    
    # Model architecture parameters
    parser.add_argument("--encoding_dim", type=int, default=128,
                        help="Encoding vector dimension")
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--head_size", type=int, default=64)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Load model
    model, _ = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        vocabs_dir=args.vocabs_dir,
        device=args.device,
        encoding_dim=args.encoding_dim,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        head_size=args.head_size
    )
    
    # Define evaluation combinations
    # Index correspondence: 0=O0, 1=O1, 2=O2, 3=O3, 4=Os
    combinations = [
        (0, 3),  # O0 vs O3
        (1, 3),  # O1 vs O3
        (2, 3),  # O2 vs O3
        (0, 4),  # O0 vs Os
        (1, 4),  # O1 vs Os
        (2, 4),  # O2 vs Os
    ]
    
    # Run experiment
    results = run_bcsd_experiment(
        model=model,
        dataset=dataset,
        device=args.device,
        combinations=combinations,
        num_experiments=args.num_experiments,
        candidate_pool_size=args.candidate_pool_size,
        batch_size=args.batch_size,
        metric=args.similarity_metric
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "bcsd_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment results saved to: {results_path}")
    print("Summary results:")
    for combination, metrics in results.items():
        print(f"  {combination}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()