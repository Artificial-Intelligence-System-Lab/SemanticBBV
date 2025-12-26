import os
import pickle
import torch
import numpy as np
import argparse
import json
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

def load_embeddings_dataset(dataset_path: str) -> Dict[str, List[Optional[np.ndarray]]]:
    """
    Load pickle format dataset file containing pre-computed embeddings.

    Args:
        dataset_path: dataset file path

    Returns:
        Dictionary containing function embeddings, e.g., {func_name: [emb_O0, emb_O1, ...]}
        Some embeddings may be None.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    print(f"Loading embedding dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Successfully loaded data, containing embeddings of {len(data)} functions")
    return data

def calculate_metrics(query_encoding: torch.Tensor, candidate_encodings: torch.Tensor, 
                     true_idx: int, metric: str = 'euclidean') -> Dict[str, float]:
    """
    Calculate MRR and Recall@k metrics.

    Args:
        query_encoding: encoding of single query sample (shape [1, D])
        candidate_encodings: encoding of candidate samples (shape [N, D])
        true_idx: index of true match in candidate pool
        metric: similarity metric method ('euclidean' or 'cosine')

    Returns:
        Dictionary containing MRR and Recall@k
    """
    # Calculate distance/similarity matrix
    if metric == 'euclidean':
        # Calculate Euclidean distance
        distances = torch.cdist(query_encoding, candidate_encodings, p=2).squeeze(0) # Remove batch dimension
        # Sort by distance ascending (smaller distance is more similar)
        _, sorted_indices = torch.sort(distances)
    else:  # cosine
        # Calculate cosine similarity
        query_norm = query_encoding / torch.norm(query_encoding)
        candidates_norm = candidate_encodings / torch.norm(candidate_encodings, dim=1, keepdim=True)
        similarities = torch.matmul(query_norm, candidates_norm.t()).squeeze(0)
        # Sort by similarity descending (larger similarity is more similar)
        _, sorted_indices = torch.sort(similarities, descending=True)
    
    # Find position of true match in sorted order (rank)
    # .item() converts 0-dimensional tensor to python number
    rank = (sorted_indices == true_idx).nonzero(as_tuple=True)[0].item() + 1
    
    # Calculate metrics
    mrr = 1.0 / rank
    recall_at_1 = 1.0 if rank <= 1 else 0.0
    
    return {
        'mrr': mrr,
        'recall@1': recall_at_1,
        'avg_rank': float(rank) # For single experiment, average rank is itself
    }

def run_bcsd_experiment(dataset: Dict[str, List[Optional[np.ndarray]]], device: str, 
                       combinations: List[Tuple[int, int]], num_experiments: int = 1000,
                       candidate_pool_size: int = 32, metric: str = 'euclidean') -> Dict[str, Dict[str, float]]:
    """
    Run BCSD experiment using pre-computed embeddings.

    Args:
        dataset: dataset containing embeddings
        device: 'cuda' or 'cpu'
        combinations: list of evaluation combinations, each tuple contains (source_idx, target_idx)
        num_experiments: number of experiments for each combination
        candidate_pool_size: size of candidate pool
        metric: similarity metric method

    Returns:
        Dictionary containing evaluation results for each combination
    """
    opt_levels = ['O0', 'O1', 'O2', 'O3', 'Os']
    results = {}
    
    print(f"Candidate pool size: {candidate_pool_size}")
    
    # Run experiments for each evaluation combination
    for source_idx, target_idx in combinations:
        combination_name = f"{opt_levels[source_idx]}_vs_{opt_levels[target_idx]}"
        print(f"\nRun experiment: {combination_name}")
        
        # Key step: Pre-select functions that have valid embeddings in this combination
        # A function is valid if and only if its embeddings at source and target optimization levels are both not None
        valid_funcs = [
            func_name for func_name, embeddings in dataset.items()
            if len(embeddings) > max(source_idx, target_idx) and \
               embeddings[source_idx] is not None and \
               embeddings[target_idx] is not None
        ]
        
        if len(valid_funcs) < candidate_pool_size:
            print(f"Warning: Number of valid functions ({len(valid_funcs)}) is less than candidate pool size ({candidate_pool_size}), skip this combination")
            continue
        
        print(f"Found {len(valid_funcs)} valid functions for this experiment combination.")
        
        experiment_results = []
        for _ in tqdm(range(num_experiments), desc=f"Experiment progress ({combination_name})"):
            # 1. Randomly select N functions from the valid function list as the candidate pool for this experiment
            #    This method ensures all selected functions have valid embeddings, no need to retry in the loop
            selected_funcs = random.sample(valid_funcs, candidate_pool_size)
            
            # 2. Use the first function as query, the rest as distractors
            query_func_name = selected_funcs[0]
            
            # 3. Get query embedding
            query_embedding = dataset[query_func_name][source_idx]["embedding"]
            
            # 4. Prepare candidate pool. To ensure fairness, we shuffle the order of the candidate pool
            random.shuffle(selected_funcs)
            
            # 5. Find the index of the true match (query function) in the new order
            true_idx = selected_funcs.index(query_func_name)
            
            # 6. Get all candidate embeddings
            candidate_embeddings = [dataset[func_name][target_idx]["embedding"] for func_name in selected_funcs]
            
            # 7. Convert Numpy array to Torch tensor and move to specified device
            query_tensor = torch.from_numpy(query_embedding).unsqueeze(0).to(device) # [1, D]
            candidate_tensors = torch.from_numpy(np.stack(candidate_embeddings)).to(device) # [N, D]

            # 8. Calculate metrics
            metrics = calculate_metrics(
                query_tensor, 
                candidate_tensors,
                true_idx,
                metric
            )
            experiment_results.append(metrics)
        
        # Calculate average metrics for multiple experiments
        if not experiment_results:
            print("No valid experiment results, skip metrics calculation.")
            continue
            
        avg_metrics = {}
        for metric_name in experiment_results[0].keys():
            avg_metrics[metric_name] = np.mean([res[metric_name] for res in experiment_results])
        
        results[combination_name] = avg_metrics
        
        print(f"{combination_name} evaluation results:")
        for metric_name, value in avg_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Binary code similarity detection (BCSD) experiment based on pre-computed embeddings")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset file containing pre-computed embeddings (.pkl)")
    parser.add_argument("--output_dir", type=str, default="./bcsd_results",
                        help="Output directory for saving result JSON files")
    parser.add_argument("--num_experiments", type=int, default=1000,
                        help="Number of experiments for each optimization combination")
    parser.add_argument("--candidate_pool_size", type=int, default=1000,
                        help="Candidate pool size for each experiment")
    parser.add_argument("--similarity_metric", type=str, default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Similarity/distance measurement method")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for running computation ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_embeddings_dataset(args.dataset_path)
    
    # Define evaluation combinations
    # Index correspondence: 0=O0, 1=O1, 2=O2, 3=O3, 4=Os
    # You can modify or add combinations as needed
    combinations = [
        (0, 3),  # O0 vs O3
        (1, 3),  # O1 vs O3
        (2, 3),  # O2 vs O3
        (0, 4),  # O0 vs Os
        (1, 4),  # O1 vs Os
        (2, 4),  # O2 vs Os
        (3, 4),  # O3 vs Os
    ]
    
    # Run experiment
    print(f"Will run experiments on device {args.device}...")
    results = run_bcsd_experiment(
        dataset=dataset,
        device=args.device,
        combinations=combinations,
        num_experiments=args.num_experiments,
        candidate_pool_size=args.candidate_pool_size,
        metric=args.similarity_metric
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "bcsd_embedding_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to: {results_path}")
    print("\n---------- Final Results Summary ----------")
    for combination, metrics in results.items():
        print(f"  Combination: {combination}")
        for metric_name, value in metrics.items():
            print(f"    - {metric_name:<10}: {value:.4f}")
    print("-----------------------------------")

if __name__ == "__main__":
    main()