#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from pre_processing.tokenizer import tokenize_binary_instruction

def load_embeddings(file_path):
    """
    Load embedding file
    
    Parameters:
        file_path: path to embedding file
        
    Returns:
        dictionary containing addresses and corresponding vectors
    """
    print(f"Loading embedding file: {file_path}")
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)

    for emb in embeddings.values():
        if len(emb) != 128:
            print(emb)
            sys.exit(0)
    print(f"Loading complete, total {len(embeddings)} vectors")
    return embeddings

def compute_cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    
    Parameters:
        vec1, vec2: input vectors
        
    Returns:
        cosine similarity value, range [-1, 1]
    """
    # Normalize vectors to improve numerical stability
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1_norm, vec2_norm)
    
    return similarity

def cluster_embeddings(embeddings, threshold=0.35):
    """
    Cluster embeddings based on cosine similarity threshold
    Using union-find algorithm for clustering
    
    Parameters:
        embeddings: dictionary mapping address to vector
        threshold: similarity threshold, vectors with similarity less than this are considered similar
        
    Returns:
        clustering results dictionary, key is cluster ID, value is list of addresses in that cluster
    """
    print(f"Clustering with similarity threshold {threshold}...")
    
    # Initialize union-find
    cluster_id = 0
    clusters = {}
    
    keys = list(embeddings.keys())
    embeddings = list(embeddings.values())
    min_similarity = 1.0
    max_similarity = -1.0

    for i in tqdm(range(len(embeddings))):
        id_1 = keys[i]
        if id_1 in clusters:
            continue
        vec1 = embeddings[i]
        clusters[id_1] = cluster_id
        for j in range(i+1, len(embeddings)):
            id_2 = keys[j]
            vec2 = embeddings[j]
            similarity = compute_cosine_similarity(vec1, vec2)
            
            # Update minimum and maximum similarity values
            min_similarity = min(min_similarity, similarity)
            max_similarity = max(max_similarity, similarity)

            if similarity >= threshold:
                if id_2 not in clusters:
                    clusters[id_2] = cluster_id
        
        cluster_id += 1
    
    # Print minimum and maximum similarity values
    print(f"Minimum similarity: {min_similarity:.4f}")
    print(f"Maximum similarity: {max_similarity:.4f}")
    
    # Build clustering results dictionary
    results = defaultdict(list)
    for addr, cluster_id in clusters.items():
        results[cluster_id].append(addr)
    
    new_results = {i: cluster for i, (_, cluster) in enumerate(results.items())}
    return new_results

def plot_cluster_size_distribution(cluster_sizes, output_path=None):
    """
    Plot cluster size distribution
    
    Parameters:
        cluster_sizes: list of cluster sizes
        output_path: output image path, None to display image
    """
    plt.figure(figsize=(10, 6))
    plt.hist(cluster_sizes, bins=50, alpha=0.75)
    plt.xlabel('Cluster size')
    plt.ylabel('Number of clusters')
    plt.title('Cluster size distribution')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Cluster size distribution plot saved to: {output_path}")
    else:
        plt.show()

def save_clusters(clusters, output_path):
    """
    Save clustering results
    
    Parameters:
        clusters: clustering results dictionary
        output_path: output file path
    """
    with open(output_path, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clustering results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Cluster analysis of embedding vectors')
    parser.add_argument('embedding_file', type=str, help='path to embedding file')
    parser.add_argument('--binary_file', type=str, help='path to embedding file')
    parser.add_argument('--threshold', type=float, default=0.95, help='similarity threshold, vectors with similarity less than this are considered similar')
    parser.add_argument('--output', type=str, default=None, help='path to save clustering results')
    parser.add_argument('--plot', type=str, default=None, help='path to save cluster size distribution plot')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = os.path.splitext(args.embedding_file)[0] + '.clusters.pkl'
    
    # Load embeddings
    embeddings = load_embeddings(args.embedding_file)
    
    # Clustering
    clusters = cluster_embeddings(embeddings, args.threshold)

    print(f"\nProcessing complete, total number of clusters: {len(clusters)}")
    # find the largest cluster
    largest_cluster = max(clusters.values(), key=len)
    print(f"Largest cluster contains {len(largest_cluster)} elements")
    
    # Randomly select five clusters with more than one element
    multi_element_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]
    print(f"Total number of clusters with more than one element: {len(multi_element_clusters)}")
    
    import random
    sample_clusters = random.sample(multi_element_clusters, min(5, len(multi_element_clusters)))
    
    with open(args.binary_file, 'rb') as f:
        binary_data = pickle.load(f)
    
    # Print largest cluster
    print("\nLargest cluster example:")
    for addr in largest_cluster[:5]:  # Only print first 5 elements
        print(f"Address: {addr}")
        binary_bytes = bytes(binary_data[addr])
        print(f"Corresponding binary instruction: {tokenize_binary_instruction(binary_bytes, addr)}")
        print("-----------------------")
    
    # Print randomly selected clusters
    print("\n5 randomly selected multi-element clusters:")
    for i, cluster in enumerate(sample_clusters, 1):
        print(f"\nCluster {i}, containing {len(cluster)} elements")
        for addr in cluster[:5]:  # Print only first 3 elements from each cluster
            print(f"Address: {addr}")
            binary_bytes = bytes(binary_data[addr])
            print(f"Corresponding binary instruction: {tokenize_binary_instruction(binary_bytes, addr)}")
            print("-----------------------")
    
    # revert the clusters
    clusters = {addr: cluster_id for cluster_id, cluster in clusters.items() for addr in cluster}
    
    # check if the output path exists, if not, create it
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_clusters(clusters, args.output)

if __name__ == '__main__':
    main()