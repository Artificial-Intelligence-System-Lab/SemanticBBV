import json
import os
import sys
import re
import torch
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
# Add the parent directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from train_rwkv7_deepspeed import adjust_batch_to_max_valid_length
from asm_dataset_preprocessed_fine import AsmVocab


PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
UNK_ID = 3


def load_model(checkpoint_path, vocabs_size, device="cuda", n_embd=768, n_layer=6, head_size=64, use_bf16=True):
    """
    Load RWKV7 model
    
    Args:
        checkpoint_path: checkpoint path
        device: compute device
        n_embd: embedding dimension
        n_layer: number of model layers
        vocabs_size: vocabulary size
        head_size: attention head size
        use_bf16: whether to use BF16 precision
        
    Returns:
        Loaded model
    """
    # Convert relative imports to absolute imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rwkv7_cuda import RWKV
    import torch.nn as nn

    class Args:
        def __init__(self, n_embd, n_layer, head_size):
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.head_size = head_size
    
    class RWKV7Model(nn.Module):
        def __init__(self, args, vocab_size):
            super().__init__()
            model_args = type('', (), {})()
            model_args.n_embd = args.n_embd
            model_args.n_layer = args.n_layer
            model_args.head_size_a = args.head_size

            self.asm_embd = nn.Embedding(vocab_size["asm"], args.n_embd)
            self.mne_embd = nn.Embedding(vocab_size["mne"], args.n_embd)
            self.type_embd = nn.Embedding(vocab_size["type"], args.n_embd)
            self.reg_embd = nn.Embedding(vocab_size["reg"], args.n_embd)
            self.rw_embd = nn.Embedding(vocab_size["rw"], args.n_embd)
            self.eflag_embd = nn.Embedding(vocab_size["eflag"], args.n_embd)

            self.model = RWKV(model_args)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 as ignore index

            # Add two different output layers for token-level and instruction-level MLM respectively
            #self.token_mlm_head = nn.Linear(args.n_embd, vocab_size["asm"])
            #self.instr_mlm_head = nn.Linear(args.n_embd, vocab_size["asm"])

        def forward(self, token_inputs, layer_idx=None):
            # Forward pass for token-level task
            # Embed and sum each dimension
            asm_emb = self.asm_embd(token_inputs['asm'])
            mne_emb = self.mne_embd(token_inputs["mne"])
            type_emb = self.type_embd(token_inputs['type'])
            reg_emb = self.reg_embd(token_inputs['reg'])
            rw_emb = self.rw_embd(token_inputs['rw'])
            eflag_embd = self.eflag_embd(token_inputs['eflag'])

            # Sum all embeddings
            combined_emb = asm_emb + mne_emb + type_emb + reg_emb + rw_emb + eflag_embd

            # Pass combined embedding to model
            if layer_idx is not None:
                token_hidden = self.model(combined_emb, layer_idx=layer_idx)
            else:
                token_hidden = self.model(combined_emb)
            
            return token_hidden
    
    args = Args(n_embd=n_embd, n_layer=n_layer, head_size=head_size)
    # Create model
    model = RWKV7Model(args=args, vocab_size=vocabs_size)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(checkpoint.keys())
    
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    
    # If BF16 enabled, convert model to BF16 precision
    if use_bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("Converting model to BF16 precision")
            model = model.to(torch.bfloat16)
        else:
            print("Warning: current device does not support BF16 precision, will continue using FP32")
    
    model.eval()
    print("Model loaded successfully")
    
    return model

def get_instruction_encodings(model, asm_instructions, batch_size=128, encoding=None, layer_idx=None, device="cuda", max_samples=10000):
    """
    Get encoding for each assembly instruction, supporting multiple encoding methods, using batch processing for acceleration
    
    Args:
        model: loaded model
        asm_instructions: list of assembly instructions
        batch_size: batch size, default 128
        encoding: pre-encoding information
        layer_idx: which layer's hidden states to use, default use 9th layer
        device: compute device, default cuda
        max_samples: maximum number of samples to process, default 10000
        
    Returns:
        List of instruction encodings
    """
    encodings = []
    error_count = 0
    
    # Create progress bar object
    total_instructions = len(asm_instructions)
    
    # Randomly select specified number of instructions for processing
    if total_instructions > max_samples:
        import random
        random.seed(43)  # Set random seed to ensure reproducible results
        selected_indices = random.sample(range(total_instructions), max_samples)
        asm_instructions = [asm_instructions[i] for i in selected_indices]
        print(f"Randomly selected {max_samples} instructions from {total_instructions} for processing")
        total_instructions = max_samples
    
    pbar = tqdm(total=total_instructions, desc="Processing instructions")
    
    # Pre-process all instructions
    all_tokens = []
    valid_indices = []
    
    print(f"Pre-processing assembly instructions...")
    for i, asm in enumerate(tqdm(asm_instructions, desc="Pre-processing instructions")):
        try:
            tokens = encoding[asm]
            all_tokens.append(tokens)
            valid_indices.append(i)
        except Exception as e:
            import traceback
            print(f"Error details: {str(e)}")
            print("Error stack:")
            # Fix: Convert dict_keys to list with list() before slicing
            try:
                for key in list(encoding.keys())[:5]:
                    print(key, encoding[key])
            except:
                print("Unable to print encoding keys")
            traceback.print_exc()
            # Remove sys.exit(1) to let program continue processing other instructions
            error_count += 1
            continue
    
    print(f"Starting batch inference, batch size: {batch_size}...")
    # Process in batches
    for i in range(0, len(all_tokens), batch_size):
        # Get current batch
        batch_tokens = all_tokens[i:i+batch_size]
        batch_indices = valid_indices[i:i+batch_size]
        
        with torch.no_grad():
            # Create batch tensor
            batch_tokens_dict = defaultdict(list)
            # Use instruction list for current batch
            # First find maximum length in each dimension
            max_lengths = {}
            for inst in batch_tokens:
                for key, value in inst.items():
                    if key not in max_lengths or len(value) > max_lengths[key]:
                        max_lengths[key] = len(value)
            
            # Then pad each instruction to maximum length
            for inst in batch_tokens:
                for key, value in inst.items():
                    # If current instruction length is less than maximum, pad it
                    if len(value) < max_lengths[key]:
                        # Use PAD_ID for padding
                        padded_value = value + [PAD_ID] * (max_lengths[key] - len(value))
                        batch_tokens_dict[key].append(padded_value)
                    else:
                        batch_tokens_dict[key].append(value)
            batch_tokens = dict(batch_tokens_dict)
            
            for key in batch_tokens:
                batch_tokens[key] = torch.tensor(batch_tokens[key], dtype=torch.long).to(device)
            batch_tokens = adjust_batch_to_max_valid_length(batch_inputs=batch_tokens)
            # Get hidden states and specify using hidden states from layer_idx
            # Need to modify model's forward method to support returning hidden states from specified layer
            hidden = model(batch_tokens, layer_idx=layer_idx)
            
            # Use random mapping matrix to process hidden states
            if not hasattr(get_instruction_encodings, 'random_matrix'):
                # Initialize random mapping matrix (768 -> 20)
                torch.manual_seed(42)  # Set random seed to ensure reproducible results
                get_instruction_encodings.random_matrix = torch.randn(768, 20).to(device)
                print(f"Initialized random mapping matrix, shape: {get_instruction_encodings.random_matrix.shape}")
            
            # Apply mask and calculate average hidden states for each sequence
            mask = (batch_tokens["asm"] != PAD_ID).float()
            avg_hidden = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            encoding = avg_hidden.float().cpu().numpy()
            print(avg_hidden.shape)
            
            # Use random mapping matrix to map hidden states to length 20 vector
            #encoding = torch.matmul(avg_hidden, get_instruction_encodings.random_matrix).float().cpu().numpy()
            
            # Add results to encoding list
            for j, idx in enumerate(batch_indices):
                result = {
                    'instruction': asm_instructions[idx],
                    'encoding': encoding[j].tolist()
                }
                encodings.append(result)
            
            # Update progress bar
            pbar.update(len(batch_tokens))

    # Close progress bar
    pbar.close()
    
    if error_count > 0:
        print(f"Warning: {error_count} instructions processing failed")
        
    return encodings

def save_encodings(encodings, output_file):
    """Save encodings to file"""
    # If encodings is integer, indicates how many records have been saved via streaming
    if isinstance(encodings, int):
        print(f"Saved {encodings} instruction encodings to {output_file} via streaming processing")
        return
        
    with open(output_file, "wb") as f:
        import pickle
        pickle.dump(encodings, f)
    print(f"Saved encodings for {len(encodings)} instructions to {output_file}")


def evaluate_kmeans_clustering(vectors, cluster_labels):
    """
    Evaluate K-means clustering quality
    
    Args:
        vectors: vector data
        cluster_labels: clustering labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    # Calculate Silhouette Score
    try:
        silhouette = silhouette_score(vectors, cluster_labels)
    except:
        silhouette = -1  # If calculation fails, return -1
    
    # Calculate Davies-Bouldin Index
    try:
        db_score = davies_bouldin_score(vectors, cluster_labels)
    except:
        db_score = float('inf')  # If calculation fails, return infinity
    
    # Calculate Calinski-Harabasz Index
    try:
        ch_score = calinski_harabasz_score(vectors, cluster_labels)
    except:
        ch_score = 0  # If calculation fails, return 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score
    }


def visualize_tsne(encodings, output_file="tsne_visualization.pdf", categories_file=None, max_k=30):
    """
    Visualize instruction encodings using t-SNE, use K-means clustering and determine optimal k value via Elbow method
    
    Args:
        encodings: instruction encoding list
        output_file: output image file path
        categories_file: file path containing instruction classification, for comparison only, not used for coloring
        max_k: maximum k value for K-means clustering, used for Elbow method
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from sklearn.metrics import silhouette_score
        import numpy as np
    except ImportError:
        print("Error: please install necessary libraries: pip install scikit-learn matplotlib numpy")
        return
    
    if not encodings:
        print("Error: no available encoding data")
        return
    
    # Extract instructions and encodings
    instructions = [item['instruction'] for item in encodings]
    vectors = np.array([item['encoding'] for item in encodings])
    
    print(vectors.shape)
    
    # Automatically adjust perplexity parameter based on sample size
    n_samples = len(vectors)
    perplexity = min(n_samples - 1, 30)  # Ensure perplexity is less than number of samples
    n_iter = 2000
    
    print(f"Starting t-SNE dimensionality reduction (data: {n_samples} samples, dimension: {vectors.shape[1]}, perplexity: {perplexity})")
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    reduced_data = tsne.fit_transform(vectors)
    
    # Use Elbow method to determine optimal k value
    print(f"Using Elbow method to determine optimal k value (max k: {max_k})")
    
    # Limit max_k to no more than half the number of samples
    max_k = min(max_k, n_samples // 2)
    
    # Calculate SSE (Sum of Squared Errors) for different k values
    sse = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)  # Start from 2 because k=1 doesn't make sense
    
    for k in tqdm(k_values, desc="Calculate SSE for different k values"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectors)
        sse.append(kmeans.inertia_)
        
        # Calculate silhouette coefficient
        try:
            silhouette_scores.append(silhouette_score(vectors, kmeans.labels_))
        except:
            silhouette_scores.append(-1)
    
    # Use Elbow method to determine optimal k value
    # Calculate second derivative of SSE curve, inflection point has maximum second derivative
    k_best = 2  # Default value
    if len(sse) > 2:
        # Calculate first derivative
        diffs = np.diff(sse)
        # Calculate second derivative
        diffs2 = np.diff(diffs)
        # Find point with maximum second derivative
        elbow_index = np.argmax(diffs2) + 2  # +2 because we start from k=2 and second derivative reduces 2 elements
        k_best = k_values[elbow_index]
        
        # If k value with higher silhouette coefficient differs little from Elbow method result, prioritize higher silhouette coefficient
        silhouette_best = np.argmax(silhouette_scores)
        k_silhouette_best = k_values[silhouette_best]
        
        # If k value with best silhouette coefficient differs no more than 3 from Elbow method result and silhouette is significantly better, select it
        if abs(k_silhouette_best - k_best) <= 3 and silhouette_scores[silhouette_best] > 0.1:
            k_best = k_silhouette_best
            print(f"Selected k={k_best} based on silhouette coefficient (silhouette coefficient: {silhouette_scores[silhouette_best]:.4f})")
        else:
            print(f"Selected k={k_best} based on Elbow method")
    
    # Use determined optimal k value for K-means clustering
    print(f"Using k={k_best} for K-means clustering")
    kmeans = KMeans(n_clusters=k_best, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    # Convert original categories to numeric for comparison (if provided)
    original_labels = None
    if categories_file and os.path.exists(categories_file):
        # Read all instructions and corresponding categories
        with open(categories_file, 'r') as f:
            categories = [line.strip() for line in f]
        
        # Ensure number of lines in category file matches number of instructions
        if len(categories) != len(instructions):
            print(f"Warning: number of lines in category file ({len(categories)}) does not match number of instructions ({len(instructions)})")
            # If mismatch, truncate or extend category list
            if len(categories) > len(instructions):
                categories = categories[:len(instructions)]
            else:
                categories.extend(["unknown"] * (len(instructions) - len(categories)))
        
        # Convert categories to numeric values
        unique_categories = list(set(categories))
        category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
        original_labels = np.array([category_to_id[cat] for cat in categories])
        
        # Calculate consistency between clustering results and original categories
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(original_labels, cluster_labels)
        nmi = normalized_mutual_info_score(original_labels, cluster_labels)
        print(f"Consistency evaluation between clustering results and original categories:")
        print(f"Adjusted Rand Index (ARI): {ari:.4f} (range: [-1, 1], closer to 1 is better)")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f} (range: [0, 1], closer to 1 is better)")
    
    # Calculate clustering quality evaluation metrics
    metrics = evaluate_kmeans_clustering(vectors, cluster_labels)
    
    # Output evaluation results
    print("\nK-means clustering quality evaluation metrics:")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (range: [-1, 1], closer to 1 is better)")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot Elbow method graph
    plt.subplot(2, 2, 1)
    plt.plot(k_values, sse, 'bo-')
    plt.plot(k_best, sse[k_best-2], 'ro', markersize=10)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Plot silhouette coefficient graph
    plt.subplot(2, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different k Values')
    plt.grid(True)
    
    # Plot t-SNE dimensionality reduction results, colored by K-means clustering labels
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(f't-SNE Dim. Reduction (K-means Clustering, k={k_best})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # If original categories exist, plot t-SNE using original categories for coloring
    if original_labels is not None:
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=original_labels, cmap='tab20', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Original Label')
        plt.title('t-SNE Dim. Reduction (Original Labels)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Visualization results saved to: {output_file}")
    
    # Save clustering results
    cluster_file = output_file.replace('.pdf', '_clusters.txt')
    with open(cluster_file, 'w') as f:
        for i, label in enumerate(cluster_labels):
            f.write(f"{instructions[i]}\t{label}\n")
    print(f"Clustering results saved to: {cluster_file}")
    
    # Save representative samples from each cluster
    cluster_samples_file = output_file.replace('.pdf', '_cluster_samples.txt')
    with open(cluster_samples_file, 'w') as f:
        # Group by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, instructions[i]))
        
        # Output number of samples in each cluster and representative samples
        for label in sorted(clusters.keys()):
            samples = clusters[label]
            f.write(f"Cluster {label} (Total {len(samples)} samples):\n")
            
            # Calculate distance to cluster center
            cluster_center = kmeans.cluster_centers_[label]
            samples_with_dist = []
            for idx, insn in samples:
                dist = np.linalg.norm(vectors[idx] - cluster_center)
                samples_with_dist.append((idx, insn, dist))
            
            # Sort by distance, select 10 closest samples to center
            samples_with_dist.sort(key=lambda x: x[2])
            for i, (idx, insn, dist) in enumerate(samples_with_dist[:10]):
                f.write(f"  {i+1}. {insn} (Distance: {dist:.4f})\n")
            f.write("\n")
    
    print(f"Cluster samples saved to: {cluster_samples_file}")
    
    return cluster_labels


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Assembly instruction encoding and visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--encoding_file", type=str, required=True, help="Path to instruction encoding file")
    parser.add_argument("--asm_instructions", type=str, required=False, default=None, help="Path to instruction encoding file")
    parser.add_argument("--vocabs_dir", type=str, help="Path to vocabulary directory")
    parser.add_argument("--output_file", type=str, default="tsne_visualization.pdf", help="Path to output visualization file")
    parser.add_argument("--max_k", type=int, default=30, help="Maximum k value for K-means clustering")
    parser.add_argument("--max_samples", type=int, default=4200, help="Maximum number of samples to process")
    parser.add_argument("--layer_idx", type=int, default=None, help="Which layer's hidden states to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of model layers")
    parser.add_argument("--head_size", type=int, default=64, help="Attention head size")
    parser.add_argument("--use_bf16", type=bool, default=True, help="Whether to use BF16 precision")
    
    args = parser.parse_args()
    
    # Load encoding file
    print(f"Loading encoding file: {args.encoding_file}")
    with open(args.encoding_file, "rb") as f:
        encoding_data = pickle.load(f)
    
    # Extract instruction list
    if args.asm_instructions:
        with open(args.asm_instructions, "r") as f:
            asm_instructions = [inst.strip() for inst in f]
    else:
        asm_instructions = list(encoding_data.keys())
    print(f"Found {len(asm_instructions)} instructions")

    vocabs = {}

    vocab_config = {
        "asm": "asm_tokens.txt",
        "mne": "mne_tokens.txt",
        "type": "type_tokens.txt",
        "reg": "reg_tokens.txt",
        "rw": "rw_tokens.txt",
        "eflag": "eflag_tokens.txt"
    }

    for key, filename in vocab_config.items():
        vocab_path = os.path.join(args.vocabs_dir, filename)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please ensure all vocabulary files exist in '{self.vocabs_dir}' directory.")
        vocab = AsmVocab()
        vocab.load(vocab_path)
        vocabs[key] = vocab
    
    vocabs_size = {key: vocab.length() for key, vocab in vocabs.items()}
    
    # Load model
    model = load_model(
        args.checkpoint, 
        vocabs_size=vocabs_size,
        device=args.device,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        head_size=args.head_size,
        use_bf16=args.use_bf16
    )
    
    # Get instruction encodings
    encodings = get_instruction_encodings(
        model, 
        asm_instructions, 
        batch_size=args.batch_size,
        encoding=encoding_data,
        layer_idx=args.layer_idx,
        device=args.device,
        max_samples=args.max_samples
    )
    
    # Visualize
    visualize_tsne(encodings, output_file=args.output_file, max_k=args.max_k)


if __name__ == "__main__":
    main()
