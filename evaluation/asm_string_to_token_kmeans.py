import json
import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict

# Add the parent directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from rwkv7_cuda import RWKV
from train_rwkv7_deepspeed import adjust_batch_to_max_valid_length


PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
UNK_ID = 3


def normalize_insn(asm):
    """Normalize instruction format"""
    # Handle different formats of assembly instructions
    if "\t" in asm:
        parts = asm.split("\t", 1)
        opcode, op_str = parts[0].strip(), parts[1].strip()
    else:
        # Try to handle space-separated format, such as "add [rbp+var_2630], 1"
        match = re.match(r'^\s*([a-zA-Z0-9.]+)\s+(.*?)$', asm)
        if match:
            opcode, op_str = match.group(1).strip(), match.group(2).strip()
        else:
            return asm, []

    op_str = op_str.replace(" + ", "+")
    op_str = op_str.replace(" - ", "-")
    op_str = op_str.replace(" * ", "*")
    op_str = op_str.replace(" : ", ":")

    # Define regex patterns to match various numeric forms
    pattern = r"0x[0-9a-fA-F]+h?|\b[0-9a-fA-F]+h\b|\b[0-9]\b|(?<=[+\-*/])\d+"

    # Replace numeric values in operand strings
    def repl(match):
        start = match.start()
        preceding = op_str[max(0, start - 15) : start].lower()

        if "ptr" in preceding:
            return "PTR_ADDR"
        elif "rel" in preceding:
            return "REL_ADDR"
        else:
            return "IMM"

    op_str = re.sub(pattern, repl, op_str)

    if op_str:
        opnd_strs = [x.strip() for x in op_str.split(",")]
    else:
        opnd_strs = []

    # Iterate and replace numeric values in each operand
    opnd_strs = [re.sub(pattern, repl, opnd) for opnd in opnd_strs]

    return opcode, opnd_strs


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(checkpoint_path, device="cuda", n_embd=768, n_layer=6, head_size=64, use_bf16=True):
    """
    Load RWKV7 model
    
    Args:
        checkpoint_path: checkpoint path
        device: compute device
        n_embd: embedding dimension
        n_layer: number of model layers
        head_size: attention head size
        use_bf16: whether to use BF16 precision
        
    Returns:
        Loaded model
    """
    
    class Args:
        def __init__(self):
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.vocab_size = 3759
            # Example values updated to actual values
            self.mnemonic_vocab_size = 1045  # Update to actual value
            self.op_kind_vocab_size = 20     # Update to actual value
            self.op_id_vocab_size = 193      # Update to actual value
            self.reg_r_vocab_size = 7        # Update to actual value
            self.reg_w_vocab_size = 7        # Update to actual value
            self.eflags_vocab_size = 67      # Update to actual value
            self.head_size = head_size
    
    class RWKV7Model(nn.Module):
        def __init__(self, args):
            super().__init__()
            model_args = type('', (), {})()
            model_args.n_embd = args.n_embd
            model_args.n_layer = args.n_layer
            model_args.head_size_a = args.head_size

            self.token_embd = nn.Embedding(args.vocab_size, args.n_embd)
            self.mnemonic_embd = nn.Embedding(args.mnemonic_vocab_size, args.n_embd)
            self.op_kind_embd = nn.Embedding(args.op_kind_vocab_size, args.n_embd)
            self.op_id_embd = nn.Embedding(args.op_id_vocab_size, args.n_embd)
            self.reg_r_embd = nn.Embedding(args.reg_r_vocab_size, args.n_embd)
            self.reg_w_embd = nn.Embedding(args.reg_w_vocab_size, args.n_embd)
            self.eflags_embd = nn.Embedding(args.eflags_vocab_size, args.n_embd)

            self.model = RWKV(model_args)

        def forward(self, inputs, layer_idx=None):
            # Forward pass for token-level task
            # Embed and sum each dimension
            token_emb = self.token_embd(inputs['token_ids'])
            mnemonic_emb = self.mnemonic_embd(inputs['mnemonic_ids'])
            op_kind_emb = self.op_kind_embd(inputs['op_kind_ids'])
            op_id_emb = self.op_id_embd(inputs['op_id_ids'])
            reg_r_emb = self.reg_r_embd(inputs['reg_r_ids'])
            reg_w_emb = self.reg_w_embd(inputs['reg_w_ids'])
            eflags_emb = self.eflags_embd(inputs['eflags_id'])

            # Sum all embeddings
            combined_emb = token_emb + mnemonic_emb + op_kind_emb + op_id_emb + reg_r_emb + reg_w_emb + eflags_emb

            # Pass combined embedding to model
            if layer_idx is not None:
                token_hidden = self.model(combined_emb, layer_idx=layer_idx)
            else:
                token_hidden = self.model(combined_emb)
            return token_hidden

    args = Args()
    model = RWKV7Model(args)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'module' in checkpoint:
        # Use non-strict mode for loading, ignore mismatched keys
        model.load_state_dict(checkpoint['module'], strict=False)
        print("Using non-strict mode to load checkpoint (ignore mismatched keys)")
    else:
        # Check and handle key name mismatches
        new_state_dict = {}
        for key, value in checkpoint.items():
            # Handle model head weights
            if key == 'model.head.weight':
                new_state_dict['head.weight'] = value
            elif key == 'model.head.bias' and 'head.bias' not in checkpoint:
                # If bias exists but corresponding key not in model
                new_state_dict['head.bias'] = value
            else:
                new_state_dict[key] = value
        
        # If head.bias is missing, initialize as zero
        if 'head.bias' not in new_state_dict and 'head.weight' in new_state_dict:
            print("head.bias not found, initialize as zero")
            new_state_dict['head.bias'] = torch.zeros(vocab_size)
        
        # Use non-strict mode for loading, ignore mismatched keys
        model.load_state_dict(new_state_dict, strict=False)
        print("Using non-strict mode to load checkpoint (ignore mismatched keys)")
    
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

def asm_to_tokens(asm_str, token2id, id_mapping, max_len=16, insn_info=None):
    """
    Convert assembly string to token id list and perform normalization and token mapping
    
    Args:
        asm_str: assembly instruction string
        token2id: mapping from token to id
        id_mapping: mapping from id to compressed id
        max_len: maximum sequence length, default 16
        insn_info: instruction information dictionary, format consistent with asm_dataset_preprocessed_compress_simple.py
    """
    # Check input parameters
    if not isinstance(asm_str, str):
        raise TypeError("asm_str must be string type")
    if not isinstance(token2id, dict):
        raise TypeError("token2id must be dict type")
    if not isinstance(id_mapping, dict):
        raise TypeError("id_mapping must be dict type")
    if not asm_str:
        raise ValueError("asm_str cannot be empty")

    try:
        if not insn_info:
            opcode, operands = normalize_insn(asm_str)
            tokens = [opcode] + operands

            # Find token id
            token_ids = []
            for t in tokens:
                tid = token2id.get(t, UNK_ID)
                # token mapping
                mapped_id = id_mapping.get(str(tid), UNK_ID)
                if mapped_id == UNK_ID and tid != UNK_ID:
                    raise ValueError(f"Unable to map token ID: {tid}, corresponding token: {t}")
                token_ids.append(mapped_id)

            # Add special token
            token_ids = [CLS_ID] + token_ids + [SEP_ID]
        else:
            tokens = {
                "token_ids": [],
                "mnemonic_ids": [],
                "op_kind_ids": [],
                "op_id_ids": [],
                "reg_r_ids": [],
                "reg_w_ids": [],
                "eflags_id": []
            }
            opcode_id = id_mapping.get(insn_info["opcode_id"], UNK_ID)
            opnd_ids = [id_mapping.get(opnd_id, UNK_ID) for opnd_id in insn_info["opnd_ids"]]
            mnemonic_id = insn_info["mnemonic_id"]
            op_kind_ids = insn_info["op_kind_ids"]
            op_id_ids = insn_info["op_id_ids"]
            reg_r_ids = insn_info["reg_r_ids"]
            reg_w_ids = insn_info["reg_w_ids"]
            eflags_id = insn_info["eflags_id"]

            tokens["token_ids"].append(opcode_id)
            tokens["mnemonic_ids"].append(mnemonic_id)
            tokens["op_kind_ids"].append(PAD_ID)
            tokens["op_id_ids"].append(PAD_ID)
            tokens["reg_r_ids"].append(PAD_ID)
            tokens["reg_w_ids"].append(PAD_ID)
            tokens["eflags_id"].append(eflags_id)
            for i in range(min(len(opnd_ids), len(op_kind_ids))):
                if len(op_kind_ids) == 3 and len(opnd_ids) == 2 and i == 1:
                    tokens["token_ids"].append(opnd_ids[i])
                    tokens["mnemonic_ids"].append(mnemonic_id)
                    tokens["op_kind_ids"].append(op_kind_ids[i+1])
                    tokens["op_id_ids"].append(op_id_ids[i+1])
                    tokens["reg_r_ids"].append(reg_r_ids[i+1])
                    tokens["reg_w_ids"].append(reg_w_ids[i+1])
                    tokens["eflags_id"].append(eflags_id)
                else:
                    tokens["token_ids"].append(opnd_ids[i])
                    tokens["mnemonic_ids"].append(mnemonic_id)
                    tokens["op_kind_ids"].append(op_kind_ids[i])
                    tokens["op_id_ids"].append(op_id_ids[i])
                    tokens["reg_r_ids"].append(reg_r_ids[i])
                    tokens["reg_w_ids"].append(reg_w_ids[i])
                    tokens["eflags_id"].append(eflags_id)
            # Handle special token
            for key in tokens:
                tokens[key] = [CLS_ID] + tokens[key]
            # Handle sequence length
            if len(tokens["token_ids"]) < max_len:
                for key in tokens:
                    tokens[key] += [PAD_ID] * (max_len - len(tokens[key]))
            elif len(tokens["token_ids"]) > max_len:
                # Truncate to specified length
                for key in tokens:
                    tokens[key] = tokens[key][:max_len]
            return tokens
        
        return token_ids
    except Exception as e:
        raise RuntimeError(f"Error when processing assembly string: {str(e)}")

def get_instruction_encodings(model, asm_instructions, token2id, id_mapping, device="cuda", encoding_type="avg", batch_size=128, encoding=None, layer_idx=None):
    """
    Get encoding for each assembly instruction, supporting multiple encoding methods, using batch processing for acceleration
    
    Args:
        model: loaded model
        asm_instructions: list of assembly instructions
        token2id: mapping from token to id
        id_mapping: mapping from id to compressed id
        device: compute device
        encoding_type: encoding type, optional "avg"(average pooling), "cls"(first token), "last"(last non-PAD token)
        batch_size: batch size, default 128
        encoding: pre-encoding information
        layer_idx: which layer's hidden states to use, default use 9th layer
        
    Returns:
        List of instruction encodings
    """
    encodings = []
    error_count = 0
    
    # Create progress bar object
    total_instructions = len(asm_instructions)
    pbar = tqdm(total=total_instructions, desc="Processing instructions")
    
    # Pre-process all instructions
    all_tokens = []
    valid_indices = []
    
    print(f"Pre-processing assembly instructions...")
    for i, asm in enumerate(tqdm(asm_instructions, desc="Pre-processing instructions")):
        try:
            if encoding:
                insn_info = encoding[i]
            else:
                insn_info = None
            tokens = asm_to_tokens(asm, token2id, id_mapping, insn_info=insn_info)
            all_tokens.append(tokens)
            valid_indices.append(i)
        except Exception as e:
            import traceback
            print(f"Error details: {str(e)}")
            print("Error stack:")
            traceback.print_exc()
            sys.exit(1)
            error_count += 1
            continue
    
    print(f"Starting batch inference, batch size: {batch_size}...")
    # Process in batches
    for i in range(0, len(all_tokens), batch_size):
        # Get current batch
        batch_tokens_list_for_current_batch = all_tokens[i:i+batch_size] # Actual instruction list for current batch
        batch_indices = valid_indices[i:i+batch_size]

        # transform the format of batch_tokens from [{key1: value, key2: value}, ..., {key1: value, key2:value}] to {key1: [value, ..., value], key2: [...]}
        batch_tokens_dict = defaultdict(list)
        for inst in batch_tokens_list_for_current_batch: # Use instruction list for current batch
            for key, value in inst.items():
                batch_tokens_dict[key].append(value)
        batch_tokens = dict(batch_tokens_dict)
        
        for key in batch_tokens:
            batch_tokens[key] = torch.tensor(batch_tokens[key], dtype=torch.long)
        batch_tokens = adjust_batch_to_max_valid_length(batch_tokens)
        
        with torch.no_grad():
            # Create batch tensor
            for key in batch_tokens:
                batch_tokens[key] = batch_tokens[key].to(device)
            
            # Get hidden states and specify using hidden states from layer_idx
            # Need to modify model's forward method to support returning hidden states from specified layer
            hidden = model(batch_tokens, layer_idx=layer_idx)
            
            # Choose different representation methods based on encoding type
            if encoding_type == "cls":
                # Use first token (CLS) state as representation
                # First convert to float32 then numpy to solve BFloat16 compatibility issue
                encoding = hidden[:, 0, :].float().cpu().numpy()
            elif encoding_type == "last":
                # Use last non-PAD token state as representation
                mask = (batch_tokens["token_ids"]!= PAD_ID).float()
                last_indices = mask.sum(dim=1).long() - 1  # Index of last non-PAD token
                batch_indices_tensor = torch.arange(hidden.size(0)).to(device)
                last_hidden = hidden[batch_indices_tensor, last_indices]
                encoding = last_hidden.float().cpu().numpy()
            else:  # Default use average pooling
                # Use average pooling to get representation of entire sequence
                mask = (batch_tokens["token_ids"]!= PAD_ID).float()
                avg_encoding = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                encoding = avg_encoding.float().cpu().numpy()
            
            # Add results to encoding list
            for j, idx in enumerate(batch_indices):
                result = {
                    'instruction': asm_instructions[idx],
                    'encoding': encoding[j].tolist()
                }
                encodings.append(result)
            
            # Update progress bar
            pbar.update(len(batch_indices)) # Use number of instructions in current batch to update progress bar

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


def evaluate_embeddings(encodings, categories):
    """
    Evaluate embedding quality, calculate intra-class and inter-class distances
    
    Args:
        encodings: list of instruction encodings, each element contains 'instruction' and 'encoding'
        categories: corresponding list of categories
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import numpy as np
    from collections import defaultdict
    
    # Extract encoding vectors and categories
    vectors = np.array([item['encoding'] for item in encodings])
    labels = np.array(categories)
    
    # Group by category
    category_vectors = defaultdict(list)
    for vec, cat in zip(vectors, labels):
        category_vectors[cat].append(vec)
    
    # Calculate intra-class distance (average Euclidean distance between samples within each category)
    intra_class_distances = {}
    for cat, vecs in category_vectors.items():
        if len(vecs) <= 1:
            intra_class_distances[cat] = 0
            continue
            
        vecs = np.array(vecs)
        total_dist = 0
        count = 0
        
        # Calculate distances between all sample pairs within category
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                total_dist += np.linalg.norm(vecs[i] - vecs[j])
                count += 1
        
        # Calculate average distance
        if count > 0:
            intra_class_distances[cat] = total_dist / count
        else:
            intra_class_distances[cat] = 0
    
    # Calculate inter-class distance (average Euclidean distance between different category centers)
    category_centers = {}
    for cat, vecs in category_vectors.items():
        category_centers[cat] = np.mean(vecs, axis=0)
    
    inter_class_distances = {}
    categories_list = list(category_centers.keys())
    for i, cat1 in enumerate(categories_list):
        cat_distances = {}
        for j, cat2 in enumerate(categories_list):
            if i != j:
                dist = np.linalg.norm(category_centers[cat1] - category_centers[cat2])
                cat_distances[cat2] = dist
        inter_class_distances[cat1] = cat_distances
    
    # Calculate separation degree for each category (inter-class distance / intra-class distance)
    separation_ratios = {}
    for cat in category_vectors.keys():
        if intra_class_distances[cat] > 0 and cat in inter_class_distances:
            # Calculate average distance to other categories
            avg_inter_dist = np.mean(list(inter_class_distances[cat].values())) if inter_class_distances[cat] else 0
            separation_ratios[cat] = avg_inter_dist / intra_class_distances[cat]
        else:
            separation_ratios[cat] = float('inf') if cat in inter_class_distances and inter_class_distances[cat] else 0
    
    # Calculate overall metrics
    avg_intra_dist = np.mean(list(intra_class_distances.values()))
    
    # Calculate average distance between all category centers
    total_inter_dist = 0
    count = 0
    for cat1, distances in inter_class_distances.items():
        for cat2, dist in distances.items():
            total_inter_dist += dist
            count += 1
    avg_inter_dist = total_inter_dist / count if count > 0 else 0
    
    # Calculate overall separation degree
    overall_separation = avg_inter_dist / avg_intra_dist if avg_intra_dist > 0 else float('inf')
    
    # Use sklearn's clustering evaluation metrics
    try:
        silhouette = silhouette_score(vectors, labels)
    except:
        silhouette = -1  # If only one category, silhouette_score will fail
        
    try:
        davies_bouldin = davies_bouldin_score(vectors, labels)
    except:
        davies_bouldin = float('inf')
        
    try:
        calinski_harabasz = calinski_harabasz_score(vectors, labels)
    except:
        calinski_harabasz = 0
    
    # Return evaluation results
    return {
        'avg_intra_class_distance': avg_intra_dist,
        'avg_inter_class_distance': avg_inter_dist,
        'overall_separation_ratio': overall_separation,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'intra_class_distances': intra_class_distances,
        'separation_ratios': separation_ratios
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
    k_best = 12
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
    plt.xlabel('Number of clusters (k)') # Number of clusters (k) -> Number of clusters (k)
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k') # Elbow method to determine optimal k -> Elbow Method for Optimal k
    plt.grid(True)
    
    # Plot silhouette coefficient graph
    plt.subplot(2, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters (k)') # Number of clusters (k) -> Number of clusters (k)
    plt.ylabel('Silhouette Score') # Silhouette coefficient -> Silhouette Score
    plt.title('Silhouette Score for Different k Values') # Silhouette coefficient for different k values -> Silhouette Score for Different k Values
    plt.grid(True)
    
    # Plot t-SNE dimensionality reduction results, colored by K-means clustering labels
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label') # Cluster label -> Cluster Label
    plt.title(f't-SNE Dim. Reduction (K-means Clustering, k={k_best})') # t-SNE dim reduction (K-means clustering, k={k_best}) -> t-SNE Dim. Reduction (K-means Clustering, k={k_best})
    plt.xlabel('t-SNE Dimension 1') # t-SNE dimension 1 -> t-SNE Dimension 1
    plt.ylabel('t-SNE Dimension 2') # t-SNE dimension 2 -> t-SNE Dimension 2
    
    # If original categories exist, plot t-SNE using original categories for coloring
    if original_labels is not None:
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=original_labels, cmap='tab20', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Original Label') # Original category -> Original Label
        plt.title('t-SNE Dim. Reduction (Original Labels)') # t-SNE dim reduction (original categories) -> t-SNE Dim. Reduction (Original Labels)
        plt.xlabel('t-SNE Dimension 1') # t-SNE dimension 1 -> t-SNE Dimension 1
        plt.ylabel('t-SNE Dimension 2') # t-SNE dimension 2 -> t-SNE Dimension 2
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Visualization results saved to: {output_file}") # Visualization results saved to -> Visualization results saved to
    
    # Save clustering results
    cluster_file = output_file.replace('.pdf', '_clusters.txt')
    with open(cluster_file, 'w') as f:
        for i, label in enumerate(cluster_labels):
            f.write(f"{instructions[i]}\t{label}\n")
    print(f"Clustering results saved to: {cluster_file}") # Clustering results saved to -> Clustering results saved to
    
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
            f.write(f"Cluster {label} (Total {len(samples)} samples):\n") # Cluster {label} (total {len(samples)} samples): -> Cluster {label} (Total {len(samples)} samples):
            
            # Calculate distance to cluster center
            cluster_center = kmeans.cluster_centers_[label]
            samples_with_dist = []
            for idx, insn in samples:
                dist = np.linalg.norm(vectors[idx] - cluster_center)
                samples_with_dist.append((idx, insn, dist))
            
            # Sort by distance, output all samples
            samples_with_dist.sort(key=lambda x: x[2])
            for i, (idx, insn, dist) in enumerate(samples_with_dist):
                f.write(f"  {i+1}. {insn} (Distance: {dist:.4f})\n")
            f.write("\n")
    
    print(f"Cluster samples saved to: {cluster_samples_file}") # Cluster samples saved to -> Cluster samples saved to
    
    return cluster_labels

def evaluate_kmeans_clustering(vectors, labels):
    """
    Evaluate K-means clustering quality
    
    Args:
        vectors: vector data
        labels: clustering labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    try:
        silhouette = silhouette_score(vectors, labels)
    except:
        silhouette = -1
        
    try:
        davies_bouldin = davies_bouldin_score(vectors, labels)
    except:
        davies_bouldin = float('inf')
        
    try:
        calinski_harabasz = calinski_harabasz_score(vectors, labels)
    except:
        calinski_harabasz = 0
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get encoding representations of assembly instructions using RWKV7 model and perform clustering analysis")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", "-i", required=True, help="Path to file containing assembly instructions")
    parser.add_argument("--output", "-o", default="instruction_encodings.pkl", help="Path to output file for encodings")
    parser.add_argument("--token2id", "-t", default="./tokens/assemble_tokens.txt", help="File mapping token to id")
    parser.add_argument("--id_mapping", "-m", default="./tokens/id_mapping.json", help="File mapping id to compressed id")
    parser.add_argument("--encoding_type", "-e", default="avg", choices=["avg", "cls", "last"], help="Encoding type")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--layer_idx", "-l", type=int, default=None, help="Which layer's hidden states to use")
    parser.add_argument("--device", "-d", default="cuda", help="Compute device")
    parser.add_argument("--visualize", "-v", action="store_true", help="Whether to visualize")
    parser.add_argument("--categories", default=None, help="Path to file containing instruction classification for comparison")
    parser.add_argument("--encoding_path", default=None, help="Path to file containing pre-encoding information")
    parser.add_argument("--max_k", type=int, default=30, help="Maximum k value for K-means clustering, used for Elbow method")
    
    args = parser.parse_args()
    
    # Load token2id and id_mapping
    # Load token mapping
    token2id_array = []
    with open(args.token2id, "r", encoding="utf-8") as f:
        for line in f:
            token2id_array.append(line.strip())

    token2id = {token: idx for idx, token in enumerate(token2id_array)}
    id_mapping = load_json(args.id_mapping)
    
    # Read assembly instructions
    with open(args.input, 'r') as f:
        asm_instructions = [line.strip() for line in f]
    
    # Load pre-encoding information (if provided)
    encoding = None
    if args.encoding_path and os.path.exists(args.encoding_path):
        with open(args.encoding_path, 'rb') as f:
            encoding = pickle.load(f)
        print(f"Loaded pre-encoding information: {args.encoding_path}")
    
    # Load model
    model = load_model(args.checkpoint, device=args.device)
    
    # Get instruction encodings
    encodings = get_instruction_encodings(
        model, 
        asm_instructions, 
        token2id, 
        id_mapping, 
        device=args.device,
        encoding_type=args.encoding_type,
        batch_size=args.batch_size,
        encoding=encoding,
        layer_idx=args.layer_idx
    )
    
    # Save encodings
    save_encodings(encodings, args.output)
    
    # Visualize
    if args.visualize:
        output_file = args.output.replace('.pkl', '_tsne.pdf')
        visualize_tsne(encodings, output_file=output_file, categories_file=args.categories, max_k=args.max_k)


