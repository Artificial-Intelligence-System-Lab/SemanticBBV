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
from asm_dataset_preprocessed_fine import AsmVocab


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

def load_model(checkpoint_path, vocabs, device="cuda", n_embd=768, n_layer=6, head_size=64, use_bf16=True):
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
            self.head_size = head_size
    
    class RWKV7Model(nn.Module):
        def __init__(self, args, vocab_size):
            super().__init__()
            model_args = type('', (), {})()
            model_args.n_embd = args.n_embd
            model_args.n_layer = args.n_layer
            model_args.head_size_a = args.head_size

            self.asm_embd = nn.Embedding(vocab_size["asm"], args.n_embd)
            self.type_embd = nn.Embedding(vocab_size["type"], args.n_embd)
            self.reg_embd = nn.Embedding(vocab_size["reg"], args.n_embd)
            self.rw_embd = nn.Embedding(vocab_size["rw"], args.n_embd)
            self.eflag_embd = nn.Embedding(vocab_size["eflag"], args.n_embd)

            self.model = RWKV(model_args)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 as ignore index

            # Add two different output layers for token-level and instruction-level MLM respectively
            self.token_mlm_head = nn.Linear(args.n_embd, args.vocab_size)
            self.instr_mlm_head = nn.Linear(args.n_embd, args.vocab_size)

        def forward(self, inputs, layer_idx=None):
            # Forward pass for token-level task
            # Embed and sum each dimension
            asm_emb = self.asm_embd(inputs['asm'])
            type_emb = self.type_embd(inputs['type'])
            reg_emb = self.reg_embd(inputs['reg'])
            rw_emb = self.rw_embd(inputs['rw'])
            eflag_embd = self.eflag_embd(inputs['eflag'])

            # Sum all embeddings
            combined_emb = asm_emb + type_emb + reg_emb + rw_emb + eflag_embd

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


def get_instruction_encodings(model, asm_instructions,device="cuda", encoding_type="avg", batch_size=128, encoding=None, layer_idx=None):
    """
    Get encoding for each assembly instruction, supporting multiple encoding methods, using batch processing for acceleration
    
    Args:
        model: loaded model
        asm_instructions: list of assembly instructions
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

    if encoding is None:
        print("Encoding not provided")
        sys.exit(0)
    
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


def visualize_tsne(encodings, output_file="tsne_visualization.pdf", categories_file=None):
    """
    Visualize instruction encodings using t-SNE, optionally color by category information
    
    Args:
        encodings: instruction encoding list
        output_file: output image file path
        categories_file: file path containing instruction classification, use category information for coloring if provided
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Error: please install necessary libraries: pip install scikit-learn matplotlib")
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
    
    # If category file provided, load category information
    if categories_file and os.path.exists(categories_file):
        # Read all instructions and corresponding categories
        with open(categories_file, 'r') as f:
            categories = [line.strip() for line in f]
        
        # Ensure number of lines in category file matches number of instructions
        if len(categories) != len(instructions):
            print("Error: number of lines in category file does not match number of instructions")
            sys.exit(1)
        else:
            use_categories = True
            # Use categories as labels
            labels = categories
            print(f"Categories: {len(categories)}")
            
            # Calculate embedding quality evaluation metrics
            metrics = evaluate_embeddings(encodings, categories)
            
            # Output evaluation results
            print("\nEmbedding quality evaluation metrics:")
            print(f"Average intra-class distance: {metrics['avg_intra_class_distance']:.4f}")
            print(f"Average inter-class distance: {metrics['avg_inter_class_distance']:.4f}")
            print(f"Overall separation ratio (inter-class/intra-class): {metrics['overall_separation_ratio']:.4f}")
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (closer to 1 is better)")
            print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
            print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
            
            # Output detailed evaluation results to file
            metrics_file = output_file.replace('.pdf', '_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write("Category\tIntra-class distance\tSeparation degree\n")
                for cat in sorted(metrics['intra_class_distances'].keys()):
                    intra_dist = metrics['intra_class_distances'][cat]
                    sep_ratio = metrics['separation_ratios'][cat]
                    f.write(f"{cat}\t{intra_dist:.4f}\t{sep_ratio:.4f}\n")
                
                f.write("\nOverall metrics:\n")
                f.write(f"Average intra-class distance: {metrics['avg_intra_class_distance']:.4f}\n")
                f.write(f"Average inter-class distance: {metrics['avg_inter_class_distance']:.4f}\n")
                f.write(f"Overall separation ratio (inter-class/intra-class): {metrics['overall_separation_ratio']:.4f}\n")
                f.write(f"Silhouette Score: {metrics['silhouette_score']:.4f}\n")
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}\n")
                f.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}\n")
            
            print(f"Detailed evaluation metrics saved to: {metrics_file}")
    
    # Assign different colors to different labels
    unique_labels = list(set(labels))
    
    # Try to convert labels to numeric for sorting (if labels are numeric or contain numbers)
    try:
        # For pure numeric labels
        numeric_labels = [int(label) if label.isdigit() else float(label) for label in unique_labels]
        sorted_labels = [label for _, label in sorted(zip(numeric_labels, unique_labels))]
    except (ValueError, TypeError):
        # If conversion fails, sort by string
        sorted_labels = sorted(unique_labels)
    
    # Use continuous color map to ensure adjacent numeric labels have similar colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(sorted_labels)}
    
    # Create 2D figure
    plt.figure(figsize=(14, 12))
    
    # Plot 2D scatter
    for label in sorted_labels:
        # Find all points belonging to current label
        mask = [l == label for l in labels]
        plt.scatter(
            reduced_data[mask, 0], 
            reduced_data[mask, 1], 
            color=label_to_color[label], 
            alpha=0.7,
            label=label
        )
    
    # Add some labels (max 30, avoid overcrowding)
    if len(instructions) <= 30:
        for i, (x, y) in enumerate(reduced_data):
            plt.text(x, y, labels[i], fontsize=8)
    
    # Set title and axis labels
    if use_categories:
        plt.title('2D t-SNE Visualization: Assembly Instruction Encodings Colored by Category', fontsize=14)
    else:
        plt.title('2D t-SNE Visualization: Assembly Instruction Encodings Colored by Opcode', fontsize=14)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Improve legend display
    if len(unique_labels) <= 30:  # Increase label display limit
        # If too many labels, place legend outside figure on right
        if len(unique_labels) > 15:
            plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), 
                     title="Categories" if use_categories else "Opcodes", fontsize=9)
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust main plot area to leave space for legend
        else:
            plt.legend(loc='best', 
                     title="Categories" if use_categories else "Opcodes", fontsize=10)
            plt.tight_layout()
    else:
        # If labels too many, only show top 30 most common labels
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Select 30 labels with highest frequency
        top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        top_label_names = [label for label, _ in top_labels]
        
        # Redraw legend containing only top 30 labels
        handles = []
        for label in top_label_names:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=label_to_color[label], label=label, markersize=8))
        
        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.1, 0.5), 
                 title="Top 30 Categories" if use_categories else "Top 30 Opcodes", fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save or display image
    if output_file:
        # Modify output filename to reflect 2D visualization
        output_file_2d = output_file.replace('.pdf', '_2d.pdf')
        plt.savefig(output_file_2d, format='pdf', bbox_inches='tight')
        print(f"2D t-SNE visualization saved to: {output_file_2d}")
    else:
        plt.show()

# Add logic in main function to check if encoding file exists
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Assembly instruction to token tool")
    parser.add_argument("--vocabs_dir", type=str, help="Path to vocabulary directory")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, help="Path to input file, one assembly instruction per line")
    parser.add_argument("--output", type=str, default="instruction_encodings.pkl", help="Path to output encoding file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    # Add model parameters
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of model layers")
    parser.add_argument("--head_size", type=int, default=64, help="Attention head size")
    # Add t-SNE related parameters
    parser.add_argument("--categories", type=str, help="Path to file containing instruction classification for t-SNE visualization")
    parser.add_argument("--tsne_output", type=str, default="tsne_visualization.pdf", help="t-SNE visualization output file path")
    parser.add_argument("--encoding_path", type=str, help="encoding pickle file")
    # Add encoding type parameter
    parser.add_argument("--encoding_type", type=str, default="avg", 
                        choices=["avg", "cls", "last"],
                        help="Encoding type: avg(average pooling), cls(first token), last(last non-PAD token)")
    
    args = parser.parse_args()
    
    # Determine output file name
    output_file = f"instruction_encodings_{args.encoding_type}.pkl"
    
    # Check if encoding file already exists
    if os.path.exists(output_file):
        print(f"Found existing encoding file: {output_file}")
        try:
            with open(output_file, "rb") as f:
                import pickle
                encodings = pickle.load(f)
            print(f"Successfully loaded encodings for {len(encodings)} instructions")
            
            # Print some encoding examples
            if encodings:
                print("\nEncoding examples:")
                example = encodings[0]
                print(f"Instruction: {example['instruction']}")
                print(f"Encoding dimension: {len(example['encoding'])}")
                
                # Perform t-SNE visualization
                visualize_tsne(encodings, args.tsne_output, args.categories)
                
            sys.exit(0)  # Successfully loaded, exit program
        except Exception as e:
            print(f"Failed to load existing encoding file: {e}, will regenerate encodings")
    
    vocab_config = {
        "asm": "asm_tokens.txt",
        "type": "type_tokens.txt",
        "reg": "reg_tokens.txt",
        "rw": "rw_tokens.txt",
        "eflag": "eflag_tokens.txt"
    }

    vocabs = {}
    for key, filename in vocab_config.items():
        vocab_path = os.path.join(args.vocabs_dir, filename)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please ensure all vocabulary files exist in '{args.vocabs_dir}' directory.")
        vocab = AsmVocab()
        vocab.load(vocab_path)
        vocabs[key] = vocab
    
    
    # Encoding mode
    if not args.checkpoint:
        print("Error: encoding mode requires providing model checkpoint path (--checkpoint)")
        sys.exit(1)
        
    print(f"Using device: {args.device}")
    print(f"Model parameters: embedding dimension={args.n_embd}, layers={args.n_layer}, attention head size={args.head_size}")
    
    # Load model, passing model parameters
    model = load_model(
        checkpoint_path=args.checkpoint, 
        vocabs=vocabs,
        device=args.device,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        head_size=args.head_size,
        use_bf16=True
    )
    
    # Load assembly instructions
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            asm_instructions = [line.strip() for line in f if line.strip()]
    else:
        print("No input file provided")
        sys.exit(1)
        
    print(f"Loaded {len(asm_instructions)} assembly instructions")
    
    if args.encoding_path and os.path.exists(args.encoding_path):
        with open(args.encoding_path, "rb") as f:
            encoding_dict = pickle.load(f)
    else:
        encoding_dict = None
    # Get instruction encodings
    encodings = get_instruction_encodings(
        model, 
        asm_instructions, 
        token2id, 
        id_mapping, 
        args.device,
        encoding_type=args.encoding_type,
        encoding=encoding_dict
        )
    
    save_encodings(encodings, output_file)
    
    # Print some encoding examples
    if encodings:
        print("\nEncoding examples:")
        example = encodings[0]
        print(f"Instruction: {example['instruction']}")
        print(f"Encoding dimension: {len(example['encoding'])}")
        
        # Perform t-SNE visualization
        visualize_tsne(encodings, args.tsne_output, args.categories)


