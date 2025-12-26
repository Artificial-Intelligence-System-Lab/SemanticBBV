import sys
import os
import argparse
import torch
import pickle
import numpy as np
from tqdm import tqdm

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from pre_processing.tokenizer import tokenize_binary_instruction
from asm_dataset_preprocessed_fine import AsmVocab
from train_rwkv7_deepspeed import RWKV7Model

# Define a modified SiameseRWKV7Model that accepts only single input
class BasicBlockEncoder(torch.nn.Module):
    """
    Modified SiameseRWKV7Model that accepts only single input, used to generate embedding of basic blocks
    """
    
    def __init__(self, pretrained_model, args, encoding_dim=64):
        super().__init__()
        
        # Save components of pre-trained model
        self.num_embedding_types = pretrained_model.num_embedding_types
        
        # Copy embedding layer
        self.asm_embd = pretrained_model.asm_embd
        self.mne_embd = pretrained_model.mne_embd
        self.type_embd = pretrained_model.type_embd
        self.reg_embd = pretrained_model.reg_embd
        self.rw_embd = pretrained_model.rw_embd
        self.eflag_embd = pretrained_model.eflag_embd
        
        # Copy projection layer and normalization layer
        self.embedding_projection_layer = pretrained_model.embedding_projection_layer
        
        # Copy RWKV backbone network
        self.model = pretrained_model.model
        
        # Improved encoding head
        self.encoding_head = torch.nn.Sequential(
            torch.nn.Linear(args.n_embd, encoding_dim),
            torch.nn.Dropout(0.1),
        )
        
        # Initialize attention query vectors
        self.attention_query = torch.nn.Sequential(
            torch.nn.Linear(args.n_embd, 256),  # Dimension reduction to reduce parameters
            torch.nn.GELU(),
            torch.nn.Linear(256, 1)
        )
        
        self.encoding_dim = encoding_dim
        
    def _get_sequence_info(self, sample_dict):
        """Get sequence information - find first non-zero position starting from tail"""
        asm_tokens = sample_dict['asm']  # shape: [batch_size, seq_len]

        # Method: use torch.where to find position of last non-zero element in each row
        batch_size, seq_len = asm_tokens.shape

        # Create indices for each position
        indices = torch.arange(seq_len, device=asm_tokens.device).expand(batch_size, seq_len)

        # Create mask marking non-zero positions, set indices at zero positions to -1
        non_zero_mask = (asm_tokens != 0)
        masked_indices = torch.where(non_zero_mask, indices, torch.tensor(-1, device=asm_tokens.device))

        # Find maximum index in each row (position of last non-zero element)
        last_non_zero_pos = masked_indices.max(dim=1)[0]

        # Sequence length = position of last non-zero element + 1
        seq_lengths = last_non_zero_pos + 1

        # Handle all-zero sequences (case where last_non_zero_pos is -1)
        seq_lengths = torch.clamp(seq_lengths, min=1)

        return seq_lengths
    
    def pooling(self, x, seq_lengths):
        """Attention pooling"""
        # Ensure pooling uses float32 precision
        if x.dtype == torch.bfloat16:
            x = x.float()
        
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Create mask: for each sample, only positions within sequence length are True
        seq_lengths = seq_lengths.long()
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        
        # Calculate attention scores - use Sequential module
        attention_scores = self.attention_query(x)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask (set positions outside mask to a very small negative number)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Weighted summation
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return pooled
    
    def _encode_single_sample(self, sample_dict):
        """Encode single sample"""
        # Get embedding of each feature
        asm_emb = self.asm_embd(sample_dict['asm'])
        mne_emb = self.mne_embd(sample_dict['mne'])
        type_emb = self.type_embd(sample_dict['type'])
        reg_emb = self.reg_embd(sample_dict['reg'])
        rw_emb = self.rw_embd(sample_dict['rw'])
        eflag_emb = self.eflag_embd(sample_dict['eflag'])
        
        seq_lengths = self._get_sequence_info(sample_dict)
        
        # Concatenate all embeddings
        combined_emb = torch.cat(
            (asm_emb, mne_emb, type_emb, reg_emb, rw_emb, eflag_emb),
            dim=-1
        )
        
        # Through projection layer
        projected_emb = self.embedding_projection_layer(combined_emb)
        
        # Through RWKV backbone network
        hidden_states = self.model(projected_emb)

        # Apply improved pooling strategy
        pooled_output = self.pooling(hidden_states, seq_lengths)

        # Through encoding head to get final encoding
        encoding = self.encoding_head(pooled_output)
        
        return encoding
    
    def encode(self, sample_dict):
        """Encode single sample (for inference)"""
        with torch.no_grad():
            return self._encode_single_sample(sample_dict)
    
    def encode_batch(self, sample_dicts):
        """Batch encode samples"""
        with torch.no_grad():
            # Convert sample list to batch format
            batch_dict = {}
            for key in sample_dicts[0].keys():
                batch_dict[key] = torch.stack([sample[key] for sample in sample_dicts])
            
            return self._encode_single_sample(batch_dict)

def load_model_from_checkpoint(checkpoint_path, vocabs_dir, device='cuda', **model_args):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load vocabulary
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
        vocab_path = os.path.join(vocabs_dir, filename)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        vocab = AsmVocab()
        vocab.load(vocab_path)
        vocabs[key] = vocab
    
    vocabs_size = {key: vocab.length() for key, vocab in vocabs.items()}
    
    # Build args object
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(
        n_embd=model_args.get('n_embd', 768),
        n_layer=model_args.get('n_layer', 6),
        head_size=model_args.get('head_size', 64)
    )
    
    # Create pre-trained model
    pretrained_model = RWKV7Model(args, vocabs_size)
    
    # Create encoder model
    model = BasicBlockEncoder(
        pretrained_model, 
        args,
        encoding_dim=model_args.get('encoding_dim', 128)
    )
    
    # Load trained weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Set mixed precision
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("Using BF16 mixed precision")
        model.asm_embd = model.asm_embd.to(torch.bfloat16)
        model.mne_embd = model.mne_embd.to(torch.bfloat16)
        model.type_embd = model.type_embd.to(torch.bfloat16)
        model.reg_embd = model.reg_embd.to(torch.bfloat16)
        model.rw_embd = model.rw_embd.to(torch.bfloat16)
        model.eflag_embd = model.eflag_embd.to(torch.bfloat16)
        model.embedding_projection_layer = model.embedding_projection_layer.to(torch.bfloat16)
        model.model = model.model.to(torch.bfloat16)
        # model.encoding_head = model.encoding_head.to(torch.bfloat16)
    
    model.eval()
    model.to(device)
    
    print(f"Model loaded, encoding dimension: {model.encoding_dim}")
    return model, vocabs

def prepare_sample_for_model(tokens, vocabs):
    """Convert token sequence to model input format"""
    # Initialize input dictionary
    sample_dict = {
        'asm': [],
        'mne': [],
        'type': [],
        'reg': [],
        'rw': [],
        'eflag': []
    }
    
    # Fill input dictionary
    for token in tokens:
        sample_dict['asm'].extend([vocabs['asm'].get_id(tok) for tok in token['asm']])
        sample_dict['mne'].extend([vocabs['mne'].get_id(token['mne'])] * len(token['asm']))
        sample_dict['type'].extend([vocabs['type'].get_id(tok) for tok, count in token['type'] for _ in range(count)])
        sample_dict['reg'].extend([vocabs['reg'].get_id(tok) for tok in token['reg']])
        sample_dict['rw'].extend([vocabs['rw'].get_id(tok) for tok, count in token['rw'] for _ in range(count)])
        sample_dict['eflag'].extend([vocabs['eflag'].get_id(token['eflag'])] * len(token['asm']))
    
    # Ensure all feature lengths are consistent
    seq_len = len(sample_dict['asm'])
    for key in sample_dict:
        if len(sample_dict[key]) != seq_len:
            raise ValueError(f"Feature {key} length {len(sample_dict[key])} does not match asm length {seq_len}")
    
    # Convert to tensor, but do not add batch dimension or pad
    for key in sample_dict:
        sample_dict[key] = torch.tensor(sample_dict[key], dtype=torch.long)
    
    return sample_dict

def main():
    parser = argparse.ArgumentParser(description='Convert binary basic blocks to vector representations')
    parser.add_argument('pkl_file', type=str, help='path to pickle file containing basic blocks')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--vocab_dir', type=str, required=True, help='path to vocabulary directory')
    parser.add_argument('--output', type=str, default=None, help='output file path, default is input filename.vectors.pkl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device type')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--n_embd', type=int, default=768, help='model embedding dimension')
    parser.add_argument('--n_layer', type=int, default=6, help='model number of layers')
    parser.add_argument('--head_size', type=int, default=64, help='attention head size')
    parser.add_argument('--encoding_dim', type=int, default=128, help='encoding dimension')
    
    args = parser.parse_args()
    
    # Set output file path
    if args.output is None:
        args.output = os.path.splitext(args.pkl_file)[0] + '.vectors.pkl'
    
    # Load model and vocabulary
    model, vocabs = load_model_from_checkpoint(
        args.checkpoint, 
        args.vocab_dir, 
        device=args.device,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        head_size=args.head_size,
        encoding_dim=args.encoding_dim
    )
    
    # Load pickle file
    print(f"Loading pickle file: {args.pkl_file}")
    with open(args.pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Process each basic block
    results = {}
    batch_addresses = []
    batch_samples = []
    
    print(f"Processing {len(data)} basic blocks...")
    for address, binary_data in tqdm(data.items()):
        # Convert Python list to bytes object
        binary_bytes = bytes(binary_data)
        
        # Use tokenize_binary_instruction to parse binary data
        tokens = tokenize_binary_instruction(binary_bytes, address)
        
        # Prepare model input
        sample = prepare_sample_for_model(tokens, vocabs)
        
        # Add to batch
        batch_addresses.append(address)
        batch_samples.append(sample)
        
        # Encode when batch reaches specified size or all data processed
        if len(batch_samples) >= args.batch_size or address == list(data.keys())[-1]:
            # Find longest sequence length in current batch and align to multiple of 16
            max_len_in_batch = max([sample['asm'].size(0) for sample in batch_samples])
            # Round up to multiple of 16
            max_len_in_batch = ((max_len_in_batch + 15) // 16) * 16
            if max_len_in_batch > 512:
                print("longer:", max_len_in_batch)
            
            # Pad samples to batch longest sequence and move to device
            batch_dict = {}
            for key in batch_samples[0].keys():
                # Pad each sample
                padded_samples = []
                for sample in batch_samples:
                    sample_len = sample[key].size(0)
                    if sample_len < max_len_in_batch:
                        # Pad to batch longest length
                        padding = torch.zeros(max_len_in_batch - sample_len, dtype=torch.long)
                        padded_sample = torch.cat([sample[key], padding])
                    else:
                        padded_sample = sample[key]
                    padded_samples.append(padded_sample.unsqueeze(0))
                
                # Concatenate as batch tensor and move to device
                batch_dict[key] = torch.cat(padded_samples).to(args.device)
            
            # Encode batch
            with torch.no_grad():
                encodings = model.encode(batch_dict)
            
            # Save results
            for i, addr in enumerate(batch_addresses):
                results[addr] = encodings[i].cpu().numpy()
            
            # Clear batch
            batch_addresses = []
            batch_samples = []
    
    # Save results
    print(f"Saving results to: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Processing complete, generated {len(results)} vector representations")

if __name__ == '__main__':
    main()

