#!/usr/bin/env python3
"""
CPI task-specific Set-Transformer model fine-tuning script
Based on train_bbv_model.py and cpi_prediction_evaluation.py
Specifically for fine-tuning models for CPI prediction tasks
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import time
from typing import Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import random
import json
import gzip
import pickle
import re
from datetime import datetime
from collections import defaultdict

# Add project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, 'Set-Transformer'))

from evaluate_bbv_model import SetTransformerModel


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


class CPIDataset(Dataset):
    """
    Dataset class for CPI prediction task
    Load data from BBV files, mapping files and JSON statistics files
    """
    
    def __init__(self, bbv_files, map_files, json_files):
        """
        Initialize CPI dataset
        
        Args:
            bbv_files: List of BBV file paths
            map_files: List of mapping file paths
            json_files: List of JSON statistics file paths
        """
        self.samples = []
        
        # Load all data files
        for bbv_file, map_file, json_file in zip(bbv_files, map_files, json_files):
            self._load_data(bbv_file, map_file, json_file)
        
        print(f"Loaded {len(self.samples)} CPI training samples")
    
    def _load_map(self, path: str):
        """
        Load mapping file
        
        Args:
            path: Mapping file path
            
        Returns:
            Mapping dictionary
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_reference_stats(self, ref_json_path: str):
        """
        Load reference simulation statistics data
        
        Args:
            ref_json_path: Reference JSON file path
            
        Returns:
            Dictionary containing simulation statistics data
        """
        with open(ref_json_path, 'r') as f:
            return json.load(f)
    
    def _vectors_mapping(self, input_gz: str, map_file: str):
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
        
        # Load ID mapping
        id_map = self._load_map(map_file)
        
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
        
        return vector_embs
    
    def _calculate_true_cpi(self, ref_stats):
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
    
    def _load_data(self, bbv_file, map_file, json_file):
        """
        Load data from single file combination
        
        Args:
            bbv_file: BBV file path
            map_file: Mapping file path
            json_file: JSON statistics file path
        """
        try:
            # Load BBV data and embeddings
            vector_embs = self._vectors_mapping(bbv_file, map_file)
            
            # Load reference statistics data and calculate true CPI
            ref_stats = self._load_reference_stats(json_file)
            true_cpis = self._calculate_true_cpi(ref_stats)
            
            # Ensure data length consistency
            min_len = min(len(vector_embs), len(true_cpis))
            vector_embs = vector_embs[:min_len]
            true_cpis = true_cpis[:min_len]
            
            # Add to sample list
            for (embs, weights), cpi in zip(vector_embs, true_cpis):
                self.samples.append({
                    'embeddings': embs,
                    'weights': weights,
                    'cpi': float(cpi)
                })
                
        except Exception as e:
            print(f"Error loading data file {bbv_file}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get single sample
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Dictionary containing embeddings, weights and CPI
        """
        sample = self.samples[idx]
        
        # Process embedding vector
        embs = sample['embeddings']
        weights = sample['weights']
        
        # Ensure embeddings are numpy array
        if isinstance(embs[0], list):
            embs = [np.array(emb, dtype=np.float32) for emb in embs]
        elif not isinstance(embs[0], np.ndarray):
            embs = [np.array(emb, dtype=np.float32) for emb in embs]
        
        # Convert to tensor list, no padding
        emb_tensors = [torch.from_numpy(emb) for emb in embs]
        weight_tensors = [torch.tensor(weight, dtype=torch.float32) for weight in weights]
        
        return {
            'embeddings': emb_tensors,
            'weights': weight_tensors,
            'cpi': torch.tensor(sample['cpi'], dtype=torch.float32),
            'seq_len': len(emb_tensors)
        }


def collate_fn(batch):
    """
    Data batch processing function, dynamic padding to longest sequence in batch
    
    Args:
        batch: List of batch data
        
    Returns:
        dict: Batch processed data dictionary
    """
    # Find longest sequence length in batch
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    # Create padded tensor
    embeddings = torch.zeros(batch_size, max_seq_len, 128, dtype=torch.float32)
    weights = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    cpis = torch.stack([item['cpi'] for item in batch])
    
    # Fill data
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        # Pad embedding vector
        for j, emb in enumerate(item['embeddings']):
            embeddings[i, j] = emb
        # Pad weights
        for j, weight in enumerate(item['weights']):
            weights[i, j] = weight
    
    return {
        'embeddings': embeddings,
        'weights': weights,
        'cpi': cpis
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train one epoch
    
    Args:
        model: Model
        dataloader: Data loader
        optimizer: Optimizer
        device: Computing device
        epoch: Current epoch number
        
    Returns:
        float: Average loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Move data to device
        embeddings = batch['embeddings'].to(device)
        weights = batch['weights'].to(device)
        target_cpi = batch['cpi'].to(device)
        
        # Forward propagation
        optimizer.zero_grad()
        
        # Directly use model's _process_sample method to get set embedding
        weight_expanded = weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Get set embedding (without using torch.no_grad())
        _, set_embedding = model._process_sample(embeddings, weights, return_set_embedding=True)
        
        # Get prediction through CPI prediction head
        predicted_cpi = model.cpi_head(set_embedding).squeeze(-1)
        
        # Calculate loss (use Huber loss, more robust to outliers)
        loss = F.huber_loss(predicted_cpi, target_cpi, reduction='mean', delta=1.0)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'avg_loss': f'{total_loss/num_batches:.6f}'
        })
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, device):
    """
    Validate one epoch
    
    Args:
        model: Model
        dataloader: Data loader
        device: Computing device
        
    Returns:
        tuple: (Average loss, MAE, RMSE)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move data to device
            embeddings = batch['embeddings'].to(device)
            weights = batch['weights'].to(device)
            target_cpi = batch['cpi'].to(device)
            
            # Forward propagation
            _, set_embedding = model._process_sample(embeddings, weights, return_set_embedding=True)
            predicted_cpi = model.cpi_head(set_embedding).squeeze(-1)
            
            # Calculate loss
            loss = F.huber_loss(predicted_cpi, target_cpi, reduction='mean', delta=1.0)
            
            # Calculate evaluation metrics
            mae = F.l1_loss(predicted_cpi, target_cpi, reduction='mean')
            rmse = torch.sqrt(F.mse_loss(predicted_cpi, target_cpi, reduction='mean'))
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches, total_rmse / num_batches


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        checkpoint_path: Checkpoint path
        device: Computing device
        
    Returns:
        int: Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resume training from checkpoint, starting epoch: {start_epoch}")
    
    return start_epoch


def main():
    """
    Main function: Execute CPI model fine-tuning process
    """
    parser = argparse.ArgumentParser(description='CPI task-specific Set-Transformer model fine-tuning script')
    
    # Data parameters
    parser.add_argument('--data_dir', required=True, help='Path to data directory')
    parser.add_argument('--file_list', required=True, help='File list, format: bbv_file,map_file,json_file one group per line')
    parser.add_argument('--train_ratio', type=float, default=0.2, help='Training set ratio')
    
    # Model parameters
    parser.add_argument('--st_dim', type=int, default=256, help='SetTransformer dimension D')
    parser.add_argument('--st_inducing_points', type=int, default=12, help='SetTransformer inducing points count m')
    parser.add_argument('--st_heads', type=int, default=4, help='SetTransformer attention heads count h')
    parser.add_argument('--st_k', type=int, default=4, help='SetTransformer seed vectors count k')
    parser.add_argument('--encoding_dim', type=int, default=512, help='Encoding dimension')
    parser.add_argument('--with_cpi', action='store_true', default=True, help='Enable CPI prediction head')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Computing device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--margin', type=float, default=0.45, help='Triplet loss margin')
    parser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'simclr'], 
                        help='Loss function type: triplet or simclr')
    parser.add_argument('--temperature', type=float, default=0.1, 
                        help='Temperature parameter for SimCLR loss')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./cpi_finetune_checkpoints', help='Output directory')
    parser.add_argument('--save_every', type=int, default=5, help='Save every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Checkpoint path to resume training from')
    
    # Pretrained model parameters
    parser.add_argument('--pretrained_model', type=str, default='', help='Pretrained model path')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== CPI Model Fine-tuning Started ===")
    print(f"Data directory: {args.data_dir}")
    print(f"File list: {args.file_list}")
    print(f"Output directory: {args.output_dir}")
    print(f"Computing device: {args.device}")
    
    # Read file list
    bbv_files = []
    map_files = []
    json_files = []
    
    with open(args.file_list, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 3:
                    bbv_file = os.path.join(args.data_dir, parts[0].strip())
                    map_file = os.path.join(args.data_dir, parts[1].strip())
                    json_file = os.path.join(args.data_dir, parts[2].strip())
                    
                    bbv_files.append(bbv_file)
                    map_files.append(map_file)
                    json_files.append(json_file)
    
    print(f"Loaded {len(bbv_files)} data file combinations")
    
    # Create dataset
    print("\n=== Create Dataset ===")
    dataset = CPIDataset(bbv_files, map_files, json_files)
    
    # Split training and validation sets
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 4, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    print("\n=== Create Model ===")
    device = torch.device(args.device)
    model = SetTransformerModel(args)
    model = model.to(device)
    
    # Load pretrained model (if provided)
    if args.pretrained_model:
        print(f"Load pretrained model: {args.pretrained_model}")
        ckpt = torch.load(args.pretrained_model, map_location=device)
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt
        
        # Only load compatible weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in sd.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully loaded {len(pretrained_dict)} pretrained weights")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume training (if checkpoint provided)
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # Training loop
    print("\n=== Start Training ===")
    best_val_loss = float('inf')
    patience = 6  # early stopping patience
    patience_counter = 0  # Record consecutive non-improvement times
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_mae, val_rmse = validate_epoch(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
            best_model_path = os.path.join(args.output_dir, 'best_cpi_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"Save best model, validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"Validation loss no improvement, patience: {patience_counter}/{patience}")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"\n=== Early Stopping ===")
                print(f"Validation loss no improvement for {patience} consecutive times, early termination")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        
        # Regularly save checkpoints
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    print("\n=== Training Completed ===")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved at: {os.path.join(args.output_dir, 'best_cpi_model.pt')}")


if __name__ == "__main__":
    main()