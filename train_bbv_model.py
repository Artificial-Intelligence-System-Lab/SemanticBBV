import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import argparse
import time
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
import random
from asm_bbv_dataset import TripleLossDataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Set-Transformer'))
from set_transformer_model import SetTransformerWithWeights


# Set random seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set CUDA deterministic options
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global random seed set to: {seed}")

# Import RWKV7 model and custom dataset
from rwkv7_cuda import RWKV


def split_dataset(dataset, test_ratio=0.3, seed=42):
    """
    Split dataset into training and test sets by ratio
    
    Args:
        dataset: Dataset to split
        test_ratio: Test set ratio, default 0.3 (30%)
        seed: Random seed
    
    Returns:
        train_dataset, test_dataset: Split training and test sets
    """
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # Calculate dataset size
    dataset_size = len(dataset)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    
    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {train_size} ({(1-test_ratio)*100:.1f}%)")
    print(f"Test set size: {test_size} ({test_ratio*100:.1f}%)")
    
    # Generate random indices
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    
    # Split indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset


class TripleLoss(nn.Module):
    """Triple Loss function"""
    def __init__(self, margin=1.0, distance_metric="cosine", loss_type="triplet", temperature=0.1):
        super(TripleLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.loss_type = loss_type  # "triplet" or "simclr"
        self.temperature = temperature  # temperature parameter
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)
        assert distance_metric in ["euclidean", "cosine"], "Distance metric must be 'euclidean' or 'cosine'"
        assert loss_type in ["triplet", "simclr"], "Loss type must be 'triplet' or 'simclr'"
    
    def _get_distance(self, x1, x2):
        """Calculate distance between two sets of vectors"""
        if self.distance_metric == "euclidean":
            return F.pairwise_distance(x1, x2, p=2)
        elif self.distance_metric == "cosine":
            # Ensure input vectors are normalized
            x1_normalized = F.normalize(x1, p=2, dim=1)
            x2_normalized = F.normalize(x2, p=2, dim=1)
            # Convert cosine similarity to distance: 1 - cos_sim
            return 1 - F.cosine_similarity(x1_normalized, x2_normalized)
    
    def forward(self, anchor, positive, negative=None):
        if self.loss_type == "triplet":
            pos_dist = self._get_distance(anchor, positive)
            neg_dist = self._get_distance(anchor, negative)
            # Modified: Use F.relu to implement bounded Triplet Loss
            loss = F.relu(pos_dist - neg_dist + self.margin).mean()
            return loss, pos_dist, neg_dist
        
        elif self.loss_type == "simclr":
            # Ensure input vectors are normalized
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            
            batch_size = anchor.size(0)
            device = anchor.device
            
            # Concatenate anchor and positive to form feature matrix
            features = torch.cat([anchor, positive], dim=0)  # [2*batch_size, dim]
            
            # Calculate complete similarity matrix
            sim_matrix = torch.mm(features, features.t()) / self.temperature  # [2*batch_size, 2*batch_size]
            
            # Create labels: for each anchor, its positive is the corresponding sample after batch_size
            # For each positive, its anchor is the corresponding sample before batch_size
            sim_labels = torch.arange(2 * batch_size, device=device)
            sim_labels = (sim_labels + batch_size) % (2 * batch_size)
            
            # Exclude self-similarity (diagonal)
            mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            
            # Calculate bidirectional cross-entropy loss: A→P and P→A
            loss = F.cross_entropy(sim_matrix, sim_labels)
            
            # To maintain interface consistency, we still return positive and negative sample distances
            # But in this case, we only calculate the corresponding positive sample distance
            pos_dist = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                pos_dist[i] = 1 - F.cosine_similarity(anchor[i:i+1], positive[i:i+1])
            
            # Since there are no explicit negative samples, we return a dummy value
            neg_dist = torch.ones_like(pos_dist)
            
            return loss, pos_dist, neg_dist


class RWKV7Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_args = type('', (), {})()
        model_args.n_embd = args.n_embd
        model_args.n_layer = args.n_layer
        model_args.head_size_a = args.head_size
        self.with_cpi = getattr(args, 'with_cpi', False)
        # Modified: Add CPI loss weight attributes
        self.cpi_regression_weight = getattr(args, 'cpi_regression_weight', 0.0)
        self.cpi_consistency_weight = getattr(args, 'cpi_consistency_weight', 0.0)

        # Projection layer to project 128-dim vectors to model dimension
        self.embedding_projection_layer = nn.Linear(128, args.n_embd)
        
        # Weight processing layer to convert scalar weights to vector influence
        self.weight_projection = nn.Linear(1, args.n_embd)
        
        self.model = RWKV(model_args)
        
        # Initialize attention query vector
        self.attention_query = nn.Sequential(
            nn.Linear(args.n_embd, 256),  # Reduce dimension to decrease parameters
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self.pool_ln = nn.LayerNorm(args.n_embd)

        self.encoding_head = nn.Sequential(
            nn.Linear(args.n_embd, args.encoding_dim * 2),
            nn.GELU(),
            nn.Linear(args.encoding_dim * 2, args.encoding_dim),
            nn.Dropout(0.1),
        )
        
        # Add CPI prediction task head
        if self.with_cpi:
            self.cpi_head = nn.Sequential(
                nn.Linear(args.n_embd, 256),
                nn.GELU(),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )

        self._init_weights()
        self.triple_loss = TripleLoss(
            margin=args.margin, 
            distance_metric="cosine",
            loss_type=args.loss_type,
            temperature=args.temperature
        )
    
    def _init_weights(self):
        """Initialize encoding head and attention query weights"""
        # Initialize encoding head
        for module in self.encoding_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize attention query
        for module in self.attention_query:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize CPI prediction head
        if self.with_cpi:
            for module in self.cpi_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
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
        
        # Calculate attention scores - modified to use Sequential module
        attention_scores = self.attention_query(x)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask (set positions outside mask to a very small negative number)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Weighted sum
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return pooled
    
    
    def _process_sample(self, embs, weight, return_set_embedding=False):
        """Process a single sample
        
        Args:
            embs: Embedding vectors [batch_size, seq_len, 128]
            weight: Weights [batch_size, seq_len]
            return_set_embedding: Whether to return set embedding for CPI prediction
            
        Returns:
            encoding: Sample encoding [batch_size, encoding_dim]
            set_embedding: (optional) Set embedding [batch_size, input_dim]
        """
        # Check if bfloat16 is used and ensure input data type matches
        if self.embedding_projection_layer.weight.dtype == torch.bfloat16:
            embs = embs.to(torch.bfloat16)
            weight = weight.to(torch.bfloat16)
        
        # Project embedding vectors to model dimension
        projected = self.embedding_projection_layer(embs)  # [batch_size, seq_len, n_embd]
        
        # Process weights
        weight_expanded = weight.unsqueeze(-1)  # [batch_size, seq_len, 1]
        gate = torch.sigmoid(self.weight_projection(weight_expanded))  # [B,T,n_embd]
        weighted = projected * gate                                    # [B,T,n_embd]
        
        # Get hidden states through RWKV model
        hidden = self.model(weighted)  # [batch_size, seq_len, n_embd]
        
        # Calculate sequence lengths (positions with non-zero weights)
        seq_lengths = torch.sum(weight > 0, dim=1)  # [batch_size]
        
        # Use attention pooling to get representation of entire sequence
        pooled = self.pooling(hidden, seq_lengths)  # [batch_size, n_embd]
        pooled = self.pool_ln(pooled)
        # pooled = torch.tanh(pooled)
        
        # Encode the sample
        encoding = self.encoding_head(pooled)  # [batch_size, encoding_dim]
        encoding = F.normalize(encoding, p=2, dim=-1)
        
        if return_set_embedding:
            return encoding, pooled  # Modified here, return pooled as set_embedding
        return encoding
    
    def forward(self, anchors_embs, anchors_weight, positive_embs, positive_weight, negative_embs=None, negative_weight=None, anchor_cpi=None, positive_cpi=None, negative_cpi=None):
        """Forward propagation function that processes triplets (anchor, positive, negative) and calculates loss

        Args:
            anchors_embs: Anchor sample embeddings [batch_size, seq_len, 128]
            anchors_weight: Anchor sample weights [batch_size, seq_len]
            positive_embs: Positive sample embeddings [batch_size, seq_len, 128]
            positive_weight: Positive sample weights [batch_size, seq_len]
            negative_embs: Negative sample embeddings [batch_size, seq_len, 128], can be None for simclr
            negative_weight: Negative sample weights [batch_size, seq_len], can be None for simclr
            anchor_cpi: Anchor sample CPI values [batch_size]
            positive_cpi: Positive sample CPI values [batch_size]
            negative_cpi: Negative sample CPI values [batch_size]

        Returns:
            (loss, triplet_loss, cpi_loss, pos_dist, neg_dist): Total loss, triplet loss, CPI loss, positive distance, negative distance
            encodings: Dictionary containing anchor, positive, negative encodings
        """
        # Process anchor and positive samples
        anchor_encoding, anchor_set_embedding = self._process_sample(anchors_embs, anchors_weight, return_set_embedding=True)
        positive_encoding, positive_set_embedding = self._process_sample(positive_embs, positive_weight, return_set_embedding=True)
        
        # ---- LOSS CALCULATION ----
        # 1. Triplet/SimCLR Loss
        negative_encoding = None
        negative_set_embedding = None
        if self.triple_loss.loss_type == "triplet" and negative_embs is not None:
            negative_encoding, negative_set_embedding = self._process_sample(negative_embs, negative_weight, return_set_embedding=True)
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding, negative_encoding)
        else:
            # For SimCLR, negative is None
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding)

        # 2. CPI-related Losses
        cpi_regression_loss = torch.tensor(0.0, device=anchor_encoding.device)
        cpi_consistency_loss = torch.tensor(0.0, device=anchor_encoding.device)

        if self.with_cpi and anchor_cpi is not None:
            # CPI Regression Loss (Huber)
            anchor_cpi_pred = self.cpi_head(anchor_set_embedding).squeeze(-1)
            anchor_cpi = anchor_cpi.to(anchor_cpi_pred.device)
            reg_loss = F.huber_loss(anchor_cpi_pred, anchor_cpi, reduction='mean', delta=1.0)
            num_cpi_samples = 1

            if positive_cpi is not None:
                positive_cpi_pred = self.cpi_head(positive_set_embedding).squeeze(-1)
                positive_cpi = positive_cpi.to(positive_cpi_pred.device)
                reg_loss += F.huber_loss(positive_cpi_pred, positive_cpi, reduction='mean', delta=1.0)
                num_cpi_samples += 1
            
            if negative_cpi is not None and negative_set_embedding is not None:
                negative_cpi_pred = self.cpi_head(negative_set_embedding).squeeze(-1)
                negative_cpi = negative_cpi.to(negative_cpi_pred.device)
                reg_loss += F.huber_loss(negative_cpi_pred, negative_cpi, reduction='mean', delta=1.0)
                num_cpi_samples += 1

            cpi_regression_loss = reg_loss / num_cpi_samples

            # CPI Consistency Loss: Penalize small distance for pairs with large CPI difference
            if positive_cpi is not None:
                cpi_diff = torch.abs(anchor_cpi - positive_cpi)
                cpi_consistency_loss = (pos_dist * cpi_diff).mean()

        # 3. Combine Losses
        cpi_loss = (self.cpi_regression_weight * cpi_regression_loss) + \
                   (self.cpi_consistency_weight * cpi_consistency_loss)
        loss = triplet_loss + cpi_loss

        # ---- END LOSS CALCULATION ----
    
        # Return loss and encodings
        encodings = {
            'anchor': anchor_encoding,
            'positive': positive_encoding
        }
        
        if negative_encoding is not None:
            encodings['negative'] = negative_encoding
    
        return (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings

class SetTransformerModel(nn.Module):
    """BBV model using SetTransformer"""
    
    def __init__(self, args):
        """Initialize SetTransformer model
        
        Args:
            args: Namespace containing model parameters
        """
        super().__init__()
        
        # Model configuration
        self.feature_dim = 128  # BBV feature dimension
        self.encoding_dim = args.encoding_dim
        self.with_cpi = getattr(args, 'with_cpi', False)
        # Modification: Add CPI loss weight attributes
        self.cpi_regression_weight = getattr(args, 'cpi_regression_weight', 0.0)
        self.cpi_consistency_weight = getattr(args, 'cpi_consistency_weight', 0.0)
        
        # Create SetTransformer model
        # D=128, m=16, h=4, k=4 (k*D=512)
        self.set_transformer = SetTransformerWithWeights(
            feature_dim=self.feature_dim,
            D=args.st_dim,
            m=args.st_inducing_points,
            h=args.st_heads,
            k=args.st_k
        )
        
        # Encoding head to map SetTransformer output to desired encoding dimension
        input_dim = args.st_dim * args.st_k  # Automatically calculate input dimension
        self.encoding_head = nn.Sequential(
            nn.Linear(input_dim, self.encoding_dim),
            nn.LayerNorm(self.encoding_dim)
        )
        
        # Add CPI prediction task head
        if self.with_cpi:
            hidden_dim1 = 256  # First hidden layer
            hidden_dim2 = 64   # Bottleneck layer

            self.cpi_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim1),
                nn.GELU(),
                nn.Dropout(0.1),  # Adding Dropout between deeper layers is a good regularization technique
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim2, 1) # Output directly from bottleneck layer
            )
        
        # Triplet loss
        self.triple_loss = TripleLoss(
            margin=args.margin,
            loss_type=args.loss_type,
            temperature=args.temperature
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Encoding head uses normal distribution initialization
        for module in self.encoding_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize CPI prediction head
        if self.with_cpi:
            for module in self.cpi_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def _process_sample(self, embs, weight, return_set_embedding=False):
        """Process a single sample
        
        Args:
            embs: embeddings [batch_size, seq_len, 128]
            weight: weights [batch_size, seq_len]
            return_set_embedding: whether to return set embedding for CPI prediction
            
        Returns:
            encoding: sample encoding [batch_size, encoding_dim]
            set_embedding: (optional) set embedding [batch_size, input_dim]
        """
        # Ensure weight is [batch_size, seq_len, 1] shape
        weight_expanded = weight.unsqueeze(-1)
        
        # Process sample using SetTransformer
        set_embedding = self.set_transformer(embs, weight_expanded)
        
        # Get final encoding through encoding head
        encoding = self.encoding_head(set_embedding)
        
        # L2 normalization
        encoding = F.normalize(encoding, p=2, dim=-1)
        
        if return_set_embedding:
            return encoding, set_embedding
        return encoding
    
    def forward(self, anchors_embs, anchors_weight, positive_embs, positive_weight, negative_embs=None, negative_weight=None, anchor_cpi=None, positive_cpi=None, negative_cpi=None):
        """Forward propagation function that processes triplets (anchor, positive, negative) and calculates loss

        Args:
            anchors_embs: Anchor sample embeddings [batch_size, seq_len, 128]
            anchors_weight: Anchor sample weights [batch_size, seq_len]
            positive_embs: Positive sample embeddings [batch_size, seq_len, 128]
            positive_weight: Positive sample weights [batch_size, seq_len]
            negative_embs: Negative sample embeddings [batch_size, seq_len, 128], can be None for simclr
            negative_weight: Negative sample weights [batch_size, seq_len], can be None for simclr
            anchor_cpi: Anchor sample CPI values [batch_size]
            positive_cpi: Positive sample CPI values [batch_size]
            negative_cpi: Negative sample CPI values [batch_size]

        Returns:
            (loss, triplet_loss, cpi_loss, pos_dist, neg_dist): Total loss, triplet loss, CPI loss, positive distance, negative distance
            encodings: Dictionary containing anchor, positive, negative encodings
        """
        # Process anchor and positive samples
        anchor_encoding, anchor_set_embedding = self._process_sample(anchors_embs, anchors_weight, return_set_embedding=True)
        positive_encoding, positive_set_embedding = self._process_sample(positive_embs, positive_weight, return_set_embedding=True)
    
        # ---- LOSS CALCULATION ----
        # 1. Triplet/SimCLR Loss
        negative_encoding = None
        negative_set_embedding = None
        if self.triple_loss.loss_type == "triplet" and negative_embs is not None:
            negative_encoding, negative_set_embedding = self._process_sample(negative_embs, negative_weight, return_set_embedding=True)
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding, negative_encoding)
        else:
            # For SimCLR, negative is None
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding)

        # 2. CPI-related Losses
        cpi_regression_loss = torch.tensor(0.0, device=anchor_encoding.device)
        cpi_consistency_loss = torch.tensor(0.0, device=anchor_encoding.device)

        if self.with_cpi and anchor_cpi is not None:
            # CPI Regression Loss (Huber)
            anchor_cpi_pred = self.cpi_head(anchor_set_embedding).squeeze(-1)
            anchor_cpi = anchor_cpi.to(anchor_cpi_pred.device)
            reg_loss = F.huber_loss(anchor_cpi_pred, anchor_cpi, reduction='mean', delta=1.0)
            num_cpi_samples = 1

            if positive_cpi is not None:
                positive_cpi_pred = self.cpi_head(positive_set_embedding).squeeze(-1)
                positive_cpi = positive_cpi.to(positive_cpi_pred.device)
                reg_loss += F.huber_loss(positive_cpi_pred, positive_cpi, reduction='mean', delta=1.0)
                num_cpi_samples += 1
            
            if negative_cpi is not None and negative_set_embedding is not None:
                negative_cpi_pred = self.cpi_head(negative_set_embedding).squeeze(-1)
                negative_cpi = negative_cpi.to(negative_cpi_pred.device)
                reg_loss += F.huber_loss(negative_cpi_pred, negative_cpi, reduction='mean', delta=1.0)
                num_cpi_samples += 1

            cpi_regression_loss = reg_loss / num_cpi_samples

            # CPI Consistency Loss: Penalize small distance for pairs with large CPI difference
            if positive_cpi is not None:
                cpi_diff = torch.abs(anchor_cpi - positive_cpi)
                cpi_consistency_loss = (pos_dist * cpi_diff).mean()

        # 3. Combine Losses
        cpi_loss = (self.cpi_regression_weight * cpi_regression_loss) + \
                   (self.cpi_consistency_weight * cpi_consistency_loss)
        loss = triplet_loss + cpi_loss

        # ---- END LOSS CALCULATION ----

        # Return loss and encodings
        encodings = {
            'anchor': anchor_encoding,
            'positive': positive_encoding
        }
    
        if negative_encoding is not None:
            encodings['negative'] = negative_encoding

        return (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings


# Add EarlyStopping class
class EarlyStopping:
    """Early stopping to stop training when a monitored metric has stopped improving."""
    def __init__(self, patience=5, monitor='val_loss', mode='min'):
        """
        Args:
            patience: Number of evaluation rounds to tolerate
            monitor: Monitored metric
            mode: 'max' means higher metric is better, 'min' means lower metric is better
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, current_metrics, model):
        current_score = current_metrics[self.monitor]
        """Check if should early stop"""
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        # Check if there is improvement
        if self.mode == 'max':
            improved = current_score > self.best_score
        else:
            improved = current_score < self.best_score
        
        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            print(f"No improvement for {self.counter}/{self.patience} evaluations")
            return self.counter >= self.patience
    
    def load_best_model(self, model):
        """Load best model state"""
        if self.best_model_state is not None:
            # Move state back to correct device
            device = next(model.parameters()).device
            state_dict = {k: v.to(device) for k, v in self.best_model_state.items()}
            model.load_state_dict(state_dict)
            print(f"Loaded best model with {self.monitor}: {self.best_score:.4f}")

def pad_to_max_length_collate_fn(batch):
    """
    Custom collate_fn function that performs internal padding alignment for anchor, positive, negative respectively
    
    Args:
        batch: Batch data from DataLoader, each element is (anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight)
              or (anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight, anchor_cpi, positive_cpi, negative_cpi)
    
    Returns:
        tuple: Aligned (anchor_vectors, anchor_weights, positive_vectors, positive_weights, negative_vectors, negative_weights)
               or (anchor_vectors, anchor_weights, positive_vectors, positive_weights, negative_vectors, negative_weights, anchor_cpis, positive_cpis, negative_cpis)
    """
    # Check if CPI values are included
    has_cpi = len(batch[0]) > 6
    
    # Unpack batch data
    if has_cpi:
        anchor_vectors, anchor_weights, positive_vectors, positive_weights, negative_vectors, negative_weights, anchor_cpis, positive_cpis, negative_cpis = zip(*batch)
    else:
        anchor_vectors, anchor_weights, positive_vectors, positive_weights, negative_vectors, negative_weights = zip(*batch)
    
    def pad_vectors_and_weights(vectors, weights):
        """Pad vectors and weights of the same type"""
        # Find the maximum length in current type
        max_length = max(len(vec) for vec in vectors)
        # Ensure max_length is a multiple of 16
        max_length = (max_length + 15) // 16 * 16
        
        padded_vectors = []
        padded_weights = []
        
        for vec, weight in zip(vectors, weights):
            current_length = len(vec)
            if current_length < max_length:
                # Zero-pad vectors
                pad_length = max_length - current_length
                padded_vec = torch.cat([vec, torch.zeros(pad_length, vec.size(-1), dtype=vec.dtype, device=vec.device)], dim=0)
                # Zero-pad weights
                padded_weight = torch.cat([weight, torch.zeros(pad_length, dtype=weight.dtype, device=weight.device)], dim=0)
            else:
                padded_vec = vec
                padded_weight = weight
            
            padded_vectors.append(padded_vec)
            padded_weights.append(padded_weight)
        
        return torch.stack(padded_vectors), torch.stack(padded_weights)
    
    # Pad anchor, positive, negative respectively
    anchor_vectors_padded, anchor_weights_padded = pad_vectors_and_weights(anchor_vectors, anchor_weights)
    positive_vectors_padded, positive_weights_padded = pad_vectors_and_weights(positive_vectors, positive_weights)
    negative_vectors_padded, negative_weights_padded = pad_vectors_and_weights(negative_vectors, negative_weights)
    
    if has_cpi:
        # Convert CPI values to tensor
        anchor_cpis = torch.tensor(anchor_cpis, dtype=torch.float32)
        positive_cpis = torch.tensor(positive_cpis, dtype=torch.float32)
        negative_cpis = torch.tensor(negative_cpis, dtype=torch.float32)
        
        return (
            anchor_vectors_padded, anchor_weights_padded,
            positive_vectors_padded, positive_weights_padded,
            negative_vectors_padded, negative_weights_padded,
            anchor_cpis, positive_cpis, negative_cpis
        )
    else:
        return (
            anchor_vectors_padded, anchor_weights_padded,
            positive_vectors_padded, positive_weights_padded,
            negative_vectors_padded, negative_weights_padded
        )

def evaluate_model(model, eval_loader, device, max_eval_batches=None):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_triplet_loss = 0
    total_cpi_loss = 0
    total_pos_distance = 0
    total_neg_distance = 0
    total_samples = 0
    margin_violations = 0
    
    with torch.no_grad():
        eval_batches = 0
        # Determine number of iterations for tqdm progress bar
        total_batches = len(eval_loader) if max_eval_batches is None else min(max_eval_batches, len(eval_loader))
        for batch in tqdm(eval_loader, total=total_batches, desc="Evaluating"):
            if max_eval_batches and eval_batches >= max_eval_batches:
                break
            
            # Check if CPI values are included
            has_cpi = len(batch) > 6
            
            if has_cpi:
                anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight, anchor_cpi, positive_cpi, negative_cpi = batch
                
                # Move data to specified device
                anchor_vector = anchor_vector.to(device)
                anchor_weight = anchor_weight.to(device)
                positive_vector = positive_vector.to(device)
                positive_weight = positive_weight.to(device)
                negative_vector = negative_vector.to(device)
                negative_weight = negative_weight.to(device)
                anchor_cpi = anchor_cpi.to(device)
                positive_cpi = positive_cpi.to(device)
                negative_cpi = negative_cpi.to(device)
                
                # Forward propagation
                (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings = model(
                    anchor_vector, anchor_weight,
                    positive_vector, positive_weight,
                    negative_vector, negative_weight,
                    anchor_cpi, positive_cpi, negative_cpi
                )
                
                # Accumulate CPI loss
                total_cpi_loss += cpi_loss.item() * anchor_vector.size(0)
            else:
                anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight = batch
                
                # Move data to specified device
                anchor_vector = anchor_vector.to(device)
                anchor_weight = anchor_weight.to(device)
                positive_vector = positive_vector.to(device)
                positive_weight = positive_weight.to(device)
                negative_vector = negative_vector.to(device)
                negative_weight = negative_weight.to(device)
                
                # Forward propagation
                (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings = model(
                    anchor_vector, anchor_weight,
                    positive_vector, positive_weight,
                    negative_vector, negative_weight
                )
            
            batch_size = anchor_vector.size(0)
            total_loss += loss.item() * batch_size
            total_triplet_loss += triplet_loss.item() * batch_size
            total_pos_distance += pos_dist.sum().item()
            total_neg_distance += neg_dist.sum().item()
            total_samples += batch_size
            
            # Calculate margin violation
            margin_violations += (pos_dist >= neg_dist).sum().item()
            eval_batches += 1
    
    avg_loss = total_loss / total_samples
    avg_triplet_loss = total_triplet_loss / total_samples
    avg_pos_distance = total_pos_distance / total_samples
    avg_neg_distance = total_neg_distance / total_samples
    margin_violation_rate = margin_violations / total_samples
    distance_separation = avg_neg_distance - avg_pos_distance
    
    result = {
        'avg_loss': avg_loss,
        'avg_triplet_loss': avg_triplet_loss,
        'avg_pos_distance': avg_pos_distance,
        'avg_neg_distance': avg_neg_distance,
        'margin_violation_rate': margin_violation_rate,
        'distance_separation': distance_separation
    }
    
    # If CPI data exists, add avg_cpi_loss
    if has_cpi:
        avg_cpi_loss = total_cpi_loss / total_samples
        result['avg_cpi_loss'] = avg_cpi_loss
    
    return result

def train_model(
    model,
    train_dataset,
    eval_dataset=None,
    batch_size=32,
    eval_batch_size=64,
    num_epochs=10,
    learning_rate=1e-4,
    eval_steps=500,
    save_steps=1000,
    output_dir="./checkpoints",
    num_workers=4,
    device="cuda",
    gradient_accumulation_steps=1  # Add gradient accumulation steps parameter
):
    """Train BBV model"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize evaluation metrics history
    metrics_history = {
        'steps': [],
        'avg_loss': [],
        'avg_triplet_loss': [],  # Add this line
        'avg_pos_distance': [],
        'avg_neg_distance': [],
        'margin_violation_rate': [],
        'distance_separation': []
    }
    
    # If model uses CPI prediction, also add corresponding keys
    if getattr(model, 'with_cpi', False):
        metrics_history['avg_cpi_loss'] = []
    
    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=pad_to_max_length_collate_fn
    )
    
    if eval_dataset:
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=pad_to_max_length_collate_fn
        )
    
    # Prepare optimizer parameter groups - all parameters use same learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),  # Modified β2 to 0.98, this is the empirical value recommended by RWKV author
        weight_decay=0.01
    )
    
    # Calculate total training steps and learning rate scheduler
    # Consider effective steps after gradient accumulation
    total_physical_steps = len(train_loader) * num_epochs
    total_effective_steps = total_physical_steps // gradient_accumulation_steps
    warmup_steps = min(100, total_effective_steps // 10)  # 10% of steps as warmup
    
    def get_warmup_decay_lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, float(total_effective_steps - current_step) / float(max(1, total_effective_steps - warmup_steps)))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_decay_lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=4,
        monitor='avg_loss', 
        mode='min'
    )
    
    global_step = 0  # Effective steps (considering gradient accumulation)
    evaluation_count = 0
    
    print("Starting training...")
    print(f"Evaluate every {eval_steps} steps")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Evaluation batch size: {eval_batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total physical steps: {total_physical_steps}, total effective steps: {total_effective_steps}, warmup steps: {warmup_steps}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        
        # Add tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            # Check if CPI values are included
            has_cpi = len(batch) > 6
            
            if has_cpi:
                anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight, anchor_cpi, positive_cpi, negative_cpi = batch
                
                # Move data to specified device
                anchor_vector = anchor_vector.to(device)
                anchor_weight = anchor_weight.to(device)
                positive_vector = positive_vector.to(device)
                positive_weight = positive_weight.to(device)
                negative_vector = negative_vector.to(device)
                negative_weight = negative_weight.to(device)
                anchor_cpi = anchor_cpi.to(device)
                positive_cpi = positive_cpi.to(device)
                negative_cpi = negative_cpi.to(device)
                
                # Forward propagation
                (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings = model(
                    anchor_vector, anchor_weight,
                    positive_vector, positive_weight,
                    negative_vector, negative_weight,
                    anchor_cpi, positive_cpi, negative_cpi
                )
            else:
                anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight = batch
                
                # Move data to specified device
                anchor_vector = anchor_vector.to(device)
                anchor_weight = anchor_weight.to(device)
                positive_vector = positive_vector.to(device)
                positive_weight = positive_weight.to(device)
                negative_vector = negative_vector.to(device)
                negative_weight = negative_weight.to(device)
                
                # Forward propagation
                (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings = model(
                    anchor_vector, anchor_weight,
                    positive_vector, positive_weight,
                    negative_vector, negative_weight
                )
            
            # Scale loss to accommodate gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backpropagation
            loss.backward()
            
            # Update statistics - use original loss values (unscaled) for recording
            batch_size_actual = anchor_vector.size(0)
            epoch_loss += (loss.item() * gradient_accumulation_steps) * batch_size_actual
            epoch_samples += batch_size_actual
            
            # Only update parameters after accumulation is complete
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update global step
                global_step += 1
                
                # Evaluate model
                if global_step % eval_steps == 0 and eval_dataset:
                    print(f"\nStep {global_step}: Starting evaluation...")
                    eval_metrics = evaluate_model(model, eval_loader, device, max_eval_batches=50)
                    
                    print(f"Triplet task - Loss: {eval_metrics['avg_triplet_loss']:.4f}, "
                          f"Positive distance: {eval_metrics['avg_pos_distance']:.4f}, "
                          f"Negative distance: {eval_metrics['avg_neg_distance']:.4f}, "
                          f"Violation rate: {eval_metrics['margin_violation_rate']:.4f}")
                    
                    if 'avg_cpi_loss' in eval_metrics:
                        print(f"CPI related task - Weighted loss: {eval_metrics['avg_cpi_loss']:.4f}")
                    
                    print(f"Total average loss: {eval_metrics['avg_loss']:.4f}")
                    sys.stdout.flush()  # Force flush output

                    # Record metrics
                    metrics_history['steps'].append(global_step)
                    for key, value in eval_metrics.items():
                        if key in metrics_history:
                           metrics_history[key].append(value)
                    
                    # Early stopping check
                    if early_stopping(eval_metrics, model):
                        print(f"Early stopping triggered at step {global_step}")
                        early_stopping.load_best_model(model)
                        return metrics_history
                    
                    evaluation_count += 1
                    model.train()  # Switch back to training mode
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                        'metrics_history': metrics_history
                    }, os.path.join(checkpoint_path, "pytorch_model.bin"))
                    print(f"\nSave checkpoint to {checkpoint_path}")
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'triplet': f'{triplet_loss.item():.4f}',
                'cpi': f'{cpi_loss.item():.4f}' if model.with_cpi else 'N/A',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Statistics after each epoch
        avg_epoch_loss = epoch_loss / epoch_samples
        print(f"\nEpoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")
    
    # Training finished, load best model
    early_stopping.load_best_model(model)
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_checkpoint_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'metrics_history': metrics_history
    }, os.path.join(final_checkpoint_path, "pytorch_model.bin"))
    
    print(f"Training completed! Final model saved to {final_checkpoint_path}")
    return metrics_history

if __name__ == "__main__":
    # Set random seed
    set_seed(42)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BBV model')
    parser.add_argument('--data_path', type=str, required=True, help='Training data path')
    parser.add_argument('--eval_data_path', type=str, help='Evaluation data path')
    parser.add_argument('--test_split_ratio', type=float, default=0.3, help='Test set ratio split from training data when evaluation data is not specified (default 0.3)')
    parser.add_argument('--output_dir', type=str, default='./bbv_checkpoints', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation steps interval')
    parser.add_argument('--save_steps', type=int, default=100, help='Save steps interval')
    parser.add_argument('--num_workers', type=int, default=3, help='Number of data loader worker processes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=12, help='Gradient accumulation steps')
    
    # Model selection parameters
    parser.add_argument('--model_type', type=str, default='set_transformer', choices=['rwkv7', 'set_transformer'], 
                        help='Model type: rwkv7 or set_transformer')
    
    # RWKV7 model parameters
    parser.add_argument('--n_embd', type=int, default=256, help='RWKV embedding dimension')
    parser.add_argument('--n_layer', type=int, default=6, help='RWKV number of layers')
    parser.add_argument('--head_size', type=int, default=64, help='RWKV attention head size')
    
    # SetTransformer model parameters
    parser.add_argument('--st_dim', type=int, default=256, help='SetTransformer dimension D')
    parser.add_argument('--st_inducing_points', type=int, default=12, help='SetTransformer number of inducing points m')
    parser.add_argument('--st_heads', type=int, default=4, help='SetTransformer number of attention heads h')
    parser.add_argument('--st_k', type=int, default=4, help='SetTransformer number of seed vectors k')
    
    # General model parameters
    parser.add_argument('--encoding_dim', type=int, default=512, help='Encoding dimension')
    parser.add_argument('--margin', type=float, default=0.45, help='Triplet loss margin')
    parser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'simclr'], 
                        help='Loss function type: triplet or simclr')
    parser.add_argument('--temperature', type=float, default=0.1, 
                        help='Temperature parameter for SimCLR loss')
    # Modification: Add CPI related parameters
    parser.add_argument("--with_cpi", action='store_true', help="Enable CPI regression and consistency pre-training tasks (default: False)")
    parser.add_argument('--cpi_regression_weight', type=float, default=0.01, help='Weight of CPI regression loss in total loss')
    parser.add_argument('--cpi_consistency_weight', type=float, default=0.01, help='Weight of CPI consistency loss in total loss')

    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Create model based on selected model type
    if args.model_type == 'rwkv7':
        print("Using RWKV7 model")
        model = RWKV7Model(args)
        
        # Set mixed precision, parameters except attention and encoding_head are bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("Using BF16 mixed precision, attention and encoding_head maintain float32 precision")
            model.embedding_projection_layer = model.embedding_projection_layer.to(torch.bfloat16)
            model.weight_projection = model.weight_projection.to(torch.bfloat16)
            model.model = model.model.to(torch.bfloat16)
            # attention_query and encoding_head maintain float32 precision
    else:
        print("Using SetTransformer model")
        model = SetTransformerModel(args)
    
    model = model.to(device)
    
    # Load dataset
    print(f"Loading training data: {args.data_path}")
    full_dataset = TripleLossDataset(args.data_path, with_cpi=args.with_cpi)
    datasets_to_combine = [full_dataset]
    
    # Process evaluation dataset
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        print(f"Loading evaluation data: {args.eval_data_path}")
        eval_dataset = TripleLossDataset(args.eval_data_path, with_cpi=args.with_cpi)
        datasets_to_combine.append(eval_dataset)
    
    # 3. Merge all loaded datasets
    if len(datasets_to_combine) > 1:
        print("Mixing training set and test set...")
        full_dataset = ConcatDataset(datasets_to_combine)
    
    # 4. Perform final random split on the merged complete dataset
    print(f"Will perform {1-args.test_split_ratio:.1%}/{args.test_split_ratio:.1%} random split on total {len(full_dataset)} samples...")
    train_dataset, eval_dataset = split_dataset(
        full_dataset, 
        test_ratio=args.test_split_ratio, 
        seed=42
    )
    
    # Start training
    metrics_history = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps  # Pass gradient accumulation steps
    )
    
    print("Training completed!")