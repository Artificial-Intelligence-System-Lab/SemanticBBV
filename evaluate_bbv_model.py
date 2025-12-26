import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
    """
    Set global random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set CUDA deterministic options
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global random seed set to: {seed}")

# Import RWKV7 model
from rwkv7_cuda import RWKV


class TripleLoss(nn.Module):
    """Triple Loss function"""
    def __init__(self, margin=1.0, distance_metric="cosine", loss_type="triplet", temperature=0.1):
        super(TripleLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.loss_type = loss_type  # "triplet" or "simclr"
        self.temperature = temperature  # Temperature parameter
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
            # loss = self.triplet(anchor, positive, negative)
            pos_dist = self._get_distance(anchor, positive)
            neg_dist = self._get_distance(anchor, negative)
            loss = F.softplus(pos_dist - neg_dist + self.margin, beta=10).mean()
            return loss.mean(), pos_dist, neg_dist
        
        elif self.loss_type == "simclr":
            # Ensure input vectors are normalized
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            
            batch_size = anchor.size(0)
            device = anchor.device
            
            # Concatenate anchor and positive to form feature matrix
            features = torch.cat([anchor, positive], dim=0)  # [2*batch_size, dim]
            
            # Calculate full similarity matrix
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
    """BBV model using RWKV7"""
    def __init__(self, args):
        super().__init__()
        model_args = type('', (), {})()
        model_args.n_embd = args.n_embd
        model_args.n_layer = args.n_layer
        model_args.head_size_a = args.head_size
        self.with_cpi = args.with_cpi
        
        # Projection layer, projecting 128-dimensional vectors to model dimension
        self.embedding_projection_layer = nn.Linear(128, args.n_embd)
        
        # Weight processing layer, converting scalar weights to vector effects
        self.weight_projection = nn.Linear(1, args.n_embd)
        
        self.model = RWKV(model_args)
        
        # Initialize attention query vectors
        self.attention_query = nn.Sequential(
            nn.Linear(args.n_embd, 256),  # Dimension reduction to reduce parameters
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
        
        # Weighted summation
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return pooled
    
    
    def _process_sample(self, embs, weight, return_set_embedding=False):
        """
        Process single sample (anchor, positive, or negative)
        
        Parameters:
            embs: sample embedding vectors [batch_size, seq_len, 128]
            weight: sample weights [batch_size, seq_len]
            return_set_embedding: whether to return pooled features for CPI prediction
            
        Returns:
            encoding: sample encoding vectors [batch_size, encoding_dim]
            pooled: (optional) pooled features [batch_size, n_embd]
        """
        # Check if bfloat16 is used and ensure input data types match
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
        
        # Calculate sequence length (positions with non-zero weights)
        seq_lengths = torch.sum(weight > 0, dim=1)  # [batch_size]
        
        # Use attention pooling to get representation of entire sequence
        pooled = self.pooling(hidden, seq_lengths)  # [batch_size, n_embd]
        pooled = self.pool_ln(pooled)
        
        # Encode the sample
        encoding = self.encoding_head(pooled)  # [batch_size, encoding_dim]
        encoding = F.normalize(encoding, p=2, dim=-1)
        
        if return_set_embedding:
            return encoding, pooled
        return encoding
    
    def forward(self, anchors_embs, anchors_weight, positive_embs, positive_weight, negative_embs=None, negative_weight=None, anchor_cpi=None, positive_cpi=None, negative_cpi=None):
        """
        Forward pass function handles triplets (anchor, positive, negative) and calculates loss
    
        Parameters:
            anchors_embs: anchor sample embedding vectors [batch_size, seq_len, 128]
            anchors_weight: anchor sample weights [batch_size, seq_len]
            positive_embs: positive sample embedding vectors [batch_size, seq_len, 128]
            positive_weight: positive sample weights [batch_size, seq_len]
            negative_embs: negative sample embedding vectors [batch_size, seq_len, 128], can be None for simclr
            negative_weight: negative sample weights [batch_size, seq_len], can be None for simclr
            anchor_cpi: CPI values for anchor samples [batch_size]
            positive_cpi: CPI values for positive samples [batch_size]
            negative_cpi: CPI values for negative samples [batch_size]
    
        Returns:
            loss: total loss value
            triplet_loss: triplet loss value
            cpi_loss: CPI prediction loss value
            pos_dist: distance between anchor and positive
            neg_dist: distance between anchor and negative
            encodings: dictionary containing encodings of anchor, positive, negative
        """
        # Process anchor and positive samples
        anchor_encoding, anchor_set_embedding = self._process_sample(anchors_embs, anchors_weight, return_set_embedding=True)
        positive_encoding, positive_set_embedding = self._process_sample(positive_embs, positive_weight, return_set_embedding=True)
        
        # Decide whether to process negative samples based on loss type
        negative_encoding = None
        negative_set_embedding = None
        if self.triple_loss.loss_type == "triplet" and negative_embs is not None:
            negative_encoding, negative_set_embedding = self._process_sample(negative_embs, negative_weight, return_set_embedding=True)
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding, negative_encoding)
        else:
            # Use SimCLR-style loss
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding)
        
        # CPI prediction loss
        cpi_loss = 0.0
        if self.with_cpi and anchor_cpi is not None:
            # Predict CPI values
            anchor_cpi_pred = self.cpi_head(anchor_set_embedding).squeeze(-1)
            
            # Calculate Huber loss
            anchor_cpi = anchor_cpi.to(anchor_cpi_pred.device)
            cpi_loss = F.huber_loss(anchor_cpi_pred, anchor_cpi, reduction='mean', delta=1.0)
            
            # If there are CPI values for positive and negative, also calculate their loss
            if positive_cpi is not None:
                positive_cpi_pred = self.cpi_head(positive_set_embedding).squeeze(-1)
                positive_cpi = positive_cpi.to(positive_cpi_pred.device)
                cpi_loss += F.huber_loss(positive_cpi_pred, positive_cpi, reduction='mean', delta=1.0)
            
            if negative_cpi is not None and negative_set_embedding is not None:
                negative_cpi_pred = self.cpi_head(negative_set_embedding).squeeze(-1)
                negative_cpi = negative_cpi.to(negative_cpi_pred.device)
                cpi_loss += F.huber_loss(negative_cpi_pred, negative_cpi, reduction='mean', delta=1.0)
            
            # Take average
            cpi_loss = cpi_loss / (1 + (positive_cpi is not None) + (negative_cpi is not None and negative_set_embedding is not None))
            
            # Combine losses
            loss = triplet_loss + cpi_loss
        else:
            loss = triplet_loss
            cpi_loss = 0.0
    
        # Return loss and encoding
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
        """
        Initialize SetTransformer model
        
        Parameters:
            args: namespace containing model parameters
        """
        super().__init__()
        
        # Model configuration
        self.feature_dim = 128  # BBV feature dimension
        self.encoding_dim = args.encoding_dim
        self.with_cpi = args.with_cpi
        
        # Create SetTransformer model
        # D=128, m=16, h=4, k=4 (k*D=512)
        self.set_transformer = SetTransformerWithWeights(
            feature_dim=self.feature_dim,
            D=args.st_dim,
            m=args.st_inducing_points,
            h=args.st_heads,
            k=args.st_k
        )
        
        # Encoding head, mapping SetTransformer output to desired encoding dimension
        input_dim = args.st_dim * args.st_k  # Auto-calculate input dimension
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
        """
        Process single sample
        
        Parameters:
            embs: embedding vectors [batch_size, seq_len, 128]
            weight: weights [batch_size, seq_len]
            return_set_embedding: whether to return set embedding for CPI prediction
            
        Returns:
            encoding: sample encoding [batch_size, encoding_dim]
            set_embedding: (optional) set embedding [batch_size, input_dim]
        """
        # Ensure weight is shape [batch_size, seq_len, 1]
        weight_expanded = weight.unsqueeze(-1)
        
        # Use SetTransformer to process samples
        set_embedding = self.set_transformer(embs, weight_expanded)
        
        # Get final encoding through encoding head
        encoding = self.encoding_head(set_embedding)
        
        # L2 normalization
        encoding = F.normalize(encoding, p=2, dim=-1)
        
        if return_set_embedding:
            return encoding, set_embedding
        return encoding
    
    def forward(self, anchors_embs, anchors_weight, positive_embs, positive_weight, negative_embs=None, negative_weight=None, anchor_cpi=None, positive_cpi=None, negative_cpi=None):
        """
        Forward pass function handles triplets (anchor, positive, negative) and calculates loss
    
        Parameters:
            anchors_embs: anchor sample embedding vectors [batch_size, seq_len, 128]
            anchors_weight: anchor sample weights [batch_size, seq_len]
            positive_embs: positive sample embedding vectors [batch_size, seq_len, 128]
            positive_weight: positive sample weights [batch_size, seq_len]
            negative_embs: negative sample embedding vectors [batch_size, seq_len, 128], can be None for simclr
            negative_weight: negative sample weights [batch_size, seq_len], can be None for simclr
            anchor_cpi: CPI values for anchor samples [batch_size]
            positive_cpi: CPI values for positive samples [batch_size]
            negative_cpi: CPI values for negative samples [batch_size]
    
        Returns:
            loss: total loss value
            triplet_loss: triplet loss value
            cpi_loss: CPI prediction loss value
            pos_dist: distance between anchor and positive
            neg_dist: distance between anchor and negative
            encodings: dictionary containing encodings of anchor, positive, negative
        """
        # Process anchor and positive samples
        anchor_encoding, anchor_set_embedding = self._process_sample(anchors_embs, anchors_weight, return_set_embedding=True)
        positive_encoding, positive_set_embedding = self._process_sample(positive_embs, positive_weight, return_set_embedding=True)
        
        # Decide whether to process negative samples based on loss type
        negative_encoding = None
        negative_set_embedding = None
        if self.triple_loss.loss_type == "triplet" and negative_embs is not None:
            negative_encoding, negative_set_embedding = self._process_sample(negative_embs, negative_weight, return_set_embedding=True)
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding, negative_encoding)
        else:
            # Use SimCLR-style loss
            triplet_loss, pos_dist, neg_dist = self.triple_loss(anchor_encoding, positive_encoding)
        
        # CPI prediction loss
        cpi_loss = 0.0
        if self.with_cpi and anchor_cpi is not None:
            # Predict CPI values
            anchor_cpi_pred = self.cpi_head(anchor_set_embedding).squeeze(-1)
            
            # Calculate Huber loss
            anchor_cpi = anchor_cpi.to(anchor_cpi_pred.device)
            cpi_loss = F.huber_loss(anchor_cpi_pred, anchor_cpi, reduction='mean', delta=1.0)
            
            # If there are CPI values for positive and negative, also calculate their loss
            if positive_cpi is not None:
                positive_cpi_pred = self.cpi_head(positive_set_embedding).squeeze(-1)
                positive_cpi = positive_cpi.to(positive_cpi_pred.device)
                cpi_loss += F.huber_loss(positive_cpi_pred, positive_cpi, reduction='mean', delta=1.0)
            
            if negative_cpi is not None and negative_set_embedding is not None:
                negative_cpi_pred = self.cpi_head(negative_set_embedding).squeeze(-1)
                negative_cpi = negative_cpi.to(negative_cpi_pred.device)
                cpi_loss += F.huber_loss(negative_cpi_pred, negative_cpi, reduction='mean', delta=1.0)
            
            # Take average
            cpi_loss = cpi_loss / (1 + (positive_cpi is not None) + (negative_cpi is not None and negative_set_embedding is not None))
            
            # Combine losses
            loss = triplet_loss + cpi_loss
        else:
            loss = triplet_loss
            cpi_loss = 0.0
    
        # Return loss and encoding
        encodings = {
            'anchor': anchor_encoding,
            'positive': positive_encoding
        }
        
        if negative_encoding is not None:
            encodings['negative'] = negative_encoding
    
        return (loss, triplet_loss, cpi_loss, pos_dist, neg_dist), encodings
    
    def predict_cpi(self, embs_list, weights_list):
        """
        Method specifically for CPI prediction
        
        Parameters:
            embs_list: list of embedding vectors, each element is a tensor of [seq_len, 128]
            weights_list: list of weights, each element is a tensor of [seq_len]
            
        Returns:
            torch.Tensor: predicted CPI values [batch_size]
        """
        if not self.with_cpi:
            raise ValueError("Model does not have CPI prediction enabled, please set with_cpi=True during initialization")
        
        self.eval()
        batch_size = len(embs_list)
        
        # Find maximum sequence length for padding
        max_len = max(emb.shape[0] for emb in embs_list)
        # Ensure length is a multiple of 16
        max_len = (max_len + 15) // 16 * 16
        
        # Prepare batch data
        batch_embs = torch.zeros(batch_size, max_len, self.feature_dim, 
                                device=embs_list[0].device, dtype=embs_list[0].dtype)
        batch_weights = torch.zeros(batch_size, max_len, 
                                   device=weights_list[0].device, dtype=weights_list[0].dtype)
        
        # Fill data
        for i, (emb, weight) in enumerate(zip(embs_list, weights_list)):
            seq_len = emb.shape[0]
            batch_embs[i, :seq_len] = emb
            batch_weights[i, :seq_len] = weight
        
        with torch.no_grad():
            # Get set embedding
            _, set_embedding = self._process_sample(batch_embs, batch_weights, return_set_embedding=True)
            
            # Predict CPI
            cpi_pred = self.cpi_head(set_embedding).squeeze(-1)
            
        return cpi_pred


def pad_to_max_length_collate_fn(batch):
    """
    Custom collate_fn for handling variable-length sequences in batch processing
    
    Parameters:
        batch: batch data from DataLoader, each element is (anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight)
              or (anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight, anchor_cpi, positive_cpi, negative_cpi)
    
    Returns:
        tuple: aligned (anchor_vectors, anchor_weights, positive_vectors, positive_weights, negative_vectors, negative_weights)
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
        # Find the maximum length in the current type
        max_length = max(len(vec) for vec in vectors)
        # Ensure max_length is a multiple of 16
        max_length = (max_length + 15) // 16 * 16
        
        padded_vectors = []
        padded_weights = []
        
        for vec, weight in zip(vectors, weights):
            current_length = len(vec)
            if current_length < max_length:
                # Pad vector with zeros
                pad_length = max_length - current_length
                padded_vec = torch.cat([vec, torch.zeros(pad_length, vec.size(-1), dtype=vec.dtype, device=vec.device)], dim=0)
                # Pad weight with zeros
                padded_weight = torch.cat([weight, torch.zeros(pad_length, dtype=weight.dtype, device=weight.device)], dim=0)
            else:
                padded_vec = vec
                padded_weight = weight
            
            padded_vectors.append(padded_vec)
            padded_weights.append(padded_weight)
        
        return torch.stack(padded_vectors), torch.stack(padded_weights)
    
    # Separately pad anchor, positive, negative
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
                
                # Forward pass
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
                
                # Forward pass
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
    
    # If there is CPI data, add avg_cpi_loss
    if has_cpi:
        avg_cpi_loss = total_cpi_loss / total_samples
        result['avg_cpi_loss'] = avg_cpi_loss
    
    return result


def main():
    # Set random seed
    set_seed(42)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate BBV model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader worker processes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--max_eval_batches', type=int, default=None, help='Maximum number of evaluation batches, None means evaluate all data')
    parser.add_argument("--with_cpi", type=lambda x: x.lower() == 'true', default=False, help="Enable CPI Regression Pre-train Task")
    
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
    parser.add_argument('--margin', type=float, default=0.3, help='Triplet loss margin')
    parser.add_argument('--loss_type', type=str, default='triplet', choices=['triplet', 'simclr'], 
                        help='Loss function type: triplet or simclr')
    parser.add_argument('--temperature', type=float, default=0.1, 
                        help='Temperature parameter for SimCLR loss')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print(f"CPI enabled: {args.with_cpi}")
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Load model checkpoint
    print(f"Loading model checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        # Checkpoint contains complete training state
        print("Loading model from complete training state checkpoint")
        # Handle with_cpi=False case
        if not args.with_cpi:
            state_dict = checkpoint['model_state_dict']
            # Filter out cpi_head related parameters
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('cpi_head.')}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint contains only model weights
        if not args.with_cpi:
            # Filter out cpi_head related parameters
            filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('cpi_head.')}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint)
    
    # Load test dataset
    print(f"Loading test data: {args.data_path}")
    test_dataset = TripleLossDataset(args.data_path, with_cpi=args.with_cpi)
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=pad_to_max_length_collate_fn
    )
    
    # Evaluate model
    print("Start evaluating model...")
    eval_metrics = evaluate_model(model, test_loader, device, max_eval_batches=args.max_eval_batches)
    
    # Print evaluation results
    print("\nEvaluation results:")
    print("Triplet task:")
    print(f"Average triplet loss: {eval_metrics['avg_triplet_loss']:.4f}")
    print(f"Average positive distance: {eval_metrics['avg_pos_distance']:.4f}")
    print(f"Average negative distance: {eval_metrics['avg_neg_distance']:.4f}")
    print(f"Margin violation rate: {eval_metrics['margin_violation_rate']:.4f}")
    print(f"Distance separation: {eval_metrics['distance_separation']:.4f}")
    
    if 'avg_cpi_loss' in eval_metrics:
        print("\nCPI prediction task:")
        print(f"Average CPI loss: {eval_metrics['avg_cpi_loss']:.4f}")
    
    print(f"\nTotal average loss: {eval_metrics['avg_loss']:.4f}")
    
    # Save evaluation results to file
    result_path = os.path.join(args.output_dir, "eval_results.txt")
    with open(result_path, 'w') as f:
        f.write("Evaluation results:\n")
        f.write("Triplet task:\n")
        f.write(f"Average triplet loss: {eval_metrics['avg_triplet_loss']:.4f}\n")
        f.write(f"Average positive distance: {eval_metrics['avg_pos_distance']:.4f}\n")
        f.write(f"Average negative distance: {eval_metrics['avg_neg_distance']:.4f}\n")
        f.write(f"Margin violation rate: {eval_metrics['margin_violation_rate']:.4f}\n")
        f.write(f"Distance separation: {eval_metrics['distance_separation']:.4f}\n")
        
        if 'avg_cpi_loss' in eval_metrics:
            f.write("\nCPI prediction task:\n")
            f.write(f"Average CPI loss: {eval_metrics['avg_cpi_loss']:.4f}\n")
        
        f.write(f"\nTotal average loss: {eval_metrics['avg_loss']:.4f}\n")
    
    print(f"Evaluation results saved to: {result_path}")


if __name__ == "__main__":
    main()