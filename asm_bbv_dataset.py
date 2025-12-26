import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

class TripleLossDataset(Dataset):
    """
    Triple Loss learning dataset class
    
    Return format: (anchor_dict, positive_dict, negative_dict)
    Each dictionary contains features of assembly instructions as tensors
    """
    
    def __init__(self, pickle_file_path: str, with_cpi: bool = False):
        """
        Initialize dataset
        
        Args:
            pickle_file_path: path to pickle file
            max_sequence_length: maximum sequence length for padding/truncating
        """
        self.with_cpi = with_cpi
        self.data = self._load_data(pickle_file_path)
        
        ## Get feature dimension information
        if self.data:
            vector, weight = self.data[0][0], self.data[0][1]  # anchor of the first tuple
            print(f"Dataset size: {len(self.data)}")
    
    def _load_data(self, pickle_file_path: str) -> List[Tuple]:
        """Load pickle data file and filter out items with length > 2k"""
        try:
            with open(pickle_file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Raw data loaded successfully, contains {len(data)} triplets")
            
            # Filter out items with length > 2k
            filtered_data = []
            max_len = 0
            for item in data:
                anchor_vector, _, positive_vector, _, negative_vector, _ = item[:6]
                # if len(anchor_vector) <= 5000 and len(positive_vector) <= 5000 and len(negative_vector) <= 5000:
                #     if self.with_cpi:
                #         filtered_data.append(item)
                #     else:
                #         filtered_data.append(item[:6])
                max_len = max(max_len, len(anchor_vector), len(positive_vector), len(negative_vector))
                if self.with_cpi:
                    filtered_data.append(item)
                else:
                    filtered_data.append(item[:6])
            
            print(f"Filtered data contains {len(filtered_data)} triplets, filtered out {len(data) - len(filtered_data)} overlength triplets, max_len={max_len}")
            return filtered_data
        except Exception as e:
            print(f"Failed to load data: {e}")
            return []
    

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.data)
    

    def __getitem__(self, idx: int):
        """
        Get a triplet sample
        Args:
            idx: sample index

        Returns:
            (anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight) sextuple, each element is a tensor
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of dataset range {len(self.data)}")
        
        anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight = self.data[idx][:6]
        
        # Use np.array() to first convert the list to a single numpy array, then convert to tensor
        anchor_vector = torch.tensor(np.array(anchor_vector), dtype=torch.float32)
        anchor_weight = torch.tensor(np.array(anchor_weight), dtype=torch.float32)
        positive_vector = torch.tensor(np.array(positive_vector), dtype=torch.float32)
        positive_weight = torch.tensor(np.array(positive_weight), dtype=torch.float32)
        negative_vector = torch.tensor(np.array(negative_vector), dtype=torch.float32)
        negative_weight = torch.tensor(np.array(negative_weight), dtype=torch.float32)

        def normalize_weight(weight):
            # Convert weights to float for normalization
            weight = weight.float()
            # Find mask for non-zero weights
            mask = weight > 0
            if mask.sum() > 0:  # Ensure there are non-zero weights
                # Only normalize non-zero weights
                non_zero_weights = weight[mask]
                min_val = non_zero_weights.min()
                max_val = non_zero_weights.max()
                if min_val < max_val:  # Avoid division by zero
                    # Normalize to [0.001, 0.999] range, retain certain minimum weight
                    normalized = 0.001 + 0.999 * (non_zero_weights - min_val) / (max_val - min_val)
                    weight[mask] = normalized
            return weight
        
        # Normalize three weights
        anchor_weight = normalize_weight(anchor_weight)
        positive_weight = normalize_weight(positive_weight)
        negative_weight = normalize_weight(negative_weight)

        if self.with_cpi:
            anchor_cpi, positive_cpi, negative_cpi = self.data[idx][6:]
            for cpi in (anchor_cpi, positive_cpi, negative_cpi):
                if cpi < 0:
                    print(f"cpi < 0: {cpi}")
                
            # Apply log-normal distribution transformation
            anchor_cpi = np.log(max(anchor_cpi, 1e-8))
            positive_cpi = np.log(max(positive_cpi, 1e-8))
            negative_cpi = np.log(max(negative_cpi, 1e-8))
            return anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight, anchor_cpi, positive_cpi, negative_cpi

        return anchor_vector, anchor_weight, positive_vector, positive_weight, negative_vector, negative_weight
