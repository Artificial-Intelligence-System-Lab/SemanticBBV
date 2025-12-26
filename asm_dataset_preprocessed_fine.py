import os
import pickle
import torch
import json
from torch.utils.data import Dataset, DataLoader, BatchSampler
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import time

PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
MASK_ID = 4

class AsmVocab:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_id = 0
        self.sep_id = 1
        self.cls_id = 2
        self.unk_id = self.pad_id
        self.mask_id = 4

    def load(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if not token:
                    continue
                self.token_to_id[token] = i
                self.id_to_token[i] = token

    def get_id(self, token):
        if token in ["NONE", "NULL"]:
            return self.token_to_id.get("[PAD]", self.unk_id)
        return self.token_to_id.get(token, self.unk_id)
    
    def length(self):
        return len(self.token_to_id)

class AsmInstructionPreprocessedDataset(Dataset):
    """
    Preprocessed version of assembly instruction dataset, preprocess and cache on first load, load cache directly afterwards
    """

    def __init__(
        self,
        data_files: List[str],  # <== Changed to file path list instead of directory list
        max_seq_len: int = 32 * 1024,
        min_seq_len: int = 0,  # Minimum sequence length
        cache_dir: str = None,
        force_preprocess: bool = False,
        filter_by_length: bool = True,  # Whether to filter data by length
        token_mapping_dir: str = None,  # Token mapping file path
        # New parameters
        mixed_length_training: bool = False,  # Whether to enable mixed length training for long and short contexts
        length_config: List[Tuple[int, float]] = [(1024, 0.3), (2048, 0.3), (3072, 0.2), (4096, 0.2)],  # Length configuration list, format [(length1, prob1), (length2, prob2), ...]
        batch_consistent_length: bool = True,  # Whether to ensure consistent length within each batch
    ):
        """
        Initialize dataset
        
        Args:
            data_files: list of pickle file paths
            max_seq_len: maximum sequence length
            min_seq_len: minimum sequence length, sequences shorter than this will be filtered out
            cache_dir: cache directory, defaults to cache subdirectory under current directory
            force_preprocess: whether to force re-preprocessing even if cache files exist
            filter_by_length: whether to filter data based on max_seq_len
            token_mapping_dir: token ID mapping file path, used for token compression
            mixed_length_training: whether to enable mixed length training for long and short contexts
            length_config: length configuration list, format [(length1, prob1), (length2, prob2), ...]
                           For example: [(1024, 0.3), (2048, 0.3), (4096, 0.2), (8192, 0.2)]
            batch_consistent_length: whether to ensure consistent length within each batch
        """
        self.data_files = data_files
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.filter_by_length = filter_by_length
        self.token_mapping_dir = token_mapping_dir

        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(data_files[0])), "cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.vocabs = {}
        vocab_config = {
            "asm": "asm_tokens.txt",
            "mne": "mne_tokens.txt",
            "type": "type_tokens.txt",
            "reg": "reg_tokens.txt",
            "rw": "rw_tokens.txt",
            "eflag": "eflag_tokens.txt"
        }
        for key, filename in vocab_config.items():
            vocab_path = os.path.join(self.token_mapping_dir, filename)
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please ensure all vocabulary files exist in the '{self.token_mapping_dir}' directory.")
            vocab = AsmVocab()
            vocab.load(vocab_path)
            self.vocabs[key] = vocab
        
        self.vocab_size = {
            "asm": self.vocabs["asm"].length(),
            "mne": self.vocabs["mne"].length(),
            "type": self.vocabs["type"].length(),
            "reg": self.vocabs["reg"].length(),
            "rw": self.vocabs["rw"].length(),
            "eflag": self.vocabs["eflag"].length()
        }
        
        # Added: Long and short context mixed training configuration
        self.mixed_length_training = mixed_length_training
        self.batch_consistent_length = batch_consistent_length
        self.current_max_seq_len = max_seq_len  # Initialize current sequence length
        
        # Set length configuration, use default if not provided
        if mixed_length_training and length_config is None:
            # Default length configuration: short sequences (1/4 max length) 30%, medium sequences (1/2 max length) 30%, long sequences (3/4 max length) 20%, max length 20%
            quarter_len = max(min_seq_len, max_seq_len // 4)
            half_len = max(min_seq_len, max_seq_len // 2)
            three_quarter_len = max(min_seq_len, max_seq_len * 3 // 4)
            self.length_config = [
                (quarter_len, 0.3),
                (half_len, 0.3),
                (three_quarter_len, 0.2),
                (max_seq_len, 0.2)
            ]
            print(f"Using default length configuration: {self.length_config}")
        elif mixed_length_training:
            self.length_config = length_config
            print(f"Using custom length configuration: {self.length_config}")
        else:
            self.length_config = [(max_seq_len, 1.0)]
        
        # Initialize data storage
        self.all_tokens = []
        self.instruction_boundaries = []

        # Process each file
        print(f"There are {len(data_files)} data files to process")
        for file_path in tqdm(data_files, desc="Processing data files"):
            if not os.path.exists(file_path):
                print(f"Warning: File does not exist {file_path}")
                continue
                
            # Create a unique cache filename for each file
            file_name = os.path.basename(file_path)
            cache_file = os.path.join(
                self.cache_dir,
                f"{file_name}_compressed_multi_embs.pt"
            )

            if os.path.exists(cache_file) and not force_preprocess:
                print(f"[Loading cache] {cache_file}")
                try:
                    tokens, boundaries = torch.load(cache_file)
                except:
                    tokens = torch.load(cache_file)
                    boundaries = [[] for _ in range(len(tokens))]
            else:
                print(f"[Preprocessing file] {file_path}")
                tokens, boundaries = self._preprocess_single_file(file_path)
                torch.save((tokens, boundaries), cache_file)
                print(f"[Cache saved] => {cache_file}")

            self.all_tokens.extend(tokens)
            self.instruction_boundaries.extend(boundaries)

        # Filter data based on max_seq_len after loading
        if self.filter_by_length:
            self._filter_by_length()

    def _filter_by_length(self):
        """Filter data based on max_seq_len and min_seq_len"""
        original_count = len(self.all_tokens)
        
        # Calculate the effective length of each sequence (excluding padding)
        valid_sequences = []
        valid_boundaries = []
        too_long = 0
        too_short = 0
        
        for i, seq in enumerate(self.all_tokens):
            # If sequence length is greater than max_seq_len, skip
            if len(seq["asm"]) > self.max_seq_len:
                too_long += 1
                continue
            # If sequence length is less than min_seq_len, skip
            if len(seq["asm"]) < self.min_seq_len:
                too_short += 1
                continue
            valid_sequences.append(seq)
            valid_boundaries.append(self.instruction_boundaries[i])
        
        self.all_tokens = valid_sequences
        self.instruction_boundaries = valid_boundaries
        filtered_count = original_count - len(self.all_tokens)
        
        if filtered_count > 0:
            print(f"Filtered by length: removed {filtered_count} sequences")
            print(f"  - Too long (>{self.max_seq_len}): {too_long} sequences")
            print(f"  - Too short (<{self.min_seq_len}): {too_short} sequences")
            print(f"Remaining {len(self.all_tokens)} sequences after filtering")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.all_tokens)

    def _preprocess_single_file(self, file_path):
        """
        Preprocess a single pkl file and display processing progress
        
        Args:
            file_path: pkl file path
            
        Returns:
            tokens_list: processed token list
            boundaries_list: instruction boundary list
        """
        tokens_list = []
        boundaries_list = []

        try:
            with open(file_path, "rb") as f:
                sequences = pickle.load(f)
                # Add progress bar to display sequence processing progress
                for seq in tqdm(sequences, desc=f"Processing sequences in {os.path.basename(file_path)}", unit="seq"):
                    tokens = {
                        "asm": [],
                        "mne": [],
                        "type": [],
                        "reg": [],
                        "rw": [],
                        "eflag": []
                    }
                    
                    boundaries = []
                    current_pos = 0
                    for insn in seq:
                        tokens["asm"].extend([self.vocabs["asm"].get_id(tok) for tok in insn["asm"]])
                        tokens["mne"].extend([self.vocabs["mne"].get_id(insn["mne"])] * len(insn["asm"]))
                        tokens["type"].extend([
                            self.vocabs["type"].get_id(tok) for tok, count in insn["type"] for _ in range(count)
                        ])
                        tokens["reg"].extend([self.vocabs["reg"].get_id(tok) for tok in insn["reg"]])
                        tokens["rw"].extend([
                            self.vocabs["rw"].get_id(tok) for tok, count in insn["rw"] for _ in range(count)
                        ])
                        tokens["eflag"].extend([self.vocabs["eflag"].get_id(insn["eflag"])] * len(insn["asm"]))
                        
                        # Record instruction boundaries
                        insn_len = len(insn["asm"])
                        current_pos += insn_len
                        boundaries.append((current_pos - insn_len, current_pos))

                    tokens_list.append(tokens)
                    boundaries_list.append(boundaries)
        except Exception as e:
            print(f"Preprocessing {file_path} error: {e}")

        return tokens_list, boundaries_list

    def set_random_length(self):
        # If mixed length training is enabled, randomly select a sequence length
        if self.mixed_length_training:
            # Select length based on probability
            lengths, probs = zip(*self.length_config)
            chosen_length = np.random.choice(lengths, p=probs)
            self.current_max_seq_len = chosen_length
        else:
            self.current_max_seq_len = self.max_seq_len
    
    def __getitem__(self, idx: int):
        """Get the sample at the specified index, directly return input and target for MLM or autoregressive tasks"""
        sequence = self.all_tokens[idx]
        boundaries = self.instruction_boundaries[idx]
        
        # If sequence length exceeds max length, truncate
        # If sequence length exceeds max length, truncate at instruction boundaries
        if len(sequence["asm"]) > self.current_max_seq_len:
            # Find the last instruction boundary that completely fits within max length
            last_valid_boundary = -1
            for i, (start, end) in enumerate(boundaries):
                if end < self.current_max_seq_len:
                    last_valid_boundary = i
                else:
                    break
                
            if last_valid_boundary >= 0:
                # Get the end position of the last valid boundary
                truncate_pos = boundaries[last_valid_boundary][1]
                for key in sequence:
                    sequence[key] = sequence[key][:truncate_pos]
                # Keep only valid boundaries
                boundaries = boundaries[:last_valid_boundary+1]
            else:
                # If no valid boundary is found, use default truncation method
                for key in sequence:
                    sequence[key] = sequence[key][:self.current_max_seq_len]
                # Adjust boundaries to ensure all boundaries are within the truncated sequence range
                new_boundaries = []
                for start, end in boundaries:
                    if end < self.current_max_seq_len:
                        new_boundaries.append((start, end))
                boundaries = new_boundaries
        # If sequence length is less than max length, pad
        elif len(sequence["asm"]) < self.current_max_seq_len:
            for key in sequence:
                sequence[key] = sequence[key] + [PAD_ID] * (self.current_max_seq_len - len(sequence[key]))
        
        # Convert each sequence in the dictionary to tensor separately
        tensor_dict = {}
        for key, value in sequence.items():
            tensor_dict[key] = torch.tensor(value, dtype=torch.long)
        
        # Process data based on task type
        if hasattr(self, 'task_type') and self.task_type == 'mlm':
            # MLM task - return both token-level and instruction-level masks
            token_inputs, token_targets = self._create_token_level_mlm(tensor_dict, inference_mode=self.inference_mode)
            instr_inputs, instr_targets = self._create_instruction_level_mlm(tensor_dict, boundaries, inference_mode=self.inference_mode)
            
            # Return two sets of inputs and targets
            return {
                'token_level': (token_inputs, token_targets),
                'instruction_level': (instr_inputs, instr_targets)
            }
        else:
            # Autoregressive task - return both token-level and instruction-level autoregression
            return self._create_autoregressive_sample(tensor_dict, boundaries)
    
    
    def _create_token_level_mlm(self, tensor_dict, mask_prob=0.15, inference_mode=False):
        """
        Create token-level MLM samples, process all dimensions in tensor_dict
        
        Args:
            tensor_dict: dictionary containing token_ids and other fields
            mask_prob: mask probability
            inference_mode: whether in inference mode
            
        Returns:
            inputs_dict: input dictionary
            labels: labels
        """
        # Create copies of inputs and labels
        inputs_dict = {k: v.clone() for k, v in tensor_dict.items()}
        labels = tensor_dict["asm"].clone()
        
        # Get token_ids for creating masks
        token_ids = tensor_dict["asm"]
        
        # Create 15% mask
        probability_matrix = torch.full(token_ids.shape, mask_prob)
        # Do not mask special tokens
        special_tokens_mask = (token_ids == CLS_ID) | (token_ids == SEP_ID) | (token_ids == PAD_ID)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Create a -100 tensor with the same shape as current features
        ignore_tensor = torch.full_like(labels, -100)
        # Keep only the original values at masked positions
        labels = torch.where(masked_indices, labels, ignore_tensor)
        
        if inference_mode:
            # Inference mode: replace all masked positions with MASK_ID
            for key in inputs_dict:
                inputs_dict[key][masked_indices] = MASK_ID
        else:
            # Training mode: introduce randomness
            # 80% of the time replace with [MASK]
            indices_replaced = torch.bernoulli(torch.full(token_ids.shape, 0.8)).bool() & masked_indices
            for key in inputs_dict:
                inputs_dict[key][indices_replaced] = MASK_ID
            
            # 10% of the time replace with random token
            indices_random = torch.bernoulli(torch.full(token_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            for key in inputs_dict:
                random_words = torch.randint(self.vocab_size[key], token_ids.shape, dtype=torch.long)
                inputs_dict[key][indices_random] = random_words[indices_random]
            
            # 10% of the time keep unchanged
            # No additional operation needed, as these positions remain unchanged
        
        return inputs_dict, labels
    
    def _create_instruction_level_mlm(self, tensor_dict, boundaries, mask_prob=0.15, inference_mode=False):
        """
        Create instruction-level MLM samples, process all dimensions in tensor_dict
        
        Args:
            tensor_dict: dictionary containing token_ids and other fields
            mask_prob: mask probability
            inference_mode: whether in inference mode
            
        Returns:
            inputs_dict: input dictionary
            labels: labels
        """
        # Create copies of inputs and labels
        inputs_dict = {k: v.clone() for k, v in tensor_dict.items()}
        labels = tensor_dict["asm"].clone()
        
        # Get token_ids for creating masks
        token_ids = tensor_dict["asm"]
        seq_len = token_ids.size(0)
        
        
        # Ensure boundaries do not exceed sequence length
        valid_boundaries = [(start, end) for start, end in boundaries if start < seq_len and end <= seq_len]
        
        if not valid_boundaries:
            # If no valid boundaries, fall back to token-level masking
            return self._create_token_level_mlm(tensor_dict, mask_prob, inference_mode)
        
        # Decide which instructions to mask
        mask_instruction = torch.bernoulli(torch.full((len(valid_boundaries),), mask_prob)).bool()
        
        # Create a -100 tensor with the same shape as current features
        ignore_tensor = torch.full_like(labels, -100)
        
        # Mask the selected instructions
        for i, ((start, end), mask) in enumerate(zip(valid_boundaries, mask_instruction)):
            if mask:
                # Mask all tokens of the entire instruction
                instruction_length = end - start
                
                if inference_mode:
                    # Inference mode: replace all masked positions with MASK_ID
                    for key in inputs_dict:
                        inputs_dict[key][start:end] = MASK_ID
                else:
                    # Training mode: introduce randomness
                    # 80% of the time replace with [MASK]
                    if torch.rand(1) < 0.8:
                        for key in inputs_dict:
                            inputs_dict[key][start:end] = MASK_ID
                    # 10% of the time replace with random token
                    elif torch.rand(1) < 0.5:
                        for key in inputs_dict:
                            random_words = torch.randint(self.vocab_size[key], (instruction_length,), dtype=torch.long)
                            inputs_dict[key][start:end] = random_words
                    # 10% of the time keep unchanged
                
                # Set labels - keep only the original values at masked positions
                labels[start:end] = token_ids[start:end]
            else:
                # Unselected instructions are marked as -100 in labels
                labels[start:end] = -100
        
        # Set labels for special tokens to -100
        special_tokens_mask = (token_ids == CLS_ID) | (token_ids == SEP_ID) | (token_ids == PAD_ID)
        labels[special_tokens_mask] = -100
        
        return inputs_dict, labels
    
    def _create_autoregressive_sample(self, tensor_dict, boundaries):
        """
        Create autoregressive prediction samples, conforming to traditional autoregressive definition
        
        Args:
            tensor_dict: dictionary containing token_ids and other fields
            boundaries: instruction boundary list
            
        Returns:
            dictionary containing token_level and instruction_level task data
        """
        # Get token_ids
        tensor = tensor_dict["asm"]
        seq_len = tensor.size(0)
        
        # Create new tensor_dict copy
        new_tensor_dict = {k: v.clone() for k, v in tensor_dict.items()}
        tensor = new_tensor_dict["asm"]
        
        # Create two different autoregressive task samples: token-level and instruction-level
        
        # 1. Token-level autoregression (standard next token prediction)
        token_inputs = {}
        for k, v in new_tensor_dict.items():
            token_inputs[k] = v[:-1].clone()  # Input is all tokens except the last one
        
        token_targets = tensor[1:].clone()  # Target is all tokens starting from the second token
        
        # Set targets for PAD and special tokens to -100 (ignore)
        pad_mask = (token_targets == PAD_ID) | (token_targets == CLS_ID)
        token_targets[pad_mask] = -100
        
        # 2. Instruction-level autoregression (predict next instruction)
        # Ensure boundaries do not exceed sequence length
        valid_boundaries = [(start, end) for start, end in boundaries if start < seq_len and end <= seq_len]
        
        if len(valid_boundaries) >= 2:  # At least two instructions are needed for instruction-level autoregression
            # Find the end position of the second-to-last instruction
            second_last_end = valid_boundaries[-2][1]
            
            # Input is all tokens until the end of the second-to-last instruction
            instr_inputs = {}
            for k, v in new_tensor_dict.items():
                instr_inputs[k] = v[:second_last_end].clone()
            
            # Target is all tokens of the last instruction
            last_start, last_end = valid_boundaries[-1]
            instr_targets = torch.full((second_last_end,), fill_value=-100, dtype=torch.long)
            
            # Set the positions of the last instruction as target values
            # Note: We need to align the last instruction to the end of the input sequence
            offset = second_last_end - (last_end - last_start)
            if offset > 0:
                # If there is enough space, align the last instruction to the end of the input sequence
                instr_targets[offset:second_last_end] = tensor[last_start:last_end]
            else:
                # If space is insufficient, use only the alignable part
                instr_targets = tensor[last_start:last_end + offset]
                
            # Set targets for PAD and special tokens to -100 (ignore)
            pad_mask = (instr_targets == PAD_ID) | (instr_targets == CLS_ID)
            instr_targets[pad_mask] = -100
        else:
            # If there are not enough instructions, fall back to token-level autoregression
            instr_inputs = {k: v.clone() for k, v in token_inputs.items()}
            instr_targets = token_targets.clone()
        
        # Ensure input and target lengths are consistent (pad or truncate)
        max_len = self.current_max_seq_len - 1  # Subtract 1 because autoregression task input has one less token
        
        # Process token-level autoregression
        token_inputs_processed = {}
        for k, v in token_inputs.items():
            if len(v) > max_len:
                token_inputs_processed[k] = v[:max_len]
            elif len(v) < max_len:
                # Pad input
                padding = torch.full((max_len - len(v),), PAD_ID, dtype=torch.long)
                token_inputs_processed[k] = torch.cat([v, padding])
            else:
                token_inputs_processed[k] = v
        
        if len(token_targets) > max_len:
            token_targets = token_targets[:max_len]
        elif len(token_targets) < max_len:
            # Pad target (using -100)
            target_padding = torch.full((max_len - len(token_targets),), -100, dtype=torch.long)
            token_targets = torch.cat([token_targets, target_padding])
        
        # Process instruction-level autoregression
        instr_inputs_processed = {}
        for k, v in instr_inputs.items():
            if len(v) > max_len:
                instr_inputs_processed[k] = v[:max_len]
            elif len(v) < max_len:
                # Pad input
                padding = torch.full((max_len - len(v),), PAD_ID, dtype=torch.long)
                instr_inputs_processed[k] = torch.cat([v, padding])
            else:
                instr_inputs_processed[k] = v
        
        if len(instr_targets) > max_len:
            instr_targets = instr_targets[:max_len]
        elif len(instr_targets) < max_len:
            # Pad target (using -100)
            target_padding = torch.full((max_len - len(instr_targets),), -100, dtype=torch.long)
            instr_targets = torch.cat([instr_targets, target_padding])
        
        # Return data for both tasks
        return {
            'token_level': (token_inputs_processed, token_targets),
            'instruction_level': (instr_inputs_processed, instr_targets)
        }


    def create_length_consistent_dataloader(self, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        """Create a data loader that ensures consistent lengths within each batch
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker threads for data loading
            pin_memory: Whether to pin data in memory (beneficial for GPU training)
            
        Returns:
            A special data loader that ensures consistent sequence lengths within each batch
        """
        class LengthConsistentBatchSampler(BatchSampler):
            """Sampler that sets uniform length by batch"""
            
            def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.drop_last = drop_last
                self.num_samples = len(dataset)
                self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size if not drop_last else self.num_samples // self.batch_size
                
                # Get length configuration
                self.lengths, self.probs = zip(*self.dataset.length_config)
                
                print(f"Creating batch consistent length sampler: {self.num_samples} samples, {self.num_batches} batches")
                print(f"Length configuration: {list(zip(self.lengths, self.probs))}")
            
            def __iter__(self):
                # Create index list
                indices = list(range(self.num_samples))
                if self.shuffle:
                    random.shuffle(indices)
                
                # Generate samples by batch
                for i in range(0, len(indices), self.batch_size):
                    if self.drop_last and i + self.batch_size > len(indices):
                        continue
                    
                    # Select a uniform length for the current batch
                    batch_length = np.random.choice(self.lengths, p=self.probs)
                    
                    # Set the current sequence length of the dataset
                    self.dataset.current_max_seq_len = batch_length
                    
                    # Return the indices of the current batch
                    batch_indices = indices[i:i + self.batch_size]
                    if len(batch_indices) < self.batch_size and not self.drop_last:
                        # If the last batch is insufficient, copy some samples to fill
                        batch_indices = batch_indices + [random.choice(batch_indices) for _ in range(self.batch_size - len(batch_indices))]
                    
                    yield batch_indices
            
            def __len__(self):
                return self.num_batches
        
        # Create batch sampler
        batch_sampler = LengthConsistentBatchSampler(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        
        # Create data loader
        return DataLoader(
            dataset=self,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def set_task(self, task_type='mlm', instruction_level=False, vocab_size=None, inference_mode=False, 
                mixed_length=None, length_config=None, batch_consistent_length=None):
        """Set the task type of the dataset
        
        Args:
            task_type: Task type, 'mlm' or 'autoregressive'
            instruction_level: Whether to use instruction-level masking (only valid in MLM tasks)
            vocab_size: Vocabulary size, used for random replacement
            inference_mode: Whether in inference mode (only valid in MLM tasks)
            mixed_length: Whether to enable mixed training of long and short contexts
            length_config: Length configuration list, format [(length1, prob1), (length2, prob2), ...]
            batch_consistent_length: Whether to ensure consistent lengths within each batch
        """
        self.task_type = task_type
        self.instruction_level = instruction_level
        self.inference_mode = inference_mode
        
        # Update mixed length training configuration
        if mixed_length is not None:
            self.mixed_length_training = mixed_length
            
        if length_config is not None:
            self.length_config = length_config
            
        if batch_consistent_length is not None:
            self.batch_consistent_length = batch_consistent_length
            
        
        mode_str = "inference mode" if inference_mode else "training mode"
        mixed_str = "mixed length" if self.mixed_length_training else "fixed length"
        batch_str = "consistent length within batch" if self.batch_consistent_length else "variable length within batch"
        print(f"Dataset task set to: {task_type}, {'instruction-level masking' if instruction_level else 'token-level masking'}, {mode_str}, {mixed_str}, {batch_str}")
        if self.mixed_length_training:
            print(f"Length configuration: {self.length_config}")
        return self
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(size={len(self)}, "
            f"max_seq_len={self.max_seq_len}, "
            f"task_type={getattr(self, 'task_type', 'not set')}, "
            f"mixed_length={self.mixed_length_training}, "
            f"batch_consistent_length={self.batch_consistent_length})")


if __name__ == "__main__":
    # Example usage
    dataset = AsmInstructionPreprocessedDataset(
        data_files=["./dataset/train/train_dataset.pkl"],
        max_seq_len=512,
        min_seq_len=0,
        # First run does not need force_preprocess, set to True if you want to re-preprocess later
        force_preprocess=False,
        filter_by_length=True,  # Enable length filtering
        token_mapping_dir="./tokens_multi_embs",  # Add token mapping path
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample (when task type is not set)
    sample_tokens = dataset.all_tokens[0]
    print(f"Sample type: {type(sample_tokens)}")
    print(f"Sample dict keys: {list(sample_tokens.keys())}")
    print(f"token_ids length: {len(sample_tokens['asm'])}")
    for key, value in sample_tokens.items():
        print(f"tokens {key}: ", value)
    
    sample_tokens = dataset.all_tokens[1]
    print(f"Sample type: {type(sample_tokens)}")
    print(f"Sample dict keys: {list(sample_tokens.keys())}")
    print(f"token_ids length: {len(sample_tokens['asm'])}")
    for key, value in sample_tokens.items():
        print(f"tokens {key}: ", value)
    # Set MLM task - token-level masking
    dataset.set_task(task_type='mlm', instruction_level=False)
    
    # Get an MLM sample
    token_level_data = dataset[0]['token_level']
    mlm_inputs, mlm_targets = token_level_data
    print(f"\nMLM input dict keys: {list(mlm_inputs.keys())}")
    print(f"MLM input token_ids shape: {mlm_inputs['asm'].shape}")
    print(f"MLM labels shape: {mlm_targets.shape}")
    
    # Calculate masking rate
    mask_count = (mlm_inputs['asm'] == MASK_ID).sum().item()
    total_tokens = mlm_inputs['asm'].numel()
    print(f"Token-level masking rate: {mask_count/total_tokens:.2%}")
    
    # Set MLM task - instruction-level masking
    dataset.set_task(task_type='mlm', instruction_level=True)
    
    # Get an instruction-level MLM sample
    instruction_level_data = dataset[0]['instruction_level']
    instr_mlm_inputs, instr_mlm_targets = instruction_level_data
    print(f"\nInstruction-level MLM input dict keys: {list(instr_mlm_inputs.keys())}")
    print(f"Instruction-level MLM input token_ids shape: {instr_mlm_inputs['asm'].shape}")
    print(f"Instruction-level MLM labels shape: {instr_mlm_targets.shape}")
    
    # Calculate masking rate
    mask_count = (instr_mlm_inputs['asm'] == MASK_ID).sum().item()
    total_tokens = instr_mlm_inputs['asm'].numel()
    print(f"Instruction-level masking rate: {mask_count/total_tokens:.2%}")
    
    # Set autoregression task
    dataset.set_task(task_type='autoregressive')
    
    # Get an autoregression sample
    ar_sample = dataset[0]
    
    # Get token-level and instruction-level autoregression samples respectively
    token_inputs, token_targets = ar_sample['token_level']
    instr_inputs, instr_targets = ar_sample['instruction_level']
    
    print(f"\nToken-level autoregression input dict keys: {list(token_inputs.keys())}")
    print(f"Token-level autoregression input token_ids shape: {token_inputs['asm'].shape}")
    print(f"Token-level autoregression target shape: {token_targets.shape}")
    print(f"Instruction-level autoregression input dict keys: {list(instr_inputs.keys())}")
    print(f"Instruction-level autoregression input token_ids shape: {instr_inputs['asm'].shape}")
    print(f"Instruction-level autoregression target shape: {instr_targets.shape}")
    
    # Verify the relationship between token-level autoregression input and target
    token_target_positions = (token_targets != -100).nonzero(as_tuple=True)[0]
    if len(token_target_positions) > 0:
        print(f"\nNumber of masked positions in token-level autoregression task: {len(token_target_positions)}")
        print(f"First masked position: {token_target_positions[0].item()}")
        print(f"Input token at that position: {token_inputs['asm'][token_target_positions[0]]}")
        print(f"Target token at that position: {token_targets[token_target_positions[0]]}")
    else:
        print("\nNo masked positions found in token-level autoregression task")
    
    # Verify the relationship between instruction-level autoregression input and target
    instr_target_positions = (instr_targets != -100).nonzero(as_tuple=True)[0]
    if len(instr_target_positions) > 0:
        print(f"\nNumber of masked positions in instruction-level autoregression task: {len(instr_target_positions)}")
        print(f"First masked position: {instr_target_positions[0].item()}")
        print(f"Last masked position: {instr_target_positions[-1].item()}")
        print(f"Masked position range: {instr_target_positions[0].item()} - {instr_target_positions[-1].item()}")
        
        # Print values for each dimension
        for key in instr_inputs.keys():
            if instr_target_positions[0] < instr_inputs[key].shape[0]:
                print(f"Dimension {key} input value at first masked position: {instr_inputs[key][instr_target_positions[0]]}")
    else:
        print("\nNo masked positions found in instruction-level autoregression task")
    
    # Test data processing for different dimensions
    print("\nTest data processing for different dimensions:")
    for key in mlm_inputs.keys():
        if key != "asm":
            mask_count = (mlm_inputs[key] == MASK_ID).sum().item()
            total = mlm_inputs[key].numel()
            print(f"Masking rate for dimension {key}: {mask_count/total:.2%}")
    
    # Test loading speed
    print("\nTesting loading speed...")
    start_time = time.time()
    # Second instantiation, should load directly from cache
    dataset2 = AsmInstructionPreprocessedDataset(
        data_files=["./dataset/test/test_dataset.pkl"],
        max_seq_len=512,  # Use different max_seq_len to test flexibility
        min_seq_len=64,  # Use different min_seq_len
        filter_by_length=True,
        token_mapping_dir="./tokens_multi_embs",  # Keep consistent token mapping
    )
    print(f"Second load time: {time.time() - start_time:.2f} seconds")
    print(f"Dataset size after using smaller max_seq_len and min_seq_len: {len(dataset2)}")