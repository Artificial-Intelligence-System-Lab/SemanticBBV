import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line to import F
from torch.utils.data import DataLoader, Dataset, Subset
import argparse
import deepspeed
import time
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import datetime
import random
import json

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
from rwkv7_seq import RWKV_x070
from asm_dataset_preprocessed_fine import AsmInstructionPreprocessedDataset, PAD_ID, SEP_ID, CLS_ID, MASK_ID

# Set constants
SEQ_LEN = 32*1024  # 32K sequence length
BATCH_SIZE = 2   # Batch size per GPU
EPOCHS = 1       # Default training epochs, will be overridden by command line arguments
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
LOG_INTERVAL = 10
DATA_DIR = "./output"  # Assembly data directory
USE_BF16 = True  # Enable bf16 training
SEED = 42  # Default random seed


class RWKV7Model(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        model_args = type('', (), {})()
        model_args.n_embd = args.n_embd
        model_args.n_layer = args.n_layer
        model_args.head_size_a = args.head_size

        self.num_embedding_types = 6 # asm, mne, type, reg, rw, eflag
        concatenated_n_embd = 128 * self.num_embedding_types
        
        self.asm_embd = nn.Embedding(vocab_size["asm"], 128)
        self.mne_embd = nn.Embedding(vocab_size["mne"], 128)
        self.type_embd = nn.Embedding(vocab_size["type"], 128)
        self.reg_embd = nn.Embedding(vocab_size["reg"], 128)
        self.rw_embd = nn.Embedding(vocab_size["rw"], 128)
        self.eflag_embd = nn.Embedding(vocab_size["eflag"], 128)

        self.embedding_projection_layer = nn.Linear(concatenated_n_embd, args.n_embd)
        # (Optional) Can add activation function or layer normalization after projection layer
        
        self.model = RWKV(model_args)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 as ignore index
        
        # Add two different output layers for token-level and instruction-level MLM respectively
        self.token_mlm_head = nn.Linear(args.n_embd, vocab_size["asm"])
        self.instr_mlm_head = nn.Linear(args.n_embd, vocab_size["asm"])
        
    def forward(self, inputs, targets=None):
        token_inputs, token_targets = inputs['token_mlm']
        instr_inputs, instr_targets = inputs['instr_mlm']
        
        # Forward pass for token-level task
        # Embed and sum for each dimension
        asm_emb = self.asm_embd(token_inputs['asm'])
        mne_emb = self.mne_embd(token_inputs["mne"])
        type_emb = self.type_embd(token_inputs['type'])
        reg_emb = self.reg_embd(token_inputs['reg'])
        rw_emb = self.rw_embd(token_inputs['rw'])
        eflag_emb = self.eflag_embd(token_inputs['eflag'])
        
        combined_emb_concatenated = torch.cat(
            (asm_emb, mne_emb, type_emb, reg_emb, rw_emb, eflag_emb), 
            dim=-1
        )
        
        # Pass the concatenated embedding through the projection layer
        projected_emb = self.embedding_projection_layer(combined_emb_concatenated)

        # Pass the combined embedding to the model
        token_hidden = self.model(projected_emb)
        token_outputs = self.token_mlm_head(token_hidden)  # Use token-specific output layer
        token_outputs_flat = token_outputs.view(-1, token_outputs.size(-1))
        token_targets_flat = token_targets.reshape(-1)
        token_loss = self.criterion(token_outputs_flat, token_targets_flat)
        
        # Perform the same processing for instruction-level tasks
        instr_asm_emb = self.asm_embd(instr_inputs['asm'])
        instr_mne_emb = self.mne_embd(instr_inputs["mne"])
        instr_type_emb = self.type_embd(instr_inputs['type'])
        instr_reg_emb = self.reg_embd(instr_inputs['reg'])
        instr_rw_emb = self.rw_embd(instr_inputs['rw'])
        instr_eflag_emb = self.eflag_embd(instr_inputs['eflag'])
        
        instr_combined_emb_concatenated = torch.cat(
            (instr_asm_emb, instr_mne_emb, instr_type_emb, instr_reg_emb, instr_rw_emb, instr_eflag_emb),
            dim=-1
        )
        
        # Pass the concatenated embedding through the projection layer
        instr_projected_emb = self.embedding_projection_layer(instr_combined_emb_concatenated)

        instr_hidden = self.model(instr_projected_emb)
        instr_outputs = self.instr_mlm_head(instr_hidden)  # Use instruction-specific output layer
        instr_outputs_flat = instr_outputs.view(-1, instr_outputs.size(-1))
        instr_targets_flat = instr_targets.reshape(-1)
        instr_loss = self.criterion(instr_outputs_flat, instr_targets_flat)
        
        # Combine the two losses
        combined_loss = token_loss + instr_loss
        
        # Return the combined loss and the two sets of outputs along with their respective losses
        return combined_loss, {
            'token_outputs': token_outputs,
            'instr_outputs': instr_outputs,
            'token_loss': token_loss,
            'instr_loss': instr_loss
        }


def adjust_batch_to_max_valid_length(batch_inputs, batch_targets=None, pad_id=PAD_ID, ignore_index=-100):
    """
    Adjust batch data so that its length is the longest valid sequence length in the batch (not including PAD)
    and ensure the final length is a multiple of 16 to match RWKV7 model's CHUNK_LEN
    
    Args:
        batch_inputs: Input dictionary, each key corresponds to a tensor of shape [batch_size, seq_len]
        batch_targets: Target tensor of shape [batch_size, seq_len], can be None
        pad_id: ID of PAD token
        ignore_index: Index value to ignore in loss calculation
        
    Returns:
        Adjusted input and target tensors
    """
    # Get batch size and current sequence length
    batch_size, max_seq_len = batch_inputs['asm'].shape
    
    # Find the valid length of each sequence (not including PAD)
    valid_lengths = []
    for i in range(batch_size):
        # Find the position of the last non-PAD token
        non_pad_mask = (batch_inputs['asm'][i] != pad_id)
        if torch.any(non_pad_mask):
            valid_lengths.append(torch.nonzero(non_pad_mask, as_tuple=True)[0][-1].item() + 1)
        else:
            valid_lengths.append(1)  # Keep at least one token
    
    # Find the maximum valid length in the batch
    max_valid_length = max(valid_lengths)
    
    # Adjust the maximum valid length to a multiple of 16 (round up)
    CHUNK_LEN = 16  # Keep consistent with CHUNK_LEN in rwkv7_cuda.py
    max_valid_length = ((max_valid_length + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
    
    # Process inputs and targets
    if max_valid_length <= max_seq_len:
        # If the adjusted length is less than or equal to the current sequence length, truncate
        adjusted_inputs = {}
        for key in batch_inputs:
            adjusted_inputs[key] = batch_inputs[key][:, :max_valid_length]
        
        if batch_targets is not None:
            adjusted_targets = batch_targets[:, :max_valid_length]
    else:
        # If the adjusted length is greater than the current sequence length, need to add padding
        padding_length = max_valid_length - max_seq_len
        adjusted_inputs = {}
        
        for key in batch_inputs:
            # Create corresponding padding tensor for each key
            padding = torch.full((batch_size, padding_length), pad_id, 
                                dtype=batch_inputs[key].dtype, 
                                device=batch_inputs[key].device)
            adjusted_inputs[key] = torch.cat([batch_inputs[key], padding], dim=1)
        
        if batch_targets is not None:
            padding_targets = torch.full((batch_size, padding_length), ignore_index, 
                                        dtype=batch_targets.dtype, 
                                        device=batch_targets.device)
            adjusted_targets = torch.cat([batch_targets, padding_targets], dim=1)
    
    # Return the adjusted data
    if batch_targets is not None:
        return adjusted_inputs, adjusted_targets
    return adjusted_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--local_rank", type=int, default=-1, help="Used by DeepSpeed")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--test_data_dir", type=str, default=None, help="Test data directory, if not specified, use a portion of training data")
    parser.add_argument("--test_split", type=float, default=0.1, help="If test data directory is not specified, the proportion to split from training data")
    parser.add_argument("--max_seq_len", type=int, default=SEQ_LEN, help="Maximum sequence length")
    parser.add_argument("--min_seq_len", type=int, default=0, help="Minimum sequence length")
    parser.add_argument("--use_bf16", action="store_true", default=USE_BF16, help="Use bf16 precision training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Name of checkpoint to load")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--task_type", type=str, default="mlm", choices=["mlm", "autoregressive"], 
                        help="Pre-training task type: mlm(masked language model) or autoregressive(autoregressive prediction)")
    parser.add_argument("--instruction_level", action="store_true", default=False, help="Whether to use instruction-level masking (only valid in MLM tasks)")
    parser.add_argument("--seed", type=int, default=SEED, help="Global random seed for experiment reproducibility")
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Whether to enable early stopping")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience value, stop after how many consecutive evaluations without improvement")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluate on test set every how many steps, if not specified, evaluate at the end of each epoch")
    parser.add_argument("--mixed_length_training", action="store_true", default=False, help="Whether to enable mixed length training")
    parser.add_argument("--best_loss", type=float, default=None, help="Optimal loss value specified when resuming training")
    parser.add_argument("--hard_sample_mining", action="store_true", default=False, help="Whether to enable hard sample mining")
    parser.add_argument("--hard_sample_ratio", type=float, default=0.4, help="Proportion of samples retained by hard sample mining, default retains the 50% samples with highest loss")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set global constants
    import sys
    sys.modules['__main__'].HEAD_SIZE = args.head_size
    sys.modules['__main__'].USE_CUDA_KERNEL = True
    sys.modules['__main__'].DTYPE = torch.bfloat16 if args.use_bf16 else torch.float32
    sys.modules['__main__'].D_DECAY_LORA = 64
    sys.modules['__main__'].D_AAA_LORA = 64
    sys.modules['__main__'].D_MV_LORA = 64
    sys.modules['__main__'].D_GATE_LORA = 64
    sys.modules['__main__'].MyFunction = lambda x: x
    
    # Create assembly instruction dataset
    dataset = AsmInstructionPreprocessedDataset(
        data_files=[d for d in args.data_dir.split(',') if os.path.exists(d)],
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        cache_dir=None,
        force_preprocess=False,
        filter_by_length=True,
        token_mapping_dir="./tokens_multi_embs",
        mixed_length_training=args.mixed_length_training,
    )
    
    print(f"Loaded {len(dataset)} assembly instruction sequences")
    
    # Set dataset task type
    dataset.set_task(
        task_type=args.task_type,
        instruction_level=args.instruction_level if args.task_type == 'mlm' else False,
    )

    # Create model
    model = RWKV7Model(args, dataset.vocab_size)

    with open(args.deepspeed_config, 'r') as f:
        deepspeed_config_dict = json.load(f)

    # Prepare training and test sets
    if args.test_data_dir:
        # Use separate test data directory
        test_dataset = AsmInstructionPreprocessedDataset(
            data_files=[d for d in args.test_data_dir.split(',') if os.path.exists(d)],
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
            cache_dir=None,
            force_preprocess=False,
            filter_by_length=True,
            token_mapping_dir="./tokens_multi_embs",
            mixed_length_training=args.mixed_length_training,
        )
        test_dataset.set_task(
            task_type=args.task_type,
            instruction_level=args.instruction_level if args.task_type == 'mlm' else False,
            inference_mode=True
        )
        
        print(f"Full test set size: {len(test_dataset)}")

        # --- Modify the following code to create random test subset ---
        test_subset_size = int(len(test_dataset) * 0.1) # Calculate 10% size
        if test_subset_size == 0 and len(test_dataset) > 0:
             test_subset_size = 1 # Ensure at least 1 sample if original dataset is not empty

        all_indices = list(range(len(test_dataset))) # Get indices of all samples
        random.shuffle(all_indices) # Randomly shuffle the index list
        test_indices = all_indices[:test_subset_size] # Select the first 10% indices after shuffling

        test_subset = Subset(test_dataset, test_indices) # Create subset
        print(f"Using randomly sampled test set subset for evaluation, size: {len(test_subset)}")
        # --- Subset creation end ---
        
        test_dataloader = DataLoader(
            test_subset, # <--- Use subset test_subset
            batch_size=deepspeed_config_dict['train_micro_batch_size_per_gpu'],
            shuffle=False, # Subset is already random, no need to shuffle again during evaluation
            num_workers=0,
            pin_memory=False
        )
        print(f"Using separate test set, test set size: {len(test_subset)}")

    if args.mixed_length_training:
        train_dataloader = dataset.create_length_consistent_dataloader(
            batch_size=deepspeed_config_dict['train_micro_batch_size_per_gpu'],
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=args.deepspeed_config,
        )
    else:
        model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=args.deepspeed_config,
            training_data=dataset
        )

    # Load checkpoint (if specified)
    start_epoch = 0
    if args.checkpoint:
        if model_engine.local_rank == 0:
            print(f"Loading checkpoint: {args.checkpoint}")
        _, client_state = model_engine.load_checkpoint(
            CHECKPOINT_DIR,  # Checkpoint directory
            args.checkpoint,  # Checkpoint name
            load_optimizer_states=True,  # Load optimizer states
            load_lr_scheduler_states=True  # Load learning rate scheduler states
        )
        if model_engine.local_rank == 0:
            print(f"Checkpoint loaded successfully")
            
        # Extract epoch number from checkpoint name
        if "_epoch_" in args.checkpoint:
            try:
                start_epoch = int(args.checkpoint.split("_")[-1]) + 1
                if model_engine.local_rank == 0:
                    print(f"Starting training from epoch {start_epoch}")
            except ValueError:
                if model_engine.local_rank == 0:
                    print(f"Unable to extract epoch number from checkpoint name, starting from epoch 0")
        
        if args.hard_sample_mining:
            if model_engine.local_rank == 0:
                print(f"Enabling hard sample mining, will retain the {args.hard_sample_ratio:.1%} samples with highest loss...")
                sample_losses = []
                temp_dataloader = DataLoader(
                    dataset,
                    batch_size=deepspeed_config_dict['train_micro_batch_size_per_gpu'],
                    shuffle=False,
                    num_workers=4
                )

                print("Evaluating training sample losses...")
                model_engine.eval()
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(tqdm(temp_dataloader, desc="Evaluating sample difficulty")):
                        inputs = inputs.to(model_engine.device)
                        targets = targets.to(model_engine.device)
                        
                        # Modified to calculate loss for each sample
                        batch_size = inputs.size(0)
                        outputs = model_engine.model(inputs)
                        
                        # Use reduction='none' to get loss for each token position
                        outputs_flat = outputs.view(-1, outputs.size(-1))
                        targets_flat = targets.reshape(-1)
                        # Calculate loss for each token position
                        losses = F.cross_entropy(outputs_flat, targets_flat, 
                                                ignore_index=-100, reduction='none')
                        
                        # Reshape to [batch_size, seq_len] and calculate average loss for each sample
                        seq_len = targets.size(1)
                        losses = losses.view(batch_size, -1)
                        
                        # Create mask to exclude padding positions (-100)
                        mask = (targets != -100).float()
                        # Calculate number of valid tokens for each sample
                        valid_tokens = mask.sum(dim=1)
                        # Calculate average loss for each sample
                        sample_loss = (losses * mask).sum(dim=1) / valid_tokens.clamp(min=1)
                        
                        # Record loss for each sample
                        for j, loss_val in enumerate(sample_loss):
                            sample_losses.append((i * batch_size + j, loss_val.item()))

                        # Release memory
                        del inputs, targets, outputs, losses, mask, valid_tokens, sample_loss
                        torch.cuda.empty_cache()
                mean_loss = sum(loss for _, loss in sample_losses) / len(sample_losses)
                print(f"Sample loss mean: {mean_loss:.4f}")
            
                # Sort by loss
                sample_losses.sort(key=lambda x: x[1], reverse=True)
            
                # Select samples with highest loss
                num_hard_samples = int(len(sample_losses) * args.hard_sample_ratio)
                hard_sample_indices = [idx for idx, _ in sample_losses[:num_hard_samples]]
            
                # Create hard sample subset
                hard_sample_dataset = torch.utils.data.Subset(dataset, hard_sample_indices)
                print(f"Selected {len(hard_sample_indices)} hard samples (loss > {sample_losses[num_hard_samples-1][1]:.4f})")
            
                # Save hard sample indices for subsequent analysis
                os.makedirs(LOG_DIR, exist_ok=True)
                np.save(f"{LOG_DIR}/hard_sample_indices.npy", np.array(hard_sample_indices))
            
                # Save loss values for all samples
                all_losses = np.array([loss for _, loss in sample_losses])
                np.save(f"{LOG_DIR}/all_sample_losses.npy", all_losses)
                
                # Print loss distribution summary
                loss_vals = [l for _, l in sample_losses]
                print(f"Total samples: {len(loss_vals)}")
                print(f"Top 5 sample losses: {loss_vals[:5]}")
                print(f"Bottom 5 sample losses: {loss_vals[-5:]}")
                print(f"Loss percentiles: 25%={np.percentile(loss_vals, 25):.4f}, 50%={np.percentile(loss_vals, 50):.4f}, 75%={np.percentile(loss_vals, 75):.4f}")
                print(f"Hard sample threshold (Top {args.hard_sample_ratio:.1%}): {sample_losses[num_hard_samples-1][1]:.4f}")
            
                # Plot loss distribution histogram
                plt.figure(figsize=(10, 6))
                plt.hist(all_losses, bins=50)
                plt.axvline(x=mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.4f}')
                plt.axvline(x=sample_losses[num_hard_samples-1][1], color='g', linestyle='--', 
                           label=f'Hard sample threshold: {sample_losses[num_hard_samples-1][1]:.4f}')
                plt.title('Sample Loss Distribution')
                plt.xlabel('Loss Value')
                plt.ylabel('Number of Samples')
                plt.legend()
                plt.grid(True)
                os.makedirs(f"{LOG_DIR}/figures", exist_ok=True)
                plt.savefig(f"{LOG_DIR}/figures/hard_sample_distribution.png")
                plt.close()
            
                print(f"Hard sample mining completed, loss distribution saved to {LOG_DIR}/figures/hard_sample_distribution.png")
            
                # Replace original dataset with hard sample dataset
                original_dataset = dataset
                dataset = hard_sample_dataset
            torch.distributed.barrier()
            
            if model_engine.local_rank != 0 and args.hard_sample_mining:
                hard_sample_indices_path = f"{LOG_DIR}/hard_sample_indices.npy"
                if os.path.exists(hard_sample_indices_path):
                    hard_sample_indices = np.load(hard_sample_indices_path)
                    hard_sample_dataset = torch.utils.data.Subset(dataset, hard_sample_indices)
                    dataset = hard_sample_dataset
                else:
                    print(f"Warning: Process {model_engine.local_rank} cannot find hard sample index file")

            # reload dataloader
            if args.mixed_length_training:
                train_dataloader = dataset.create_length_consistent_dataloader(
                    batch_size=deepspeed_config_dict['train_micro_batch_size_per_gpu'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)
            else:
                # Recreate data loader
                train_dataloader = DataLoader(
                    dataset,
                    batch_size=deepspeed_config_dict['train_micro_batch_size_per_gpu'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
            model_engine.train()

    # Print training precision information
    if model_engine.local_rank == 0:
        print(f"Training with {'BF16' if args.use_bf16 else 'FP32'} precision")
    
    # Create TensorBoard logger
    if model_engine.local_rank == 0:
        os.makedirs(LOG_DIR, exist_ok=True)
        writer = SummaryWriter(log_dir=LOG_DIR)
        # Used to save average loss for each epoch
        epoch_losses = []
        # Used to save loss for each step
        step_losses = []
        global_steps = []
        # Used to save token and instruction losses
        token_step_losses = []
        instr_step_losses = []
        token_steps = []
        # For early stopping
        best_test_loss = args.best_loss if args.best_loss is not None else float('inf')
        if args.best_loss is not None:
            print(f"Restoring best test loss from command line arguments: {best_test_loss:.4f}")
        patience_counter = 0
        test_losses = []
        # Add parameters for early stopping control
        patience = args.patience
    
    # Define evaluation function
    def evaluate():
        model_engine.eval()
        total_test_loss = 0
        total_token_loss = 0
        total_instr_loss = 0
        token_loss_count = 0
        instr_loss_count = 0
        
        # Create evaluation progress bar
        if model_engine.local_rank == 0:
            eval_pbar = tqdm(total=len(test_dataloader), 
                           desc="Evaluating", 
                           position=1, 
                           leave=False,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with torch.no_grad():
            for batch in test_dataloader:
                token_inputs, token_targets = batch['token_level']
                instr_inputs, instr_targets = batch['instruction_level']
                
                # Adjust length of token-level batch data
                token_inputs, token_targets = adjust_batch_to_max_valid_length(
                    token_inputs, token_targets, pad_id=PAD_ID, ignore_index=-100)
                
                # Adjust length of instruction-level batch data
                instr_inputs, instr_targets = adjust_batch_to_max_valid_length(
                    instr_inputs, instr_targets, pad_id=PAD_ID, ignore_index=-100)
                
                for key in token_inputs:
                    token_inputs[key] = token_inputs[key].to(model_engine.device)
                for key in instr_inputs:
                    instr_inputs[key] = instr_inputs[key].to(model_engine.device)
                
                inputs = {
                    'token_mlm': (token_inputs, token_targets.to(model_engine.device)),
                    'instr_mlm': (instr_inputs, instr_targets.to(model_engine.device))
                }
                targets = None  # Targets are included in inputs
                
                # Forward propagation
                test_loss, outputs = model_engine(inputs, targets)
                # Record token_loss and instr_loss
                if isinstance(outputs, dict) and 'token_loss' in outputs and 'instr_loss' in outputs:
                    total_token_loss += outputs['token_loss'].item()
                    total_instr_loss += outputs['instr_loss'].item()
                    token_loss_count += 1
                    instr_loss_count += 1
                
                total_test_loss += test_loss.item()
                
                # Update progress bar
                if model_engine.local_rank == 0:
                    eval_pbar.update(1)
                    eval_pbar.set_description(f"Evaluating [Loss: {test_loss.item():.4f}]")
        
        # Close progress bar
        if model_engine.local_rank == 0:
            eval_pbar.close()
        
        avg_test_loss = total_test_loss / len(test_dataloader)
        model_engine.train()
        
        # Calculate average token_loss and instr_loss
        result_dict = {'combined_loss': avg_test_loss}
        if token_loss_count > 0:
            result_dict['token_loss'] = total_token_loss / token_loss_count
        if instr_loss_count > 0:
            result_dict['instr_loss'] = total_instr_loss / instr_loss_count
            
        return avg_test_loss, result_dict
    
    # Training loop
    global_step = 0
    total_steps = args.epochs * len(train_dataloader)
    early_stop_flag = False  # Add early stopping flag
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model_engine.train()
        total_loss = 0
        start_time = time.time()
        epoch_start_time = time.time()
        
        # Calculate total steps per epoch
        epoch_steps = len(train_dataloader)
        
        # Create progress bar, only display on main process
        if model_engine.local_rank == 0:
            pbar = tqdm(total=epoch_steps, desc=f"Epoch {epoch+1-start_epoch}/{args.epochs}", 
                        position=0, leave=True, 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Handle new output format in training loop
        for step, batch in enumerate(train_dataloader):
            if args.task_type == 'mlm':
                # Handle cases that include both token-level and instruction-level MLM
                token_batch_inputs, token_batch_targets = batch["token_level"]
                instr_batch_inputs, instr_batch_targets = batch["instruction_level"]
                
                # Adjust the length of token-level batch data
                token_inputs, token_targets = adjust_batch_to_max_valid_length(
                    token_batch_inputs, token_batch_targets, pad_id=PAD_ID, ignore_index=-100)
                
                # Adjust the length of instruction-level batch data
                instr_inputs, instr_targets = adjust_batch_to_max_valid_length(
                    instr_batch_inputs, instr_batch_targets, pad_id=PAD_ID, ignore_index=-100)
                
                for key in token_inputs:
                    token_inputs[key] = token_inputs[key].to(model_engine.device)
                for key in instr_inputs:
                    instr_inputs[key] = instr_inputs[key].to(model_engine.device)
                
                inputs = {
                    'token_mlm': (token_inputs, token_targets.to(model_engine.device)),
                    'instr_mlm': (instr_inputs, instr_targets.to(model_engine.device))
                }
                targets = None  # Targets are already included in inputs
            elif args.task_type == 'autoregressive':
                # Directly get token-level and instruction-level data from dictionary
                token_inputs, token_targets = batch['token_level']
                instr_inputs, instr_targets = batch['instruction_level']
                
                # Adjust the length of token-level batch data
                token_inputs, token_targets = adjust_batch_to_max_valid_length(
                    token_inputs, token_targets, pad_id=PAD_ID, ignore_index=-100)
                
                # Adjust the length of instruction-level batch data
                instr_inputs, instr_targets = adjust_batch_to_max_valid_length(
                    instr_inputs, instr_targets, pad_id=PAD_ID, ignore_index=-100)
                
                for key in token_inputs:
                    token_inputs[key] = token_inputs[key].to(model_engine.device)
                for key in instr_inputs:
                    instr_inputs[key] = instr_inputs[key].to(model_engine.device)

                inputs = {
                    'token_mlm': (token_inputs, token_targets.to(model_engine.device)),
                    'instr_mlm': (instr_inputs, instr_targets.to(model_engine.device))
                }
                targets = None  # Targets are already included in inputs
            else:
                raise RuntimeError("Task type is not specified")
            
            # Forward pass and backward pass
            loss, outputs = model_engine(inputs, targets)
            model_engine.backward(loss)
            model_engine.step()
            
            loss_value = loss.item()
            total_loss += loss_value
            global_step += 1
            
            # Update progress bar and estimated completion time
            if model_engine.local_rank == 0:
                # Update progress bar
                pbar.update(1)
                
                # Calculate estimated completion time
                elapsed_steps = epoch * epoch_steps + step + 1
                steps_per_sec = elapsed_steps / (time.time() - epoch_start_time + 1e-6)
                remaining_steps = total_steps - elapsed_steps
                eta_seconds = remaining_steps / steps_per_sec
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # Update progress bar description
                pbar.set_description(
                    f"Epoch {epoch+1}/{args.epochs} [Loss: {loss_value:.4f}] [ETA: {eta_str}]"
                )
                
                # Record loss for each step
                writer.add_scalar('Loss/step', loss_value, global_step)
                step_losses.append(loss_value)
                global_steps.append(global_step)
                
                # Record token_loss and instr_loss (if exist)
                if isinstance(outputs, dict) and 'token_loss' in outputs and 'instr_loss' in outputs:
                    token_loss = outputs['token_loss'].item()
                    instr_loss = outputs['instr_loss'].item()
                    writer.add_scalar('Loss/token_step', token_loss, global_step)
                    writer.add_scalar('Loss/instr_step', instr_loss, global_step)
                    # Record to list
                    token_step_losses.append(token_loss)
                    instr_step_losses.append(instr_loss)
                    token_steps.append(global_step)
                    # Optional: Record the ratio of token_loss and instr_loss
                    writer.add_scalar('Loss/token_instr_ratio', token_loss / (instr_loss + 1e-10), global_step)
                
                # If eval_steps is specified, periodically evaluate on test set
                # Call in eval_steps evaluation
                if args.test_data_dir and args.eval_steps and step > 0 and step % args.eval_steps == 0:
                    test_loss, test_outputs = evaluate()  # Correctly receive two return values
                    writer.add_scalar('Loss/test', test_loss, global_step)
                    test_losses.append(test_loss)
                    print(f"Step {step}/{epoch_steps}, Test set loss: {test_loss:.4f}")
                    
                    # Record token_loss and instr_loss (if exist)
                    if isinstance(test_outputs, dict):
                        if 'token_loss' in test_outputs:
                            writer.add_scalar('Loss/test_token', test_outputs['token_loss'], global_step)
                        if 'instr_loss' in test_outputs:
                            writer.add_scalar('Loss/test_instr', test_outputs['instr_loss'], global_step)
                        if 'token_loss' in test_outputs and 'instr_loss' in test_outputs:
                            print(f"Step {step}/{epoch_steps} Test set Token loss: {test_outputs['token_loss']:.4f}, Instruction loss: {test_outputs['instr_loss']:.4f}")
            
                    # Early stopping check - only when test data is specified
                    if args.early_stopping:
                        if test_loss < best_test_loss:
                            best_test_loss = test_loss
                            patience_counter = 0
                            # Save best model
                            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                            best_checkpoint_name = f"rwkv7_{args.task_type}_best"
                            model_engine.save_checkpoint(CHECKPOINT_DIR, best_checkpoint_name)
                            print(f"Better model found, saved as: {best_checkpoint_name}")
                        else:
                            patience_counter += 1
                            print(f"Test set loss not improved, patience count: {patience_counter}/{patience}")
                            if patience_counter >= patience:
                                print(f"Early stopping triggered! No improvement in {patience} consecutive evaluations.")
                                # Correctly exit all loops
                                print("Ending training early...")
                                # Set flag to break out of outer loop
                                early_stop_flag = True
                                break
            
            if step % LOG_INTERVAL == 0 and model_engine.local_rank == 0:
                elapsed = time.time() - start_time
                samples_per_sec = LOG_INTERVAL * deepspeed_config_dict['train_micro_batch_size_per_gpu'] / elapsed
                
                # Calculate estimated completion time
                eta_seconds = (epoch_steps - step - 1) / (step + 1) * (time.time() - epoch_start_time)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step+1}/{epoch_steps}, Loss: {loss_value:.4f}, "
                      f"Speed: {samples_per_sec:.2f} samples/s, ETA: {eta_str}")
                start_time = time.time()
        
        # Close progress bar
        if model_engine.local_rank == 0:
            pbar.close()
        
        # Check if training needs to end early
        if model_engine.local_rank == 0 and early_stop_flag:
            print("Training ended early due to early stopping condition triggered")
            break
        
        # Calculate and record average loss per epoch
        avg_loss = total_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        
        if model_engine.local_rank == 0:
            writer.add_scalar('Loss/epoch', avg_loss, epoch)
            epoch_losses.append(avg_loss)
            
            # Evaluate on test set at the end of each epoch - only when test data is specified
            if args.test_data_dir:
                test_loss, test_outputs = evaluate()
                writer.add_scalar('Loss/test_epoch', test_loss, epoch)
                test_losses.append(test_loss)
                print(f"Epoch {epoch+1}/{args.epochs} Test set loss: {test_loss:.4f}")
                
                # Record epoch-level token_loss and instr_loss
                if isinstance(test_outputs, dict):
                    if 'token_loss' in test_outputs:
                        writer.add_scalar('Loss/test_token_epoch', test_outputs['token_loss'], epoch)
                    if 'instr_loss' in test_outputs:
                        writer.add_scalar('Loss/test_instr_epoch', test_outputs['instr_loss'], epoch)
                    if 'token_loss' in test_outputs and 'instr_loss' in test_outputs:
                        print(f"Epoch {epoch+1}/{args.epochs} Test set Token loss: {test_outputs['token_loss']:.4f}, Instruction loss: {test_outputs['instr_loss']:.4f}")

                # Early stopping check - only check at epoch level when eval_steps is not used and test data is specified
                if args.early_stopping and not args.eval_steps:
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        patience_counter = 0
                        # Save best model
                        best_checkpoint_name = f"rwkv7_{args.task_type}_best"
                        model_engine.save_checkpoint(CHECKPOINT_DIR, best_checkpoint_name)
                        print(f"Better model found, saved as: {best_checkpoint_name}")
                    else:
                        patience_counter += 1
                        print(f"Test set loss not improved, patience count: {patience_counter}/{patience}")
                        if patience_counter >= patience:
                            print(f"Early stopping triggered! No improvement in {patience} consecutive epochs.")
                            # Correctly exit the loop
                            print("Ending training early...")
                            early_stop_flag = True
                            break
            else:
                # When test data is not specified, print prompt message
                print(f"Test data not specified, skipping test set evaluation and early stopping check, continuing training...")
            
            # Save checkpoint - save regardless of whether there is test data
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint_name = f"rwkv7_{args.task_type}_epoch_{epoch}"
            model_engine.save_checkpoint(CHECKPOINT_DIR, checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}")
            
            # Save loss data to file
            np.save(f"{LOG_DIR}/step_losses.npy", np.array(step_losses))
            np.save(f"{LOG_DIR}/epoch_losses.npy", np.array(epoch_losses))
            np.save(f"{LOG_DIR}/global_steps.npy", np.array(global_steps))
            np.save(f"{LOG_DIR}/test_losses.npy", np.array(test_losses))
            # Save token and instruction loss data
            if token_step_losses:
                np.save(f"{LOG_DIR}/token_step_losses.npy", np.array(token_step_losses))
                np.save(f"{LOG_DIR}/instr_step_losses.npy", np.array(instr_step_losses))
                np.save(f"{LOG_DIR}/token_steps.npy", np.array(token_steps))
            
            # Plot loss curve
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot loss for each step
                plt.subplot(2, 2, 1)
                plt.plot(global_steps, step_losses)
                plt.title('Step Loss')
                plt.xlabel('Global Step')
                plt.ylabel('Loss')
                
                # Plot average loss for each epoch
                plt.subplot(2, 2, 2)
                x_epochs = list(range(len(epoch_losses)))
                plt.plot(x_epochs, epoch_losses)
                plt.title('Epoch Average Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                
                # Plot test set loss
                plt.subplot(2, 2, 3)
                x_test = list(range(len(test_losses)))
                plt.plot(x_test, test_losses)
                plt.title('Test Loss')
                plt.xlabel('Evaluation')
                plt.ylabel('Loss')
                
                # Plot comparison of training and test set losses
                plt.subplot(2, 2, 4)
                plt.plot(x_epochs, epoch_losses, label='Train')
                if len(test_losses) == len(epoch_losses):  # Ensure length matches
                    plt.plot(x_epochs, test_losses, label='Test')
                plt.title('Train vs Test Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                os.makedirs(f"{LOG_DIR}/figures", exist_ok=True)
                plt.savefig(f"{LOG_DIR}/figures/loss_epoch_{epoch}.png")
                plt.close()
            except Exception as e:
                print(f"Error plotting loss curve: {e}")
            
            print(f"Epoch {epoch+1-start_epoch}/{args.epochs} completed, time taken {epoch_time:.2f}s, average loss: {avg_loss:.4f}")
    
    # Training finished, close TensorBoard writer
    if model_engine.local_rank == 0:
        writer.close()
        
        # Plot loss curve for the entire training process
        plt.figure(figsize=(15, 12))
        
        # Original four subplots
        plt.subplot(2, 3, 1)
        plt.plot(global_steps, step_losses)
        plt.title('Training Loss (Step)')
        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        x_epochs = list(range(len(epoch_losses)))
        plt.plot(x_epochs, epoch_losses)
        plt.title('Training Loss (Epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 3, 3)
        x_test = list(range(len(test_losses)))
        plt.plot(x_test, test_losses)
        plt.title('Test Loss')
        plt.xlabel('Evaluation')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Add charts for token_loss and instr_loss
        plt.subplot(2, 3, 4)
        # Directly use the recorded list data
        if token_step_losses and instr_step_losses:
            plt.plot(token_steps, token_step_losses, label='Token Loss')
            plt.plot(token_steps, instr_step_losses, label='Instruction Loss')
            plt.title('Token vs Instruction Loss')
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, "No token/instruction loss data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Token vs Instruction Loss (No Data)')
        
        plt.subplot(2, 3, 5)
        plt.plot(x_epochs, epoch_losses, label='Train')
        if len(test_losses) == len(epoch_losses):
            plt.plot(x_epochs, test_losses, label='Test')
        plt.title('Train vs Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{LOG_DIR}/figures/training_loss_summary.png")
        plt.close()
        
        print(f"Training completed! Loss curves saved to {LOG_DIR}/figures/")

if __name__ == "__main__":
    main()


