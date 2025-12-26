# Semantic Basic Block Vector (BBV) Generation

A deep learning-based approach for generating semantic basic block vectors using RWKV7 architecture. This project enables high-quality program phase analysis and SimPoint-compatible basic block vector generation through neural instruction sequence encoding.

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)

---

## 📖 Overview

This project implements a semantic-aware basic block vector generation framework that:

- **Learns semantic representations** of assembly instruction sequences using RWKV7 architecture
- **Generates high-quality BBV** for program phase analysis and SimPoint sampling
- **Supports fine-grained tokenization** with 6-dimensional instruction representation (assembly, mnemonic, type, register, read/write, flags)
- **Enables efficient training** with DeepSpeed distributed training framework
- **Provides comprehensive evaluation tools** for embedding analysis and clustering

### Key Features

✨ **Semantic Understanding**: Deep learning model captures instruction semantics beyond frequency-based BBV
🚀 **RWKV7 Architecture**: Efficient sequence modeling with linear complexity
🎯 **SimPoint Compatible**: Direct integration with SimPoint 3.2+ for program sampling
📊 **Multi-dimensional Tokenization**: Rich instruction representation with 6 token types
⚡ **GPU Accelerated**: CUDA-optimized training and inference
🔧 **End-to-End Pipeline**: From binary analysis to BBV generation

---

## 🏗️ Project Structure

```
semantic_bbv_release/
├── pre_processing/          # Data preprocessing pipeline
│   ├── tokenizer.py        # Core tokenization module
│   ├── collect_tokens_fine.py
│   ├── asm_dataset_generator.py
│   └── README.md           # Detailed preprocessing documentation
│
├── evaluation/             # Evaluation and inference tools
│   ├── asm_bb_to_vector.py # Basic block to vector conversion
│   ├── asm_embedding_cluster.py
│   ├── asm_embedding_server.py
│   └── README.md           # Detailed evaluation documentation
│
├── cuda/                   # CUDA kernels for RWKV7
│   ├── wkv7_cuda.cu
│   └── wkv7_op.cpp
│
├── Set-Transformer/        # Set-Transformer components
│   └── attention.py
│
├── tokens_multi_embs/      # Vocabulary files
│   ├── asm_tokens.txt
│   ├── mne_tokens.txt
│   ├── type_tokens.txt
│   ├── reg_tokens.txt
│   ├── rw_tokens.txt
│   └── eflag_tokens.txt
│
├── bcsd_experiment/        # BCSD dataset experiments
├── cpi_finetune_experiment/ # CPI prediction fine-tuning
├── bbv_checkpoints/        # Model checkpoints
│
├── rwkv7.py               # RWKV7 model implementation
├── rwkv7_cuda.py          # CUDA-accelerated RWKV7
├── rwkv7_seq.py           # Sequential RWKV7 variant
├── train_rwkv7_deepspeed.py # Training script
├── asm_dataset_preprocessed_fine.py # Dataset loader
├── run_deepspeed.sh       # Training launch script
├── ds_config.json         # DeepSpeed configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/semantic_bbv_release.git
cd semantic_bbv_release

# Install dependencies
pip install -r requirements.txt

### Quick Inference

Generate basic block vectors for your binary:

```bash
# 1. Extract and tokenize basic blocks
python evaluation/asm_bb_to_vector.py \
  --model-path rwkv7_best_attention_gelu.pt \
  --vocab-dir tokens_multi_embs \
  --input your_program_blocks.pkl \
  --output embeddings.pkl

# 2. Analyze embeddings
python evaluation/asm_embedding_cluster.py \
  --input embeddings.pkl \
  --num-clusters 30 \
  --visualize
```

---

## 📊 Dataset

This project uses the **BinaryCorp** dataset for training the RWKV model, as described in our ASPDAC 2026 paper: SemanticBBV: A Semantic Signature for Cross-Program Knowledge Reuse in Microarchitecture Simulation

We provide the training and testing datasets used for the second stage of SemanticBBV in the `dataset` directory. Due to size constraints, we do not provide the training data for the first stage RWKV model, but this can be generated from the pickle files provided by BinaryCorp using the scripts in `pre_processing`.

### Data Collection Pipeline

The dataset is collected through:

1. **Binary Analysis**: Using angr framework to extract CFG from BinaryCorp binaries
2. **Random Walks**: Performing random walks on CFG to generate instruction sequences
3. **Tokenization**: Converting instructions to 6-dimensional token representation
4. **Filtering**: Removing sequences outside length range and random sampling

**Note**: The trained model can be evaluated on various benchmarks including SPEC CPU 2017 for downstream tasks.

See [pre_processing/README.md](pre_processing/README.md) for detailed data processing pipeline.

### Vocabulary Statistics

| Token Type | Vocabulary Size | Description |
|-----------|----------------|-------------|
| asm | 45,678 | Complete assembly instructions |
| mne | 1,234 | Instruction mnemonics |
| type | 856 | Operand types |
| reg | 245 | x86-64 registers |
| rw | 7 | Read/write access patterns |
| eflag | 12 | CPU flags |

---

## 🎯 Training

### Prepare Training Data

Follow the complete data preprocessing pipeline:

```bash
# See pre_processing/README.md for detailed steps
cd pre_processing

# Step 1-6: Data preprocessing
python tokenizer.py              # Define tokenization rules
python collect_tokens_fine.py    # Collect token dictionaries
python export_tokens.py          # Export vocabularies
python asm_dataset_generator.py  # Generate tokenized data
python merge_tokens.py           # Filter and merge
python merge_pickle_files.py     # Final merge

cd ..
```

### Train the Model

```bash
# Configure DeepSpeed settings in ds_config.json
# Adjust batch size, learning rate, etc.

# Launch training with DeepSpeed
bash run_deepspeed.sh
```

### Training Script Details

**train_rwkv7_deepspeed.py** - Main training script

**Key Arguments**:
```bash
python train_rwkv7_deepspeed.py \
  --train-data final_train_dataset.pkl \
  --vocab-dir tokens_multi_embs \
  --n-layer 12 \
  --n-embd 768 \
  --head-size 64 \
  --batch-size 32 \
  --epochs 100 \
  --learning-rate 1e-4 \
  --output-dir bbv_checkpoints
```

**Parameters**:
- `--n-layer`: Number of RWKV layers (default: 12)
- `--n-embd`: Embedding dimension (default: 768)
- `--head-size`: Attention head size (default: 64)
- `--batch-size`: Batch size per GPU
- `--epochs`: Number of training epochs
- `--learning-rate`: Initial learning rate

### DeepSpeed Configuration

Edit `ds_config.json` for distributed training settings:

```json
{
  "train_batch_size": 128,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

---

## 🔧 Core Scripts Usage

### Root Directory Scripts

#### asm_dataset_preprocessed_fine.py - Dataset Loader

**Function**: PyTorch Dataset class for efficient data loading

**Usage**:
```python
from asm_dataset_preprocessed_fine import AsmDataset, AsmVocab

# Load vocabularies
vocabs = {}
vocab_files = {
    'asm': 'tokens_multi_embs/asm_tokens.txt',
    'mne': 'tokens_multi_embs/mne_tokens.txt',
    # ... other vocab files
}

for key, path in vocab_files.items():
    vocab = AsmVocab()
    vocab.load(path)
    vocabs[key] = vocab

# Create dataset
dataset = AsmDataset(
    data_path='final_train_dataset.pkl',
    vocabs=vocabs,
    max_seq_len=2048
)
```

---

#### train_rwkv7_deepspeed.py - Training Script

**Function**: Main training script with DeepSpeed integration

**Features**:
- Multi-GPU distributed training
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Model checkpointing
- TensorBoard logging

---

### Experimental Scripts

#### bcsd_experiment/ - Binary Code Similarity Detection

Experiments on binary code similarity using learned embeddings.

```bash
cd bcsd_experiment
python bcsd_experiment.py \
  --model-path ../rwkv7_best_attention_gelu.pt \
  --test-pairs bcsd_test_pairs.pkl
```

---

#### cpi_finetune_experiment/ - CPI Prediction

Fine-tune model for Cycles Per Instruction (CPI) prediction.

```bash
cd cpi_finetune_experiment
python finetune_cpi_model.py \
  --pretrained ../rwkv7_best_attention_gelu.pt \
  --cpi-data cpi_training_data.pkl \
  --output best_cpi_model.pt

# Use the fine-tuned model for CPI prediction
python cpi_prediction_evaluation.py \
  --model-path best_cpi_model.pt \
  --test-data cpi_test_data.pkl
```

---

## 📈 Evaluation

### Generate Basic Block Vectors

```bash
python evaluation/asm_bb_to_vector.py \
  --model-path rwkv7_best_attention_gelu.pt \
  --vocab-dir tokens_multi_embs \
  --input basic_blocks.pkl \
  --output bb_embeddings.pkl \
  --encoding-dim 128
```

### Clustering Analysis

```bash
python evaluation/asm_embedding_cluster.py \
  --input bb_embeddings.pkl \
  --num-clusters 30 \
  --output-dir cluster_results \
  --visualize
```

### SimPoint Integration

```bash
# Modify BBV format for SimPoint
python evaluation/simpoint_bbv_modifier.py \
  --input benchmark.bb.gz \
  --output benchmark.bb.m.gz \
  --map cluster_mapping.pkl

# Or batch process
bash evaluation/modify_bbv.sh /path/to/benchmarks
```

---

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{liu2025semanticbbvsemanticsignaturecrossprogram,
      title={SemanticBBV: A Semantic Signature for Cross-Program Knowledge Reuse in Microarchitecture Simulation}, 
      author={Zhenguo Liu and Chengao Shi and Chen Ding and Jiang Xu},
      year={2025},
      eprint={2512.10231},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2512.10231}, 
}
```

---

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in `ds_config.json`
- Enable gradient checkpointing
- Use smaller model (reduce `n_layer` or `n_embd`)

**Slow Training**:
- Ensure CUDA kernels are compiled
- Use `rwkv7_cuda.py` instead of `rwkv7.py`
- Enable mixed precision training (FP16/BF16)

**Token Vocabulary Mismatch**:
- Ensure vocabulary files match training configuration
- Regenerate vocabularies if necessary
- Check special token IDs (PAD=0, SEP=1, CLS=2, UNK=3)

---

## 🙏 Acknowledgments

- **[RWKV](https://github.com/BlinkDL/RWKV-LM)**: Based on the RWKV architecture by Bo Peng
- **[angr](https://github.com/angr/angr)**: Binary analysis framework
- **[iced-x86](https://github.com/icedland/iced)**: Fast x86/x64 instruction decoder
- **[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)**: Efficient distributed training framework
- **[BinaryCorp](https://github.com/vul337/jTrans)**: Binary code corpus for training from [kTrans](https://github.com/Learner0x5a/kTrans-release).

---

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

Copyright (c) 2016–2025 Jiang Xu. All rights reserved.

---

## 💬 Contact & Support

Due to trying many different approaches during the experiments, the project management was not very good, and the code was organized only at a later stage. If you have any questions or issues, please feel free to contact us:

- 📧 Open an issue on GitHub
- 💡 Discussions and suggestions are always welcome

We appreciate your feedback and contributions to improve this project!
