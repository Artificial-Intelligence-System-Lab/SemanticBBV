# Pre-processing Data Processing Pipeline

This document introduces each step of the **complete data processing and training pipeline** from token partitioning to model training.

---

## 🔄 Complete Data Processing Pipeline

```
                Data Preprocessing Stage
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step 1: Define Token Partitioning                 │
│  ├─ tokenizer.py                                   │
│  └─ Define fine-grained tokenization rules         │
│                                                     │
│  Step 2: Scan and Create Token Dictionaries        │
│  ├─ collect_tokens_fine.py                         │
│  └─ Generate asm/mne/type/reg/rw/eflag_tokens_dict.pkl │
│                                                     │
│  Step 3: Export Token Dictionaries to Text Files   │
│  ├─ export_tokens.py                               │
│  └─ Generate *_tokens.txt + Add special tokens     │
│                                                     │
│  Step 4: Generate Tokenized Data Files             │
│  ├─ asm_dataset_generator.py                       │
│  └─ Generate scattered .pkl files in train/test lib│
│                                                     │
│  Step 5: Filter and Merge Data                     │
│  ├─ merge_tokens.py                                │
│  └─ Filter by length + Random sampling + Chunked merge │
│                                                     │
│  Step 6: Final Data Merge                          │
│  ├─ merge_pickle_files.py                          │
│  └─ Merge into single training dataset file        │
│                                                     │
└─────────────────────────────────────────────────────┘
                           ↓
                Training Preparation Stage
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step 7: Dataset Loading Optimization              │
│  ├─ ../asm_dataset_preprocessed_fine.py            │
│  └─ Build training cache + Map char tokens to vectors │
│                                                     │
└─────────────────────────────────────────────────────┘
                           ↓
                   Model Training Stage
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step 8: Model Training                            │
│  ├─ ../train_rwkv7_deepspeed.py                    │
│  └─ RWKV7 model training code                      │
│                                                     │
│  Step 9: Training Launch Script                    │
│  ├─ ../run_deepspeed.sh                            │
│  └─ DeepSpeed distributed training script          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Core Pipeline Files Explained

### Step 1: tokenizer.py - Token Partitioning Definition

**Function**: Core tokenizer module that establishes fine-grained tokenization rules for instructions

**Tokenization Strategy**:
- **asm**: Complete assembly instruction token list
- **mne**: Mnemonic token
- **type**: Operand type token list
- **reg**: Register token list
- **rw**: Read/write flag token list
- **eflag**: Flag register token list

**Input Data Format**:
- Binary instructions (byte sequence)
- Instruction address

**Output Data Format**:
- Token dictionary containing the above 6 dimensions of token representations
- Uses iced-x86 for instruction decoding and analysis

**Usage Example**:
```python
from tokenizer import tokenize_binary_instruction
tokens = tokenize_binary_instruction(instruction_bytes, address)
```

---

### Step 2: collect_tokens_fine.py - Create Token Dictionaries

**Function**: Scan binary files in training and test libraries to create token dictionaries

**Processing Flow**:
1. Load binary files using angr
2. Decode instructions using iced-x86
3. Call tokenizer.py to tokenize each instruction
4. Count frequency of all tokens

**Input Data Format**:
- Binary executable files (training/test libraries)
- Analysis performed using angr

**Output Data Format**:
Generates 6 Pickle files:
- `asm_tokens_dict.pkl`: Assembly token dictionary {token: frequency}
- `mne_tokens_dict.pkl`: Mnemonic token dictionary
- `type_tokens_dict.pkl`: Operand type token dictionary
- `reg_tokens_dict.pkl`: Register token dictionary
- `rw_tokens_dict.pkl`: Read/write flag token dictionary
- `eflag_tokens_dict.pkl`: Flag register token dictionary

**Usage Example**:
```bash
python collect_tokens_fine.py --binary-dir /path/to/binaries --output-dir ./tokens_fine
```

---

### Step 3: export_tokens.py - Export Token Text Files

**Function**: Export token dictionaries from pickle format to text files and add special tokens

**Special Tokens**:
- `[PAD]` (ID=0): Padding token
- `[SEP]` (ID=1): Separator token
- `[CLS]` (ID=2): Classification token
- `[UNK]` (ID=3): Unknown token
- `[MASK]` (ID=4): Mask token

**Input Data Format**:
- Pickle file (.pkl): token dictionary

**Output Data Format**:
- Text file (.txt)
- One token per line
- Special tokens at the beginning of file
- Remaining tokens sorted alphabetically

**Usage Example**:
```bash
python export_tokens.py --input asm_tokens_dict.pkl --output asm_tokens.txt
```

---

### Step 4: asm_dataset_generator.py - Generate Tokenized Data

**Function**: Create data files in tokenizer format for training and test libraries. Due to large data volume, stored separately

**Data Generation Strategy**:
1. Load CFG (Control Flow Graph)
2. Perform random walks on CFG
3. Tokenize instructions along the walk path
4. Generate formatted sequence data

**Input Data Format**:
- Pickle files containing CFG
- CFG nodes contain raw bytecode (`raw` field)
- Token text files (*.txt)

**Output Data Format**:
- Multiple Pickle files (.pkl), stored separately
- Each file contains list of tokenized instruction sequences
- Sequence format: `[{token_dict_1}, {token_dict_2}, ...]`
- Each token_dict contains 6 dimensions of tokens

**Usage Example**:
```bash
python asm_dataset_generator.py --cfg-dir /path/to/cfg --output-dir ./dataset --vocab-dir ./tokens_fine
```

---

### Step 5: merge_tokens.py - Data Filtering and Merging

**Function**: Filter and merge separated data files based on sequence length

**Filtering Strategy**:
1. **Length Filtering**: Keep sequences within specified length range
2. **Random Sampling**: Sample by ratio (default 20%)
3. **Chunked Merging**: Avoid memory overflow

**Input Data Format**:
- Multiple scattered Pickle files (.pkl)
- Lists containing sequence data
- Sequence format: `[{asm: [...], mne: [...], ...}]`

**Output Data Format**:
- Merged and filtered Pickle file (.pkl)
- Supports output of multiple chunked files
- Data format remains unchanged

**Usage Example**:
```bash
python merge_tokens.py \
  --input-dir ./dataset \
  --output merged_dataset.pkl \
  --min-length 50 \
  --max-length 2048 \
  --sample-ratio 0.2
```

---

### Step 6: merge_pickle_files.py - Final Data Merge

**Function**: Final merge of already merged data files into a single file

**Merge Options**:
- **Merge from directory**: Specify directory and file pattern
- **Merge from file list**: Read path list from txt file
- **Merge type**: list or dict

**Input Data Format**:
- Multiple already merged Pickle files (.pkl)
- Optional: Text file (.txt) containing file paths

**Output Data Format**:
- Single final training dataset file (.pkl)
- Contains all filtered sequence data

**Usage Example**:
```bash
python merge_pickle_files.py \
  --directory ./merged_datasets \
  --output final_train_dataset.pkl \
  --merge-type list
```

---

### Step 7: asm_dataset_preprocessed_fine.py - Dataset Optimization

**Location**: `../asm_dataset_preprocessed_fine.py` (parent directory)

**Function**: Dataset code to build cache for training data

**Optimization Strategies**:
1. **Map character tokens to vectors**: Pre-load vocabulary
2. **Training cache**: Optimize data loading speed
3. **Batch processing optimization**: Improve training processing speed

**Input Data Format**:
- Final training dataset (.pkl)
- Token text files (*.txt)

**Output Data Format**:
- PyTorch Dataset object
- Returns batch data for training in real-time

---

### Step 8: train_rwkv7_deepspeed.py - Model Training

**Location**: `../train_rwkv7_deepspeed.py` (parent directory)

**Function**: RWKV7 model training code with DeepSpeed distributed training support

**Training Features**:
- Multi-GPU training support
- DeepSpeed ZeRO optimization
- Automatic mixed precision training
- Gradient accumulation

**Input Data Format**:
- Training dataset (.pkl)
- Configuration file (ds_config.json)
- Vocabulary files

**Output Data Format**:
- Model checkpoints (.pt)
- Training logs
- Evaluation metrics

---

### Step 9: run_deepspeed.sh - Training Launch Script

**Location**: `../run_deepspeed.sh` (parent directory)

**Function**: DeepSpeed distributed training launch script

**Script Contents**:
- Set environment variables
- Configure GPU devices
- Launch DeepSpeed training
- Set training hyperparameters

**Usage Example**:
```bash
bash run_deepspeed.sh
```

---

## 🛠️ Auxiliary Tool Scripts

### Data Analysis Tools

#### analyze_sequence_length.py
**Function**: Analyze statistical characteristics of sequence lengths in dataset

**Input**: Pickle file generated by `asm_dataset_generator.py`
**Output**: Statistical information (min/max/avg/median) + distribution chart

#### analyze_sequences.py
**Function**: Analyze sequence length distribution in multiple pkl files

**Input**: Multiple `*_blame.pkl` files
**Output**: Sequence length list + distribution visualization chart

#### inspect_pkl_structure.py
**Function**: Inspect and analyze data structure of pickle files

**Input**: Any Pickle file
**Output**: Console output of data structure information

### Visualization Tools

#### plot_token_frequency.py
**Function**: Plot token frequency distribution chart

**Input**: Token dictionary (.pkl)
**Output**: Frequency distribution chart (supports logarithmic scale)

### Format Conversion Tools

#### pickle_token_to_dict.py
**Function**: Convert pickle format token files to text format, add control tokens

**Input**: Token pickle file
**Output**: Token text file + control tokens

#### sort_tokens.py
**Function**: Extract and sort keys from pickle token dictionary

**Input**: Token dictionary (.pkl)
**Output**: Sorted token text file

---

## 📚 Other Data Processing Scripts

### BBV-Related Data Processing

#### asm_bbv_dataset.py
**Function**: Generate BBV (Basic Block Vector) dataset from Rust BbTracker .gz files

**Input**: `.gz` files (Rust BbTracker output)
**Output**: Triplet training data (.pkl) + optional T-SNE visualization

#### asm_bbv_cpi_dataset.py
**Function**: Generate BBV dataset with CPI (Cycles Per Instruction) information

**Input**: `.gz` files + CPI data
**Output**: Triplet training data (.pkl) + CPI information

### Triplet-Loss Data Processing

#### asm_triple_loss_dataset_generator.py
**Function**: Generate triplet-loss training dataset with multiple token types

**Input**: CFG pickle files + multiple vocabularies
**Output**: Function sequences at different optimization levels (.pkl)

#### merge_triple_loss_tokens.py
**Function**: Merge token files required for triplet-loss training

**Input**: Multiple function sequence dictionaries (.pkl)
**Output**: Merged training data (.pkl)

#### extract_triple_dataset.py
**Function**: Merge and sample 10% of data for triplet training

**Input**: Multiple Pickle files
**Output**: Sampled training data (.pkl)

#### extract_triple_dataset_circle_loss.py
**Function**: Merge and sample 10% of data for circle loss training

**Input**: Multiple Pickle files
**Output**: Sampled training data (.pkl)

### Other Text Collection Tools

#### asm_text_collect_iced.py
**Function**: Extract unique assembly instructions using iced-x86

**Input**: `saved_index.pkl` files
**Output**: Unique instruction set + encoding information

#### asm_text_collect_tokenizer.py
**Function**: Generate tokenized dataset using custom tokenizer

**Input**: `saved_index.pkl` + vocabulary
**Output**: Tokenized sequences (.pkl)

#### asm_text_collect_tokenizer_ktrans.py
**Function**: Generate tokenized dataset in K-Transformer format

**Input**: `saved_index.pkl` + vocabulary
**Output**: K-Transformer format sequences (.pkl)

#### collect_tokens.py
**Function**: Collect assembly instruction tokens from binary files (legacy version)

**Input**: Binary executable files
**Output**: Token set + frequency statistics

### Vocabulary Processing Tools

#### compress_vocabulary.py
**Function**: Compress and optimize vocabulary

**Input**: `assemble_tokens.txt`
**Output**: Compressed vocabulary + mapping dictionary

### Parser Utilities

#### program_parser.py
**Function**: Complete program parser (includes random walk, decoding, tokenization)

**Input**: Binary files + vocabulary
**Output**: Tokenized instruction sequences

#### program_parser_func.py
**Function**: Define core assembly parsing classes (AsmVocab, AsmTokenizer)

**Output**: Utility classes and functions for other modules

---

## ⚙️ Technical Details

### Dependencies
- **iced-x86**: x86/x64 instruction decoding
- **angr**: Binary analysis framework
- **PyTorch**: Deep learning framework
- **DeepSpeed**: Distributed training framework

### Special Token Conventions
| Token | ID | Purpose |
|-------|----|----|
| [PAD] | 0 | Padding |
| [SEP] | 1 | Separator |
| [CLS] | 2 | Classification |
| [UNK] | 3 | Unknown token |
| [MASK] | 4 | Mask |

### Performance Optimizations
1. **Multi-processing/Multi-threading**: Accelerate data processing
2. **Chunked processing**: Avoid memory overflow
3. **Random sampling**: Reduce data volume
4. **Caching mechanism**: Optimize training speed

### Data Format Conventions
- Sequence length limit: Maximum 32K tokens
- Minimum length: Configurable (default 100 tokens)
- Pickle serialization: Python pickle module
- Text encoding: UTF-8

---

**Document Generated**: December 17, 2025  
**Last Updated**: December 17, 2025
