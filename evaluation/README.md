# Evaluation Tools and Model Inference

This document introduces the evaluation tools and model inference scripts for generating and analyzing basic block vectors (BBV) using trained models.

---

## 🔄 Evaluation Pipeline Overview

```
                Model Inference Stage
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step 1: Load Trained Model                        │
│  ├─ Model checkpoint (.pt)                         │
│  └─ Vocabulary files (*.txt)                       │
│                                                     │
│  Step 2: Convert Assembly to Tokens                │
│  ├─ asm_string_to_token_fine.py                    │
│  └─ Generate tokenized basic blocks                │
│                                                     │
│  Step 3: Generate Basic Block Embeddings           │
│  ├─ asm_bb_to_vector.py                            │
│  └─ Produce vector representations                 │
│                                                     │
└─────────────────────────────────────────────────────┘
                           ↓
                  Analysis Stage
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step 4: Analyze Embeddings                        │
│  ├─ asm_embedding_cluster.py                       │
│  └─ Cluster and visualize embeddings               │
│                                                     │
│  Step 5: Instruction Statistics                    │
│  ├─ asm_tokens_statistic.py                        │
│  └─ Collect instruction category statistics        │
│                                                     │
│  Step 6: Evaluate Results                          │
│  ├─ rank.py                                        │
│  └─ Calculate accuracy, F1, precision, recall      │
│                                                     │
└─────────────────────────────────────────────────────┘
                           ↓
                SimPoint Integration
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Step 7: Modify BBV for SimPoint                   │
│  ├─ simpoint_bbv_modifier.py                       │
│  ├─ modify_bbv.sh                                  │
│  └─ Merge basic block vectors                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Core Scripts Explained

### asm_bb_to_vector.py - Basic Block to Vector Conversion

**Function**: Convert basic blocks to vector embeddings using trained RWKV7 model

**Key Components**:
- **BasicBlockEncoder**: Modified model that accepts single input for embedding generation
- Uses attention mechanism for sequence encoding
- Supports batch processing for efficiency

**Input Data Format**:
- Trained model checkpoint (.pt)
- Vocabulary files (asm, mne, type, reg, rw, eflag tokens)
- Basic block bytecode or assembly instructions

**Output Data Format**:
- Pickle file (.pkl) containing address-to-vector mapping
- Each vector has configurable dimensions (default: 64 or 128)
- Format: `{address: numpy.ndarray, ...}`

**Usage Example**:
```bash
python asm_bb_to_vector.py \
  --model-path model_checkpoint.pt \
  --vocab-dir ./tokens_fine \
  --input basic_blocks.pkl \
  --output bb_embeddings.pkl \
  --encoding-dim 128
```

---

### asm_string_to_token_fine.py - String to Token Conversion (Fine-grained)

**Function**: Convert assembly instruction strings to fine-grained token representations

**Tokenization Process**:
1. Load vocabulary for all 6 token dimensions
2. Parse assembly instruction strings
3. Convert to token IDs using vocabulary
4. Handle unknown tokens with UNK_ID

**Input Data Format**:
- Text file containing assembly instructions (one per line)
- Vocabulary files (*.txt)

**Output Data Format**:
- Tokenized sequences with 6 dimensions:
  - asm: Assembly tokens
  - mne: Mnemonic tokens
  - type: Operand type tokens
  - reg: Register tokens
  - rw: Read/write flags
  - eflag: Flags

**Usage Example**:
```bash
python asm_string_to_token_fine.py \
  --input instructions.txt \
  --vocab-dir ./tokens_fine \
  --output tokenized.pkl
```

---

### asm_string_to_token_fine_concate.py - String to Token with Concatenation

**Function**: Convert assembly strings to tokens with sequence concatenation support

**Features**:
- Concatenates multiple instruction sequences
- Uses RWKV sequential model (RWKV_x070)
- Optimized for long sequence processing

**Input Data Format**:
- Assembly instruction strings
- Vocabulary files

**Output Data Format**:
- Concatenated token sequences
- Ready for sequential model inference

**Usage Example**:
```bash
python asm_string_to_token_fine_concate.py \
  --input instructions.txt \
  --vocab-dir ./tokens_fine \
  --output concatenated_tokens.pkl
```

---

### asm_string_to_token_kmeans.py - String to Token with K-means Clustering

**Function**: Convert assembly strings to tokens with K-means clustering for embedding analysis

**Features**:
- Normalizes instruction format
- Supports K-means clustering on embeddings
- Provides cluster analysis and visualization

**Input Data Format**:
- Assembly instruction strings
- Model checkpoint for embedding generation

**Output Data Format**:
- Token sequences
- Cluster assignments
- Cluster centroids

**Usage Example**:
```bash
python asm_string_to_token_kmeans.py \
  --input instructions.txt \
  --model-path model.pt \
  --num-clusters 50 \
  --output clusters.pkl
```

---

### asm_string_to_token.py - Basic String to Token Conversion

**Function**: Convert assembly instruction strings to basic token format

**Features**:
- Basic instruction normalization
- Vocabulary-based tokenization
- Supports standard assembly formats

**Input Data Format**:
- Text file with assembly instructions
- Vocabulary files

**Output Data Format**:
- Basic token sequences
- Simpler format compared to fine-grained version

---

### asm_embedding_cluster.py - Embedding Clustering Analysis

**Function**: Cluster and analyze basic block embeddings

**Analysis Features**:
- Cosine similarity computation
- Hierarchical clustering
- T-SNE visualization
- Cluster quality metrics

**Input Data Format**:
- Pickle file (.pkl) containing embeddings
- Format: `{address: vector, ...}`

**Output Data Format**:
- Cluster assignments
- Similarity matrices
- Visualization plots (PNG/PDF)
- Cluster statistics

**Usage Example**:
```bash
python asm_embedding_cluster.py \
  --input bb_embeddings.pkl \
  --num-clusters 20 \
  --output-dir ./cluster_results \
  --visualize
```

---

### asm_embedding_server.py - Embedding Server

**Function**: TCP server that provides real-time basic block embedding generation

**Server Features**:
- Socket-based communication
- Multi-threaded request handling
- Supports batch processing
- Real-time embedding generation

**Protocol**:
- Receives binary instruction data
- Returns embedding vectors
- Uses struct packing for efficient transfer

**Input Data Format**:
- TCP socket connections
- Binary instruction data

**Output Data Format**:
- Embedding vectors sent over socket
- Binary format for efficiency

**Usage Example**:
```bash
python asm_embedding_server.py \
  --model-path model.pt \
  --vocab-dir ./tokens_fine \
  --port 8888 \
  --encoding-dim 128
```

---

### asm_categories_collect.py - Instruction Category Collection

**Function**: Collect and sample assembly instructions by category

**Sampling Strategy**:
- Random sampling from each category
- Configurable sample size per category
- Category filtering support
- Handles mismatched instruction/category counts

**Input Data Format**:
- Instructions file (one instruction per line)
- Categories file (one category per line)
- Optional: Token encoding dictionary (.pkl)

**Output Data Format**:
- Sampled instructions organized by category
- Directory structure: `output_dir/category_name/`
- Includes encoding information if provided

**Usage Example**:
```bash
python asm_categories_collect.py \
  --instructions instructions.txt \
  --categories categories.txt \
  --encoding encoding_dict.pkl \
  --output-dir sampled_instructions \
  --sample-size 1000 \
  --ignore-categories "0,unknown"
```

---

### asm_tokens_statistic.py - Token Statistics Analysis

**Function**: Analyze assembly instruction and opcode statistics

**Analysis Types**:
1. Complete instruction frequency
2. Opcode frequency distribution
3. Instruction category classification
4. Statistical reports generation

**Instruction Categories**:
- Data transfer (mov, push, pop, etc.)
- Arithmetic operations (add, sub, mul, div, etc.)
- Logical operations (and, or, xor, etc.)
- Control flow (jmp, call, ret, etc.)
- Comparison and test (cmp, test, etc.)
- Stack operations
- String operations
- System calls
- Floating-point operations
- SIMD/Vector operations
- And more...

**Input Data Format**:
- Text file with assembly instructions
- One instruction per line

**Output Data Format**:
- CSV reports with frequency counts
- Category classification results
- Statistical summaries

**Usage Example**:
```bash
python asm_tokens_statistic.py \
  --input instructions.txt \
  --output-dir ./statistics \
  --top-n 100
```

---

### rank.py - Model Evaluation Ranking

**Function**: Evaluate and rank model performance across different epochs

**Evaluation Metrics**:
- Accuracy
- F1 Score (Macro)
- Precision (Macro)
- Recall (Macro)

**Input Data Format**:
- Directory containing prediction and label files
- Files named: `predictions_epoch_N.npy` and `labels_epoch_N.npy`

**Output Data Format**:
- Sorted performance metrics by epoch
- Console output with rankings
- Best model identification

**Usage Example**:
```bash
python rank.py --results-dir ./inference_results
```

**Output Example**:
```
Rank  Epoch  Accuracy  F1-Macro  Precision  Recall
1     50     0.9234    0.8956    0.9012     0.8901
2     45     0.9187    0.8923    0.8978     0.8869
...
```

---

## 🛠️ SimPoint Integration Tools

### simpoint_bbv_modifier.py - BBV Format Modifier

**Function**: Transform and merge basic block vectors for SimPoint analysis

**Features**:
- Read compressed .gz files from BbTracker
- Merge basic block IDs according to clustering results
- Write modified BBV in SimPoint-compatible format
- Preserve original format structure

**Input Data Format**:
- Input: `.gz` file from Rust BbTracker
- Format: lines of `:bb_id:count` pairs
- Cluster map: Pickle file (.pkl) mapping old IDs to new IDs

**Output Data Format**:
- Output: `.gz` file with merged BB IDs
- Compatible with SimPoint 3.2+
- Maintains original BbTracker format

**Usage Example**:
```bash
python simpoint_bbv_modifier.py \
  --input original.bb.gz \
  --output merged.bb.gz \
  --map clusters.pkl
```

---

### modify_bbv.sh - Batch BBV Modification Script

**Function**: Shell script to batch process multiple BBV files

**Processing Steps**:
1. Find all `.bb` files in directory
2. Locate corresponding cluster map files
3. Run simpoint_bbv_modifier.py on each file
4. Generate modified `.bb.m` files

**Input Data Format**:
- Directory containing `.bb` files
- Corresponding `.vectors.clusters.pkl` files

**Output Data Format**:
- Modified `.bb.m` files
- Ready for SimPoint analysis

**Usage Example**:
```bash
bash modify_bbv.sh /path/to/benchmark/directory
```

---

## 📊 Typical Evaluation Workflow

### Workflow 1: Generate Basic Block Embeddings

```bash
# Step 1: Tokenize assembly instructions
python asm_string_to_token_fine.py \
  --input program_instructions.txt \
  --vocab-dir ../tokens_multi_embs \
  --output tokenized_blocks.pkl

# Step 2: Generate embeddings
python asm_bb_to_vector.py \
  --model-path ../bbv_checkpoints/model_best.pt \
  --vocab-dir ../tokens_multi_embs \
  --input tokenized_blocks.pkl \
  --output bb_embeddings.pkl \
  --encoding-dim 128

# Step 3: Analyze embeddings
python asm_embedding_cluster.py \
  --input bb_embeddings.pkl \
  --num-clusters 30 \
  --output-dir ./cluster_analysis \
  --visualize
```

### Workflow 2: Statistical Analysis

```bash
# Collect instruction statistics
python asm_tokens_statistic.py \
  --input program_instructions.txt \
  --output-dir ./statistics \
  --top-n 100

# Sample by category
python asm_categories_collect.py \
  --instructions program_instructions.txt \
  --categories instruction_categories.txt \
  --output-dir ./sampled_by_category \
  --sample-size 500
```

### Workflow 3: SimPoint Integration

```bash
# Generate BBV from program execution
# (assuming BbTracker output: benchmark.bb.gz)

# Cluster basic blocks
python asm_embedding_cluster.py \
  --input bb_embeddings.pkl \
  --num-clusters 50 \
  --output benchmark.vectors.clusters.pkl

# Modify BBV for SimPoint
python simpoint_bbv_modifier.py \
  --input benchmark.bb.gz \
  --output benchmark.bb.m.gz \
  --map benchmark.vectors.clusters.pkl

# Or batch process
bash modify_bbv.sh /benchmarks/directory
```

### Workflow 4: Model Performance Evaluation

```bash
# Run inference on test set (generates predictions)
# ... (training/inference code)

# Evaluate and rank results
python rank.py --results-dir ./inference_results
```

---

## 🔧 Technical Details

### Model Architecture
- **RWKV7**: Receptance Weighted Key Value architecture
- **Encoding Dimension**: Typically 64 or 128
- **Attention Mechanism**: Self-attention over instruction sequences
- **Multi-embedding**: 6 dimensions (asm, mne, type, reg, rw, eflag)

### Embedding Generation
- Sequence encoding using RWKV7 backbone
- Attention-weighted pooling for final representation
- Normalization and dropout for regularization
- Batch processing support for efficiency

### Clustering Methods
- Cosine similarity for distance metric
- Hierarchical clustering for dendrogram analysis
- K-means for fixed number of clusters
- T-SNE/UMAP for visualization

### Special Token Handling
| Token | ID | Purpose |
|-------|----|----|
| [PAD] | 0 | Padding |
| [SEP] | 1 | Separator |
| [CLS] | 2 | Classification |
| [UNK] | 3 | Unknown token |

### Performance Considerations
1. **GPU Acceleration**: CUDA support for model inference
2. **Batch Processing**: Process multiple blocks simultaneously
3. **Memory Management**: Chunked processing for large datasets
4. **Precision**: Support for BF16/FP16 for faster inference

---

## 📈 Evaluation Metrics

### Clustering Quality
- **Silhouette Score**: Measure of cluster cohesion
- **Davies-Bouldin Index**: Cluster separation metric
- **Within-cluster variance**: Compactness measure

### Classification Metrics
- **Accuracy**: Overall correctness
- **F1 Score (Macro)**: Balanced precision and recall
- **Precision (Macro)**: Average precision across categories
- **Recall (Macro)**: Average recall across categories

### Similarity Metrics
- **Cosine Similarity**: Range [-1, 1], higher is more similar
- **Euclidean Distance**: L2 norm distance
- **Manhattan Distance**: L1 norm distance

---

## 🔍 Common Use Cases

### 1. Program Phase Analysis
Use basic block embeddings to identify different execution phases in programs for SimPoint-style sampling.

### 2. Binary Code Similarity
Compare basic block embeddings to find similar code regions across different binaries or optimization levels.

### 3. Instruction Pattern Recognition
Cluster instructions by semantic similarity to discover common programming patterns.

### 4. Performance Optimization
Identify hotspots and frequently executed basic blocks for targeted optimization.

### 5. Malware Analysis
Compare basic block patterns between known malware and suspicious binaries.

---

## 📝 Notes

1. **Model Compatibility**: Ensure vocabulary files match the training configuration
2. **Memory Requirements**: Large-scale embedding generation may require significant RAM
3. **GPU Utilization**: Use CUDA-enabled devices for faster inference
4. **File Formats**: All pickle files use Python's pickle protocol
5. **SimPoint Version**: BBV format compatible with SimPoint 3.2 and later
6. **Thread Safety**: Embedding server supports concurrent requests
7. **Error Handling**: Unknown instructions mapped to UNK token

---

## 🚀 Quick Start

```bash
# 1. Ensure trained model and vocabularies are available
ls ../bbv_checkpoints/model_best.pt
ls ../tokens_multi_embs/*.txt

# 2. Generate embeddings for your program
python asm_bb_to_vector.py \
  --model-path ../bbv_checkpoints/model_best.pt \
  --vocab-dir ../tokens_multi_embs \
  --input your_program.pkl \
  --output your_embeddings.pkl

# 3. Analyze results
python asm_embedding_cluster.py \
  --input your_embeddings.pkl \
  --num-clusters 20 \
  --visualize

# 4. View statistics
python rank.py --results-dir ./inference_results
```

---

**Document Generated**: December 17, 2025  
**Last Updated**: December 17, 2025
