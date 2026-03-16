# ML Traffic Classifier in P4 Switch

A complete pipeline for extracting **flow-based** network traffic features from PCAP files or live capture, training ML classifiers using dimensionality-reduced features, and deploying classification rules in P4 behavioral model switches for real-time in-network inference.

**Flow-Based Architecture:** Tracks per-flow statistics (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol) using P4 registers to aggregate packets into flows and extract flow-level features for ML classification.

**Four Dimensionality Reduction Methods (Step 2):**
- **PCA** (`2_pca_generate_entries.py`): Principal Component Analysis — maximises variance
- **LDA** (`2_lda_generate_entries.py`): Linear Discriminant Analysis — maximises class separation
- **UMAP** (`2_umap_generate_entries.py`): Non-linear manifold learning with a stable transform
- **Feature Selection** (`2_feature_selection_generate_entries.py`): Direct classification on top-k raw features with no transform

**Seven Classification Backends (Steps 3 & 4 — unified scripts):**
- **Decision Tree (DT)**: Single table lookup
- **Random Forest (RF)**: Multiple tree voting with packed results
- **XGBoost (XGB)**: Gradient boosted trees with per-class score accumulation
- **Gradient Boosting (GB)**: sklearn boosting; deploys with the same XGB P4 layout
- **K-Nearest Neighbors (KNN)**: Non-parametric baseline (deploys via DT proxy)
- **Support Vector Machine (SVM)**: Margin-based classifier (deploys via DT proxy)
- **1D CNN (CNN)**: PyTorch model with optional P4 neural lookup-table export

## Pipeline Overview

```
Raw PCAP Files (control_plane/pcaps/) or Live Capture
    ↓
[1] Feature Extraction (1_extract_dataset.py)
    ↓
Feature Dataset (dataset/dataset.csv)
    ↓
[2] Dimensionality Reduction — choose ONE:
      PCA              →  2_pca_generate_entries.py
      LDA              →  2_lda_generate_entries.py
      UMAP             →  2_umap_generate_entries.py
      Feature Selection→  2_feature_selection_generate_entries.py
    ↓
Reduction Rules + tables/reduction_config.json
    ↓
[3] Model Training (3_train_model.py --model-type {dt|rf|xgb|gb|knn|svm|cnn})
    ↓
Trained Model (model/<model_type>.model)
    ↓
[4] Model Table Generation (4_generate_model_entries.py --model-type {dt|rf|xgb|gb|cnn})
    ↓
Model Rules (appended to tables/s1-commands.txt)
    ↓
[5] P4 Code Generation (5_generating_p4_code.py --model-type {dt|rf|xgb|gb|cnn})
    ↓
P4 Program (basic.p4)
    ↓
[6] Compile and Deploy (make clean && make run)
    ↓
[7] Runtime Controller (6_controller.py)
    ↓
Live Classification & Monitoring
```

---

## Prerequisites

### Repository Location

**IMPORTANT**: This project must be placed in the `/tutorials/exercises/` directory.

```bash
cd /tutorials/exercises/
git clone git@github.com:vafekt/p4sec.git
cd p4sec
pwd  # Should output: /tutorials/exercises/p4sec
```

### Required Installation

1. **P4 Tools** (BMv2, P4C Compiler, Mininet)
   ```bash
   # Follow official P4 installation guide:
   # https://github.com/p4lang/behavioral-model
   # https://github.com/p4lang/p4c
   # https://github.com/p4lang/tutorials
   ```

2. **Python Dependencies**
   ```bash
   # Install Python packages
   pip install -r requirements.txt
   
   # Install system dependencies
   sudo apt-get install tshark wireshark tcpdump
   ```

3. **PCAP Files**
   - Place `.pcap` files in `control_plane/pcaps/`
   - Filename format: `<label>.v<version>.pcap` (e.g., `skype.v1.pcap`)
   - Labels are auto-extracted from filenames

---

## Step-by-Step Execution

### Step 1: Extract Features

```bash
cd /tutorials/exercises/p4sec/control_plane

# From PCAP/PCAPNG files
python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# From live capture (requires sudo)
sudo python3 1_extract_dataset.py --mode live --interface eth0 --count 1000 --label skype --output dataset/live.csv
```

**Extracts 13 Flow-Based Features:**
- **Protocol**: IP protocol number (6=TCP, 17=UDP, etc.)
- **Duration**: Flow duration in nanoseconds (time from first to last packet)
- **MaxIAT**: Maximum inter-arrival time between consecutive packets in the flow
- **UrgCount**: Number of packets with the TCP URG flag set
- **FwdPktCount**: Number of packets in the forward direction
- **BwdPktCount**: Number of packets in the backward/return direction
- **FwdBytes**: Total payload bytes in the forward direction
- **BwdBytes**: Total payload bytes in the backward direction
- **MaxWinSize**: Maximum TCP window size observed across all packets in the flow
- **FlagsSyn**: Count of packets with the TCP SYN flag set
- **FlagsAck**: Count of packets with the TCP ACK flag set
- **FlagsFin**: Count of packets with the TCP FIN flag set
- **FlagsRst**: Count of packets with the TCP RST flag set

**Flow Aggregation:** Packets are grouped by 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol) to create per-flow statistics. The flow direction is canonicalized so that A→B and B→A map to the same flow entry.

---

### Step 2: Dimensionality Reduction & Generate Base Entries

**IMPORTANT:** Always re-run this step when switching reduction methods or model types so that `s1-commands.txt` is regenerated cleanly.

Choose **one** of the four methods:

#### Option A: PCA (maximises variance)
```bash
cd /tutorials/exercises/p4sec/control_plane

# Auto-detect number of components (95% variance)
python3 2_pca_generate_entries.py

# Or specify number of components and bit width
python3 2_pca_generate_entries.py --components 9 --bits 16
```

#### Option B: LDA (maximises class separation — recommended for better accuracy)
```bash
cd /tutorials/exercises/p4sec/control_plane

# Default: n_classes-1 components
python3 2_lda_generate_entries.py

# Or specify components, bits, and solver
python3 2_lda_generate_entries.py --components 3 --bits 16 --solver svd
```

#### Option C: UMAP (non-linear manifold learning)
```bash
cd /tutorials/exercises/p4sec/control_plane

# Default: 2 components
python3 2_umap_generate_entries.py

# Or specify components, bits, and UMAP hyperparams
python3 2_umap_generate_entries.py --components 3 --bits 16 --n-neighbors 15 --min-dist 0.1
```

#### Option D: Feature Selection (no transform — classify on raw features directly)
```bash
cd /tutorials/exercises/p4sec/control_plane

# Auto-select features using Mutual Information (default)
python3 2_feature_selection_generate_entries.py

# Or specify k features and selection method (mi | chi2 | anova)
python3 2_feature_selection_generate_entries.py --components 8 --method mi
```

**Output (all methods write to the same locations):**
- Reduction parameters and encoding → `tables/pca_encoding_params.json` (legacy shared filename)
- Integer code mapping → `tables/pca_integer_mapping.csv` (legacy shared filename)
- P4 table commands → `tables/s1-commands.txt` (empty for Feature Selection)
- **Universal config** → `tables/reduction_config.json` (read by all subsequent steps)

---

### Step 3: Train Classification Model

A **single unified script** handles all seven classifier backends via `--model-type`:

```bash
cd /tutorials/exercises/p4sec/control_plane

# Decision Tree — fast, interpretable
python3 3_train_model.py --model-type dt

# Random Forest — higher accuracy, ensemble voting
python3 3_train_model.py --model-type rf

# XGBoost — strong performance (requires xgboost package)
python3 3_train_model.py --model-type xgb

# Gradient Boosting — sklearn boosting, no extra dependency
python3 3_train_model.py --model-type gb

# KNN — simple non-parametric baseline (deploys via DT proxy)
python3 3_train_model.py --model-type knn

# SVM — margin-based classifier (deploys via DT proxy)
python3 3_train_model.py --model-type svm

# CNN — PyTorch (P4 export available)
python3 3_train_model.py --model-type cnn

# CNN P4 export (neural lookup tables; no DT/RF surrogate)
python3 3_train_model.py --model-type cnn --p4-export --p4-hidden 8
```

Common optional hyperparameters:
```bash
python3 3_train_model.py --model-type rf \
    --n-estimators 16 \
    --max-depth 6 \
    --random-state 42
```

**Output:** Trained model (`model/<model_type>.model`) and metrics (`tables/<model_type>_metrics.json`)
**Note:** KNN and SVM deploy in P4 via a DecisionTree proxy; run Steps 4–5 as usual.

---

### Step 4: Generate Model-Specific P4 Entries

A **single unified script** handles all deployable backends. Match `--model-type` to what was used in Step 3:

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 4_generate_model_entries.py --model-type dt   # Decision Tree
python3 4_generate_model_entries.py --model-type rf   # Random Forest
python3 4_generate_model_entries.py --model-type xgb  # XGBoost
python3 4_generate_model_entries.py --model-type gb   # Gradient Boosting
python3 4_generate_model_entries.py --model-type cnn  # CNN (neural lookup tables)
```

**Note:** For CNN P4 deployment, train with `--p4-export` first to generate `tables/cnn_params.json`.

**Output:** Model-specific P4 table entries appended to `tables/s1-commands.txt`, human-readable rules in `tables/<model_type>_tree(s).txt`

---

### Step 5: Generate P4 Program

**IMPORTANT:** Use `--model-type` matching your trained model. The generator automatically reads `tables/reduction_config.json` to detect the reduction method (PCA / LDA / Autoencoder / UMAP / Feature Selection) and adapts the P4 code accordingly.

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 5_generating_p4_code.py --model-type dt
python3 5_generating_p4_code.py --model-type rf
python3 5_generating_p4_code.py --model-type xgb
python3 5_generating_p4_code.py --model-type gb
python3 5_generating_p4_code.py --model-type cnn
```

**Note:** For CNN P4 deployment, train with `--p4-export` first to generate `tables/cnn_params.json`.

**Output:** `../basic.p4` and `../p4sec_tofino.p4` (auto-generated with reduction-method- and model-specific logic)

**P4 Architecture (reduction method determines transform stage):**
- **PCA / LDA / Autoencoder / UMAP:** `pca_component*` / `lda_component*` range-match tables map raw features → quantised codes; classifier tables match on codes
- **Feature Selection:** No transform tables; classifier tables match directly on selected raw features
- **Flow Tracking:** P4 registers maintain per-flow state indexed by CRC16/CRC32 hash of the 5-tuple
- **Model-Specific Classification:**
  - **DT:** Single `ml_code` table lookup
  - **RF:** Multiple `rf_tree_i` tables + `rf_vote_classify` for majority voting
  - **XGB / GB:** Multiple `xgb_tree_<c>_<t>` tables + `xgb_classify` with score accumulation
- **Digest Output:** Sends flow features + reduction codes + classification + model outputs to controller

---

### Step 6: Compile and Deploy

```bash
cd /tutorials/exercises/p4sec

# Compile P4 program
make clean
make

# Terminal 1: Start Mininet (FIRST)
sudo make run

# Terminal 2: Start controller (AFTER Mininet is running)
cd /tutorials/exercises/p4sec/control_plane
./6_controller.py
```

**Controller Features:**
- Automatically detects reduction method (PCA / LDA / Autoencoder / UMAP / Feature Selection) and model type from P4Info digest structure (reads `build/basic.p4.p4info.txtpb` — always fresh after `make`)
- Dynamically loads class labels from model parameters
- **Startup mismatch detection:** compares `model.n_features_in_` against the digest's component count; prints a clear one-time warning with remediation steps if they differ and suppresses per-flow error spam
- For **RF**: Shows per-tree vote labels (e.g., `[Tree Votes: T0=Benign T1=DDoS T2=Benign...]`)
- For **XGB / GB**: Shows per-class score accumulation (e.g., `[Scores: c0=150 c1=200 c2=100 c3=80]`)
- Records all predictions to `logs/predictions.csv`
- Universal digest parsing adapts to any number of components and features

---

## Complete Pipeline Examples

### Quick Start with PCA + Decision Tree

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generate_entries.py && \
python3 3_train_model.py --model-type dt && \
python3 4_generate_model_entries.py --model-type dt && \
python3 5_generating_p4_code.py --model-type dt && \
cd .. && make clean && make

# Terminal 1: Start Mininet
sudo make run

# Terminal 2: Start controller (new terminal)
cd /tutorials/exercises/p4sec/control_plane
./6_controller.py
```

### Using LDA + Random Forest

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_lda_generate_entries.py && \
python3 3_train_model.py --model-type rf && \
python3 4_generate_model_entries.py --model-type rf && \
python3 5_generating_p4_code.py --model-type rf && \
cd .. && make clean && make
```

### Using Feature Selection + KNN (DT proxy deployable)

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_feature_selection_generate_entries.py --components 8 && \
python3 3_train_model.py --model-type knn

### Using PCA + SVM (DT proxy deployable)

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generate_entries.py && \
python3 3_train_model.py --model-type svm
```
```

### Using PCA + XGBoost

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generate_entries.py && \
python3 3_train_model.py --model-type xgb && \
python3 4_generate_model_entries.py --model-type xgb && \
python3 5_generating_p4_code.py --model-type xgb && \
cd .. && make clean && make
```

### Using LDA + Gradient Boosting

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_lda_generate_entries.py && \
python3 3_train_model.py --model-type gb && \
python3 4_generate_model_entries.py --model-type gb && \
python3 5_generating_p4_code.py --model-type gb && \
cd .. && make clean && make
```

### Using PCA + CNN (P4 deployable)

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generate_entries.py && \
python3 3_train_model.py --model-type cnn --p4-export && \
python3 4_generate_model_entries.py --model-type cnn && \
python3 5_generating_p4_code.py --model-type cnn && \
cd .. && make clean && make
```

### Switching Reduction Method or Model

**IMPORTANT:** When switching reduction method or model type, always re-run step 2 to regenerate `s1-commands.txt` and `reduction_config.json`:

```bash
cd /tutorials/exercises/p4sec/control_plane

# Example: Switch from PCA+DT to LDA+RF
python3 2_lda_generate_entries.py          # Regenerate reduction tables & config
python3 3_train_model.py --model-type rf  # Train new model
python3 4_generate_model_entries.py --model-type rf
python3 5_generating_p4_code.py --model-type rf
cd .. && make clean && make
```

### Custom PCA Components

```bash
cd /tutorials/exercises/p4sec/control_plane

python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps

# Use 5 PCA components with 12-bit quantization
python3 2_pca_generate_entries.py --components 5 --bits 12

python3 3_train_model.py --model-type rf && \
python3 4_generate_model_entries.py --model-type rf && \
python3 5_generating_p4_code.py --model-type rf && \
cd .. && make clean && make
```

---

## Recent Improvements

### Four Dimensionality Reduction Methods
- **PCA** (`2_pca_generate_entries.py`): principal components that maximise variance
- **LDA** (`2_lda_generate_entries.py`): components that maximise class separation — typically better accuracy with fewer dimensions
- **UMAP** (`2_umap_generate_entries.py`): non-linear manifold learning with a stable transform
- **Feature Selection** (`2_feature_selection_generate_entries.py`): selects the top-k most discriminative raw features (MI / chi² / ANOVA); no transform tables required in P4
- All four methods write a shared `tables/reduction_config.json` consumed by steps 3, 4, and 5

### Unified Scripts for Steps 3 & 4
- **Single `3_train_model.py`** replaces the separate `3_dt_`, `3_rf_`, `3_xgb_` scripts; select the backend with `--model-type {dt|rf|xgb|gb|knn|svm|cnn}`
- **Single `4_generate_model_entries.py`** replaces the separate `4_dt_`, `4_rf_`, `4_xgb_` scripts; same `--model-type` flag (deployable models only)
- **Three additional backends:** KNN (`knn`, DT proxy deployable), SVM (`svm`, DT proxy deployable), and Gradient Boosting (`gb`, deploys as XGB)
- **Neural backend:** CNN (`cnn`, PyTorch; optional P4 lookup-table export)

### Universal `pipeline_utils.py`
- Shared utilities (`detect_feature_columns`, `detect_feature_max_values`, `load_reduction_config`) used by all pipeline steps
- Ensures consistent feature detection regardless of reduction method

### TCP Flag Packet Counts as Features
- **Expanded feature set from 9 to 13** by adding SYN/ACK/FIN/RST as per-packet **counts** (not booleans)
- P4 registers upgraded from `register<bit<1>>` to `register<bit<32>>` for all four flag counters

### Robust Controller Startup
- **Model/digest mismatch detection:** at startup, `model.n_features_in_` is compared against the P4 digest's component count; a clear one-time warning is printed if they differ
- Per-flow error spam is suppressed when a mismatch is already known at startup
- Controller reads `build/basic.p4.p4info.txtpb` (always regenerated by `make`)

### MTU Handling for Large PCAPs
- `send_pcap.sh` and `test_with_pcap.sh` automatically raise the replay interface MTU to 9000 before replaying and restore it afterwards
- Fixes `errno=90 Message too long` errors with DDoS amplification PCAPs

### Universal and Scalable Design
- **Dynamic Feature Count:** Digest parsing adapts to any number of reduction components
- **Schema-Driven:** Reads field layout from P4Info instead of hardcoded offsets
- **Extensible:** Add new features by updating extraction, reduction, and P4 generator — controller adapts automatically

---

## File Structure

```
p4sec/
├── README.md                                        # This file
├── LICENSE
├── Makefile                                         # P4 compilation
├── requirements.txt                                 # Python dependencies
├── basic.p4                                         # Generated P4 program (reduction+model-specific)
├── basic.p4info                                     # P4 program metadata
├── control_plane/
│   ├── pipeline_utils.py                           # Shared utilities (feature detection, config I/O)
│   ├── 1_extract_dataset.py                        # Feature extraction (PCAP/PCAPNG, live)
│   ├── 2_pca_generate_entries.py                 # Step 2 — PCA reduction
│   ├── 2_lda_generate_entries.py                 # Step 2 — LDA reduction
│   ├── 2_umap_generate_entries.py                # Step 2 — UMAP reduction
│   ├── 2_feature_selection_generate_entries.py   # Step 2 — Feature Selection (no transform)
│   ├── 3_train_model.py                         # Step 3 — unified training (dt|rf|xgb|gb|knn|svm|cnn)
│   ├── 4_generate_model_entries.py                 # Step 4 — unified entry generation
│   ├── 5_generating_p4_code.py                     # Step 5 — P4 code generation
│   ├── 6_controller.py                             # Runtime controller (auto-detects all)
│   ├── pcaps/                                      # PCAP/PCAPNG files
│   │   ├── AttackIDS/                             # AttackIDS dataset (Access, CC, Discovery, Evasion)
│   │   └── Mu-IoT/                                # Mu-IoT dataset (Benign, DDoS, PasswordHacking, Reconnaissance)
│   ├── dataset/                                    # Extracted features (dataset.csv)
│   ├── model/                                      # Trained models (dt/rf/xgb/gb/knn/svm .model)
│   ├── tables/                                     # Generated rules & configs
│   │   ├── s1-commands.txt                        # P4 table entries (reduction + model)
│   │   ├── reduction_config.json                  # Universal pipeline config (written by step 2)
│   │   ├── pca_encoding_params.json               # Transform encoding metadata (legacy shared filename)
│   │   ├── pca_integer_mapping.csv                # Quantised codes + labels (legacy shared filename)
│   │   ├── <model_type>_params.json               # RF/XGB/GB metadata
│   │   └── <model_type>_metrics.json              # Accuracy & confusion matrix
│   └── logs/                                      # Runtime logs & predictions
└── build/                                         # P4 compiler output
```

---

## Key Features

### Four Dimensionality Reduction Methods
- **PCA:** Auto-selects components for ≥95% variance (or manually specified)
- **LDA:** At most `min(n_features, n_classes-1)` components; maximises class separation
- **UMAP:** Non-linear manifold learning with a stable transform
- **Feature Selection:** Selects top-k raw features via MI, chi², or ANOVA; no transform stage in P4
- All methods write `tables/reduction_config.json` for seamless downstream compatibility

### Seven ML Backends
1. **Decision Tree (DT):** Single range-match table, fastest inference
2. **Random Forest (RF):** Multiple trees with majority voting
3. **XGBoost (XGB):** Gradient boosted trees with score accumulation
4. **Gradient Boosting (GB):** sklearn boosting; same P4 layout as XGB
5. **K-Nearest Neighbors (KNN):** Non-parametric baseline (deploys via DT proxy)
6. **Support Vector Machine (SVM):** Margin-based classifier (deploys via DT proxy)
7. **1D CNN (CNN):** PyTorch model with optional P4 lookup-table export

### Bidirectional Flow Tracking
- Normalizes flow direction (A→B and B→A map to same flow)
- Canonical 5-tuple: endpoint with smaller IP (then port) always first
- Matches offline extraction for consistent training/inference

### Model-Specific Runtime Verification
- **RF:** Shows which class each tree predicted for debugging
- **XGB / GB:** Displays per-class accumulated scores from dataplane
- **DT:** Simple class output

### Universal Digest Parsing
- Reads digest schema from P4Info at runtime
- Adapts to any feature count, component count, or model outputs
- Fallback to heuristic parsing if schema unavailable
