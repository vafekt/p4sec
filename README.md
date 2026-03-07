# PCA with Machine Learning Traffic Classifier in P4 Switch

A complete pipeline for extracting **flow-based** network traffic features from PCAP files or live capture, training ML classifiers (Decision Tree, Random Forest, or XGBoost) using PCA-reduced dimensions, and deploying classification rules in P4 behavioral model switches for real-time in-network inference.

**Flow-Based Architecture:** Tracks per-flow statistics (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol) using P4 registers to aggregate packets into flows and extract flow-level features for ML classification.

**Multi-Model Support:** Supports three classification backends with unified pipeline:
- **Decision Tree (DT)**: Single table lookup (default)
- **Random Forest (RF)**: Multiple tree voting with packed results
- **XGBoost (XGB)**: Gradient boosted trees with per-class score accumulation

## Pipeline Overview

```
Raw PCAP Files (control_plane/pcaps/) or Live Capture
    ↓
[1] Feature Extraction (1_data_extraction.py)
    ↓
Feature Dataset (dataset/dataset.csv)
    ↓
[2] PCA Training & Encoding (2_pca_generating_entries.py)
    ↓
PCA Rules (tables/s1-commands.txt)
    ↓
[3] Model Training (3_dt/rf/xgb_training_model.py)
    ↓
Trained Model (model/{dt,rf,xgb}.model)
    ↓
[4] Model Table Generation (4_dt/rf/xgb_generating_entries.py)
    ↓
Model Rules (appended to tables/s1-commands.txt)
    ↓
[5] P4 Code Generation (5_generating_p4_code.py --model-type {dt,rf,xgb})
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
git clone <repository-url> p4-pca-dt
cd p4-pca-dt
pwd  # Should output: /tutorials/exercises/p4-pca-dt
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
cd /tutorials/exercises/p4-pca-dt/control_plane

# From PCAP/PCAPNG files
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# From live capture (requires sudo)
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label skype --output dataset/live.csv
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

### Step 2: Train PCA and Generate Base Entries

**IMPORTANT:** Always run this step when switching between model types (DT/RF/XGB) to ensure the PCA entries in `s1-commands.txt` don't conflict with model-specific table names.

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Auto-detect number of components (95% variance)
python3 2_pca_generating_entries.py

# Or specify number of components and bit width
python3 2_pca_generating_entries.py --components 9 --bits 16
```

**Output:** 
- PCA parameters and encoding (`tables/pca_encoding_params.json`)
- Integer PCA codes mapping (`tables/pca_integer_mapping.csv`)
- P4 table commands for PCA transformation (`tables/s1-commands.txt`)
- DecisionTreeRegressor that maps raw features → PCA codes (used for P4 rule generation)

---

### Step 3: Train Classification Model

Choose one model type:

#### Option A: Decision Tree (Simple, Fast)
```bash
python3 3_dt_training_model.py
```

#### Option B: Random Forest (Higher Accuracy)
```bash
python3 3_rf_training_model.py
```

#### Option C: XGBoost (Best Performance)
```bash
python3 3_xgb_training_model.py
```

**Output:** Trained model (`model/{dt,rf,xgb}.model`) and classification metrics

---

### Step 4: Generate Model-Specific P4 Entries

Match the model type from Step 3:

#### For Decision Tree:
```bash
python3 4_dt_generating_entries.py
```

#### For Random Forest:
```bash
python3 4_rf_generating_entries.py
```

#### For XGBoost:
```bash
python3 4_xgb_generating_entries.py
```

**Output:** Model-specific P4 table entries appended to `tables/s1-commands.txt`

---

### Step 5: Generate P4 Program

**IMPORTANT:** Use `--model-type` matching your trained model:

#### For Decision Tree:
```bash
python3 5_generating_p4_code.py --model-type dt
```

#### For Random Forest:
```bash
python3 5_generating_p4_code.py --model-type rf
```

#### For XGBoost:
```bash
python3 5_generating_p4_code.py --model-type xgb
```

**Output:** `../basic.p4` (auto-generated with model-specific logic)

**P4 Architecture:**
- **Flow Tracking:** P4 registers maintain per-flow state (timestamps, byte/packet counts, flag counts) indexed by CRC16/CRC32 hash of the 5-tuple
- **13 Features in Hardware:** Protocol, Duration, MaxIAT, UrgCount, FwdPktCount, BwdPktCount, FwdBytes, BwdBytes, MaxWinSize, plus four TCP flag packet-count registers (SYN/ACK/FIN/RST as `bit<32>`)
- **PCA Tables:** N tables (pca_component1-N) map the 13 raw features to quantized PCA codes via range-match rules
- **Model-Specific Classification:**
  - **DT:** Single `ml_code` table lookup
  - **RF:** Multiple `rf_tree_i` tables + `rf_vote_classify` for majority voting
  - **XGB:** Multiple `xgb_tree_<c>_<t>` tables + `xgb_classify` with score accumulation
- **Digest Output:** Sends flow features + PCA codes + classification + model outputs to controller

---

### Step 6: Compile and Deploy

```bash
cd /tutorials/exercises/p4-pca-dt

# Compile P4 program
make clean
make

# Terminal 1: Start Mininet (FIRST)
sudo make run

# Terminal 2: Start controller (AFTER Mininet is running)
cd /tutorials/exercises/p4-pca-dt/control_plane
./6_controller.py
```

**Controller Features:**
- Automatically detects model type from P4Info digest structure (reads `build/basic.p4.p4info.txtpb` — always fresh after `make`)
- Dynamically loads class labels from model parameters
- **Startup mismatch detection:** compares `model.n_features_in_` against the digest's PCA component count; prints a clear one-time warning with remediation steps if they differ and suppresses per-flow error spam
- For **RF**: Shows per-tree vote labels (e.g., `[Tree Votes: T0=Benign T1=DDoS T2=Benign...]`)
- For **XGB**: Shows per-class score accumulation (e.g., `[Scores: c0=150 c1=200 c2=100 c3=80]`)
- Records all predictions to `logs/predictions.csv`
- Universal digest parsing adapts to any number of PCA components and features

---

## Complete Pipeline Examples

### Quick Start with Decision Tree

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Full DT pipeline
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generating_entries.py && \
python3 3_dt_training_model.py && \
python3 4_dt_generating_entries.py && \
python3 5_generating_p4_code.py --model-type dt && \
cd .. && make clean && make

# Terminal 1: Start Mininet
sudo make run

# Terminal 2: Start controller (new terminal)
cd /tutorials/exercises/p4-pca-dt/control_plane
./6_controller.py
```

### Using Random Forest

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Full RF pipeline
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generating_entries.py && \
python3 3_rf_training_model.py && \
python3 4_rf_generating_entries.py && \
python3 5_generating_p4_code.py --model-type rf && \
cd .. && make clean && make
```

### Using XGBoost

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Full XGB pipeline
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generating_entries.py && \
python3 3_xgb_training_model.py && \
python3 4_xgb_generating_entries.py && \
python3 5_generating_p4_code.py --model-type xgb && \
cd .. && make clean && make
```

### Switching Between Models

**IMPORTANT:** When switching model types, always regenerate PCA entries to avoid table name conflicts:

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Example: Switch from DT to RF
python3 2_pca_generating_entries.py     # Clear and regenerate base entries
python3 3_rf_training_model.py          # Train new model
python3 4_rf_generating_entries.py      # Generate RF table entries
python3 5_generating_p4_code.py --model-type rf
cd .. && make clean && make
```

**Expected Output (Controller with DT):**
```
[1   ] 192.168.1.3:50102 -> 192.168.1.205:21 | Dur=100761000 MaxIAT=100699000 Urg=0 FwdPkts=4 BwdPkts=0 FwdBytes=291 BwdBytes=0 Win=251 | Flags(S/A/F/R)=0/4/1/0 | PCA1=99 PCA2=26638 ... | Class=Benign(0)
```

**Expected Output (Controller with RF):**
```
[1   ] 192.168.1.3:50102 -> 192.168.1.205:21 | Dur=100761000 MaxIAT=100699000 Urg=0 FwdPkts=4 BwdPkts=0 FwdBytes=291 BwdBytes=0 Win=251 | Flags(S/A/F/R)=0/4/1/0 | PCA1=99 PCA2=26638 ... | Class=Benign(0)
  [Tree Votes: T0=Benign T1=Benign T2=DDoS T3=Benign T4=Benign ...]
```

**Expected Output (Controller with XGB):**
```
[1   ] 192.168.1.3:50102 -> 192.168.1.205:21 | Dur=100761000 MaxIAT=100699000 Urg=0 FwdPkts=4 BwdPkts=0 FwdBytes=291 BwdBytes=0 Win=251 | Flags(S/A/F/R)=0/4/1/0 | PCA1=99 PCA2=26638 ... | Class=Benign(0)
  [Scores: c0=185 c1=120 c2=95 c3=80]
```

### Custom PCA Components

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Extract features
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps

# Use 5 PCA components with 12-bit quantization
python3 2_pca_generating_entries.py --components 5 --bits 12

# Continue with any model type (example: RF)
python3 3_rf_training_model.py && \
python3 4_rf_generating_entries.py && \
python3 5_generating_p4_code.py --model-type rf && \
cd .. && make clean && make
```

---

## Recent Improvements

### TCP Flag Packet Counts as PCA Features
- **Expanded feature set from 9 to 13** by adding SYN/ACK/FIN/RST as per-packet **counts** (not booleans)
- `FlagsSyn/Ack/Fin/Rst` represent the number of packets in a flow that had each flag set
- **Universal formula:** both Python extraction (`state[...] += 1`) and P4 register accumulation (`flags_syn + (bit<32>)hdr.tcp.ctrl[1:1]`) use identical `+= 1` logic
- P4 registers upgraded from `register<bit<1>>` to `register<bit<32>>` for all four flag counters
- FIN/RST flow-termination detection uses `> 0` (count) instead of `== 1` (boolean)

### Robust Controller Startup
- **Model/digest mismatch detection:** at startup, `model.n_features_in_` is compared against the P4 digest's PCA component count; a clear one-time warning with remediation steps is printed if they differ
- Per-flow `RF/XGB-VERIFY=error(...)` spam is suppressed when a mismatch is already known at startup
- Controller now reads `build/basic.p4.p4info.txtpb` (always regenerated by `make`) instead of the potentially stale project-root `basic.p4info`

### MTU Handling for Large PCAPs
- `send_pcap.sh` and `test_with_pcap.sh` automatically raise the replay interface MTU to 9000 before replaying and restore it afterwards
- Fixes `errno=90 Message too long` errors with DDoS amplification PCAPs containing packets larger than 1500 bytes (tcpreplay 4.3.x does not support `--mtu-trunc`)

### Multi-Model Architecture
- **Three Classification Backends:** Decision Tree, Random Forest, XGBoost
- **Unified Pipeline:** Same feature extraction and PCA training for all models
- **Auto-Detection:** Controller automatically detects model type from P4Info digest schema
- **Model-Specific Outputs:** RF shows per-tree votes, XGB shows per-class scores

### Universal and Scalable Design
- **Dynamic Feature Count:** Digest parsing adapts to any number of PCA components
- **Schema-Driven:** Reads field layout from P4Info instead of hardcoded offsets
- **Extensible:** Add new features by updating extraction, PCA, and P4 generator — controller adapts automatically
- **No Feature Name Warnings:** Uses numpy arrays for predictions to avoid sklearn warnings

### Flow-Based Feature Extraction
- Bidirectional flow tracking with canonical 5-tuple (smaller endpoint always first)
- P4 registers maintain per-flow state (timestamps, byte/packet counts, flag counts) indexed by CRC hash
- **Benefit:** More accurate traffic classification using flow context rather than individual packets

### Multi-Format Capture Support
- Supports both `.pcap` and `.pcapng` file formats in both extraction and tcpreplay scripts
- **Benefit:** Works with any modern packet capture format without reconfiguration

---

## File Structure

```
p4-pca-dt/
├── README.md                            # This file
├── LICENSE
├── Makefile                             # P4 compilation
├── requirements.txt                     # Python dependencies
├── basic.p4                             # Generated P4 program (model-specific)
├── basic.p4info                         # P4 program metadata
├── control_plane/
│   ├── 1_data_extraction.py            # Feature extraction (PCAP/PCAPNG, live)
│   ├── 2_pca_generating_entries.py     # PCA training & encoding
│   ├── 3_dt_training_model.py          # Decision tree training
│   ├── 3_rf_training_model.py          # Random forest training
│   ├── 3_xgb_training_model.py         # XGBoost training
│   ├── 4_dt_generating_entries.py      # DT table generation
│   ├── 4_rf_generating_entries.py      # RF table generation
│   ├── 4_xgb_generating_entries.py     # XGB table generation
│   ├── 5_generating_p4_code.py         # P4 code generation (model-specific)
│   ├── 6_controller.py                 # Runtime controller (auto-detects model)
│   ├── pcaps/                          # PCAP/PCAPNG files
│   │   ├── AttackIDS/                 # AttackIDS dataset (Access, CC, Discovery, Evasion)
│   │   └── Mu-IoT/                    # Mu-IoT dataset (Benign, DDoS, PasswordHacking, Reconnaissance)
│   ├── dataset/                        # Extracted features
│   ├── model/                          # Trained models (dt.model, rf.model, xgb.model)
│   ├── tables/                         # Generated rules & configs
│   │   ├── s1-commands.txt            # P4 table entries (PCA + model)
│   │   ├── pca_encoding_params.json   # PCA metadata
│   │   ├── dt_params.json             # DT metadata (if using DT)
│   │   ├── rf_params.json             # RF metadata (if using RF)
│   │   └── xgb_params.json            # XGB metadata (if using XGB)
│   └── logs/                          # Runtime logs & predictions
└── build/                             # P4 compiler output
```

---

## Key Features

### Scalable PCA Components
- Supports any number of PCA components (auto-detect or manually specify)
- P4 code, digest structure, and controller adapt dynamically
- Default: Auto-detect for 95% variance explained

### Three ML Backends
1. **Decision Tree (DT):** Single table, fastest inference
2. **Random Forest (RF):** Multiple trees with majority voting
3. **XGBoost (XGB):** Gradient boosted trees with score accumulation

### Bidirectional Flow Tracking
- Normalizes flow direction (A→B and B→A map to same flow)
- Canonical 5-tuple: endpoint with smaller (IP, port) always first
- Matches offline extraction for consistent training/inference

### Model-Specific Runtime Verification
- **RF:** Shows which class each tree predicted for debugging
- **XGB:** Displays per-class accumulated scores from dataplane
- **DT:** Simple class output

### Universal Digest Parsing
- Reads digest schema from P4Info at runtime
- Adapts to any feature count, PCA components, or model outputs
- Fallback to heuristic parsing if schema unavailable
