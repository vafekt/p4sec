# PCA with Decision Tree Traffic Classifier in P4 Switch

A complete pipeline for extracting **flow-based** network traffic features from PCAP files or live capture, training a decision tree classifier using PCA-reduced dimensions, and deploying classification rules in P4 behavioral model switches for real-time in-network inference.

**Flow-Based Architecture:** Tracks per-flow statistics (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol) using P4 registers to aggregate packets into flows and extract flow-level features for ML classification.

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
[3] Decision Tree Training (3_dt_training_model.py)
    ↓
Decision Tree Model (model/dt.model)
    ↓
[4] DT Table Generation (4_dt_generating_entries.py)
    ↓
DT Rules (tables/dt_commands.txt)
    ↓
[5] P4 Code Generation (5_generating_p4_code.py)
    ↓
P4 Program (basic.p4)
    ↓
[6] Compile and Deploy (make)
    ↓
[7] Runtime Controller (5_controller.py)
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

**Extracts 9 Flow-Based Features:**
- **IAT**: Sum of Inter-Arrival Times right-shifted by 2 (approximation for average, P4 cannot divide)
- **Duration**: Flow duration (time from first to last packet in nanoseconds)
- **SrcPort**: Source port number
- **DstPort**: Destination port number
- **TotalBytes**: Total bytes in flow (excluding Ethernet headers)
- **FlagsSyn**: SYN flag presence (0 or 1)
- **FlagsAck**: ACK flag presence (0 or 1)
- **FlagsFin**: FIN flag presence (0 or 1)
- **FlagsRst**: RST flag presence (0 or 1)

**Flow Aggregation:** Packets are grouped by 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol) to create per-flow statistics.

**Note:** The feature extractor automatically supports both `.pcap` and `.pcapng` file formats. Labels are auto-extracted from filenames (format: `<label>.v<version>.pcap` or `.pcapng`)

**Important:** IAT calculation uses `(sum_iat) >> 2` to match P4's implementation (P4 cannot perform runtime division, so right-shift approximates averaging).

---

### Step 2: Train PCA and Generate Entries

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Auto-detect number of components (95% variance)
python3 2_pca_generating_entries.py

# Or specify number of components and bit width
python3 2_pca_generating_entries.py --components 8 --bits 16
```

**Output:** 
- PCA parameters and encoding (tables/pca_encoding_params.json)
- Integer PCA codes mapping (tables/pca_integer_mapping.csv)
- P4 table commands for PCA transformation (tables/s1-commands.txt)
- DecisionTreeRegressor that maps raw features → PCA codes (used for P4 rule generation)

**Note:** The PCA transformation is approximated using a DecisionTreeRegressor with max_depth=12 to enable P4 table-based inference. This creates ~73 leaf nodes mapping feature ranges to quantized PCA codes.

---

### Step 3: Train Decision Tree

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

python3 3_dt_training_model.py
```

**Output:** Trained model and classification metrics

---

### Step 4: Generate Decision Tree P4 Entries

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

python3 4_dt_generating_entries.py
```

**Output:** P4 table entries for classification

---

### Step 5: Generate P4 Program

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

python3 5_generating_p4_code.py
```

**Output:** `../basic.p4` (auto-generated for any number of PCA components)

**P4 Architecture:**
- **Flow Tracking:** Uses P4 registers to maintain per-flow state (timestamps, byte counts, flags)
- **Hash-Based Indexing:** CRC16/CRC32 hash of 5-tuple for flow identification
- **Feature Extraction:** Calculates IAT, Duration, TotalBytes, Ports, Flags from register state
- **PCA Tables:** 8 tables (pca_component1-8) map flow features to quantized PCA codes using range matching
- **ML Classification:** ml_code table uses PCA codes to determine traffic class
- **Digest Output:** Sends flow features + PCA codes + classification to controller

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
python3 6_controller.py
```

**Note:** The controller automatically loads class labels from the trained model (`model/dt.model`), making it flexible to any dataset labels without code modifications.

---

## Complete Pipeline Examples

### Quick Start

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Full pipeline in one sequence
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps && \
python3 2_pca_generating_entries.py && \
python3 3_dt_training_model.py && \
python3 4_dt_generating_entries.py && \
python3 5_generating_p4_code.py && \
cd .. && make clean && make

# Terminal 1: Start Mininet
sudo make run

# Terminal 2: Start controller (new terminal)
cd /tutorials/exercises/p4-pca-dt/control_plane
python3 6_controller.py
```

**Expected Output (Controller):**
```
[1   ]     172.16.66.1:51954 ->    172.16.66.36:445   | IAT=58    Dur=232   Bytes=374  | Flags(S/A/F/R)=0/1/0/0 | PCA1=9678  PCA2=37800 ... | Class=CC(1)
[2   ]    172.16.66.36:445   ->     172.16.66.1:51954 | IAT=302   Dur=1211  Bytes=502  | Flags(S/A/F/R)=0/1/0/0 | PCA1=54    PCA2=43184 ... | Class=Access(0)
```

### Custom PCA Components

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Extract features
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps

# Use 3 PCA components with 8-bit quantization
python3 2_pca_generating_entries.py --components 3 --bits 8

# Continue with training
python3 3_dt_training_model.py && \
python3 4_dt_generating_entries.py && \
python3 5_generating_p4_code.py && \
cd .. && make clean && make
```

---

## Recent Improvements

### Flow-Based Feature Extraction (Major Update)
- **Converted from packet-level to flow-level features** for better traffic characterization
- Tracks 9 flow-based features: IAT, Duration, SrcPort, DstPort, TotalBytes, TCP Flags (Syn/Ack/Fin/Rst)
- **P4 Flow State Management:** Uses registers to maintain per-flow statistics indexed by flow hash
- **IAT Approximation:** Uses `sum_iat >> 2` instead of division (P4 limitation) - training data matches this calculation
- **Benefit:** More accurate traffic classification using flow context rather than individual packets

### Range Matching for All Features
- **Flag Fields:** Use range matching (0->0, 1->1, 0->1 for wildcards) instead of exact matching
- **Benefit:** Decision tree can use wildcards in rules, improving coverage and reducing table entries

### Flexible Label Support
- **Controller (6_controller.py)** now dynamically loads class labels from the trained model instead of hardcoding them
- The `load_class_labels()` function extracts labels from `model/dt.model` at runtime
- **Benefit:** No code modifications needed when changing dataset labels - any classification labels work automatically

### Multi-Format Capture Support
- **Feature Extractor (1_data_extraction.py)** now supports both `.pcap` and `.pcapng` file formats
- Automatically globs for both formats when processing directories
- **Benefit:** Works with any modern packet capture format without reconfiguration

### Improved PCA Tree Resolution
- DecisionTreeRegressor now uses max_depth=12, min_samples_split=2, min_samples_leaf=1
- **Benefit:** Better PCA code approximation with more granular feature space partitioning

---

## File Structure

```
p4-pca-dt/
├── README.md                           # This file
├── LICENSE
├── Makefile                            # P4 compilation
├── requirements.txt                    # Python dependencies
├── basic.p4                            # Generated P4 program
├── basic.p4info                        # P4 program metadata
├── control_plane/
│   ├── 1_data_extraction.py           # Feature extraction (PCAP/PCAPNG, live capture)
│   ├── 2_pca_generating_entries.py    # PCA training & encoding
│   ├── 3_dt_training_model.py         # Decision tree training
│   ├── 4_dt_generating_entries.py     # DT table generation
│   ├── 5_generating_p4_code.py        # P4 code generation
│   ├── 6_controller.py                # Runtime controller
│   ├── pcaps/                         # PCAP/PCAPNG files
│   ├── dataset/                       # Extracted features
│   ├── model/                         # Trained models
│   ├── tables/                        # Generated rules & configs
│   └── logs/                          # Runtime logs & predictions
└── build/                             # P4 compiler output
```
