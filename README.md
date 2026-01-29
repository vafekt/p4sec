# P4 Decision Tree Classifier for Network Traffic Classification

A complete pipeline for extracting network traffic features from PCAP files or live capture, training a decision tree classifier using PCA-reduced dimensions, and deploying classification rules in P4 behavioral model switches for real-time in-network inference.

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
PCA Rules (tables/s1-commands.txt - PCA section)
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
[6] Runtime Controller (5_controller.py)
    ↓
Live Classification & Monitoring
```

---

## Prerequisites

### Repository Location

**IMPORTANT**: This project must be placed in the `/tutorials/exercises/` directory.

```bash
# Clone or move the repository to the correct location
cd /tutorials/exercises/
git clone <repository-url> p4-pca-dt
# OR move existing directory:
mv /path/to/p4-pca-dt /tutorials/exercises/p4-pca-dt

# Verify location
pwd
# Should output: /tutorials/exercises/p4-pca-dt
```

### P4 Development Environment

1. **P4 Behavioral Model (BMv2)** - Software switch implementation
   ```bash
   # Install from p4lang/behavioral-model
   git clone https://github.com/p4lang/behavioral-model.git
   cd behavioral-model
   ./autogen.sh
   ./configure
   make
   sudo make install
   ```

2. **P4C Compiler** - P4_16 compiler
   ```bash
   # Install from p4lang/p4c
   git clone --recursive https://github.com/p4lang/p4c.git
   cd p4c
   mkdir build
   cd build
   cmake ..
   make -j4
   sudo make install
   ```

3. **Mininet** - Network emulator
   ```bash
   sudo apt-get install mininet
   ```

4. **P4 Utilities**
   ```bash
   git clone https://github.com/p4lang/tutorials.git
   # Contains helper scripts and tools
   ```

### Python Environment

1. **Activate Virtual Environment** (if using p4dev):
   ```bash
   source ~/Desktop/src/p4dev-python-venv/bin/activate
   ```

2. **Required Python Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **System Dependencies**:
   ```bash
   # For pyshark (packet capture)
   sudo apt-get install tshark wireshark
   
   # For live capture (needs root privileges)
   sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)
   ```

### PCAP Files

- Place `.pcap` files in `control_plane/pcaps/` folder
- Filename format: `<label>.v<version>.pcap` (e.g., `skype.v1.pcap`, `webex.v2.pcap`)
- Labels are auto-extracted from filenames (before `.v` suffix)

---

## Step-by-Step Execution

### Step 1: Extract Features from PCAP Files or Live Capture

**File:** `control_plane/1_data_extraction.py`

**Purpose:** Extracts 3 lightweight features from PCAP files or live network capture:
- **IAT (Inter-Arrival Time)**: Time difference between consecutive packets (nanoseconds)
- **Packet Length**: Size of current packet
- **Diff Length**: Difference between current and previous packet length

**Mode 1: Process PCAP Files**

```bash
cd control_plane

# Process all .pcap files in pcaps/ folder
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# Process single PCAP file
python3 1_data_extraction.py --mode pcap --input pcaps/skype.v1.pcap --output dataset/single.csv
```

**Mode 2: Live Network Capture**

```bash
cd control_plane

# Capture 1000 packets from eth0 interface
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label skype --output dataset/live_capture.csv

# Capture for 60 seconds
sudo python3 1_data_extraction.py --mode live --interface wlan0 --duration 60 --label webex --output dataset/timed_capture.csv

# Available interfaces: eth0, wlan0, lo, etc. (check with: ip link show)
```

**Output:**
- `dataset/dataset.csv` - Features with columns: IAT, PacketLength, DiffLength, Label

**Key Parameters:**
- `--mode`: `pcap` (file processing) or `live` (interface capture)
- `--interface`: Network interface for live capture (requires sudo)
- `--count`: Number of packets to capture (default: 1000)
- `--duration`: Optional time limit in seconds
- `--label`: Label for captured packets in live mode
- `--output`: Output CSV file path

---

### Step 2: PCA Training & Quantization

**File:** `control_plane/2_pca_generating_entries.py`

**Purpose:** 
- Trains PCA to reduce 3 raw features → N principal components (auto-selected or specified)
- Quantizes PCA values to 16-bit integers (0–65535) for P4 table matching
- Generates P4 table entries for PCA transformation

**How to run:**

```bash
cd control_plane

# Auto-select number of components (95% variance)
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv

# Specify number of components
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv --n-components 2

# Custom output paths
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv --output-commands tables/custom_commands.txt
```

**Output Files:**
- `tables/pca_encoding_params.json` - PCA parameters (min, max, range, n_components)
- `tables/pca_integer_mapping.csv` - Mapping of raw features → PCA codes
- `tables/pca_metrics.json` - Explained variance and metrics
- `tables/s1-commands.txt` - P4 table_add commands for PCA transformation tables

**What it does:**
1. Loads feature dataset
2. Trains PCA with variance threshold (default 95%)
3. Quantizes PCA output to 16-bit integers
4. Trains decision tree to map raw features → PCA codes
5. Generates P4 range-based table entries

**Key Configuration:**
- Default: Auto-select components for 95% variance
- Quantization: 16-bit (0–65535)
- P4 tables generated: `pca_component1`, `pca_component2`, ..., `pca_componentN`

---

### Step 3: Decision Tree Training

**File:** `control_plane/3_dt_training_model.py`

**Purpose:** Trains DecisionTreeClassifier on PCA-quantized codes to predict traffic class labels.

**How to run:**

```bash
cd control_plane

# Train with default settings
python3 3_dt_training_model.py

# Specify custom paths
python3 3_dt_training_model.py --dataset tables/pca_integer_mapping.csv --output model/dt.model
```

**Input:** 
- `tables/pca_integer_mapping.csv` (from Step 2)

**Output:**
- `model/dt.model` - Trained DecisionTreeClassifier (pickled)
- `tables/dt_metrics.json` - Accuracy, precision, recall, F1-score

**What it does:**
1. Reads PCA integer mapping CSV
2. Extracts PCA component codes as features
3. Extracts labels as target classes
4. Trains DecisionTreeClassifier
5. Evaluates on test set
6. Saves model and metrics

**Sample Output:**
```
Training Decision Tree Classifier...
Accuracy: 0.9876
Precision: 0.9832
Recall: 0.9845
F1-Score: 0.9838
Model saved to model/dt.model
```

---

### Step 4: Generate Decision Tree P4 Table Entries

**File:** `control_plane/4_dt_generating_entries.py`

**Purpose:** Converts trained decision tree into P4 table entries for the `ml_code` table.

**How to run:**

```bash
cd control_plane

# Generate with default paths
python3 4_dt_generating_entries.py

# Specify custom model and output
python3 4_dt_generating_entries.py --model model/dt.model --output tables/dt_commands.txt
```

**Input:**
- `model/dt.model` (from Step 3)
- `tables/pca_encoding_params.json` (from Step 2)

**Output:**
- `tables/dt_commands.txt` - P4 table_add commands for decision tree
- `tables/dt_tree.txt` - Human-readable tree structure
- `tables/dt_if_rules.txt` - IF-THEN rule representation

**What it does:**
1. Loads trained decision tree
2. Extracts decision paths (feature ranges → class)
3. Converts to P4 range-based table entries
4. Maps class labels → numeric codes

**Sample Output:**
```
=== Label Encoding ===
  skype -> 0
  webex -> 1
  whatsapp -> 2

table_add MyIngress.ml_code set_result 0->15000 20000->40000 => 0 1
table_add MyIngress.ml_code set_result 15001->25000 40001->65535 => 1 2
...
```

---

### Step 5: Generate P4 Program Code

**File:** `control_plane/5_generating_p4_code.py`

**Purpose:** Automatically generates `basic.p4` with support for N PCA components (scalable).

**How to run:**

```bash
cd control_plane

# Auto-detect number of components from pca_encoding_params.json
python3 5_generating_p4_code.py

# Custom output path
python3 5_generating_p4_code.py --output ../custom.p4
```

**Input:**
- `tables/pca_encoding_params.json` (auto-detects n_components)
- `tables/s1-commands.txt` (alternative detection method)

**Output:**
- `../basic.p4` - Generated P4 program with dynamic PCA tables

**What it does:**
1. Detects number of PCA components
2. Generates metadata fields (`pc1_code`, `pc2_code`, ..., `pcN_code`)
3. Generates PCA transformation tables (`pca_component1`, ..., `pca_componentN`)
4. Generates ML classification table with all PCA codes as keys
5. Generates complete P4 program

**Key Features:**
- **Scalable**: Automatically adapts to any number of PCA components
- **Dynamic**: No manual P4 code editing required
- **Consistent**: Ensures table names match command file references

---

### Step 6: Compile and Run P4 Program

**Compile P4 Code:**

```bash
cd /tutorials/exercises/p4-pca-dt

# Clean previous build
make clean

# Compile basic.p4
make

# Or manually:
p4c-bm2-ss --p4v 16 --p4runtime-files basic.p4info --o basic.json basic.p4
```

**Run Mininet (Terminal 1):**

```bash
# IMPORTANT: Start Mininet FIRST before running the controller
cd /tutorials/exercises/p4-pca-dt

# Clean and start Mininet with P4 topology
make clean
sudo make run

# Mininet will start and show the CLI prompt: mininet>
# Keep this terminal running
```

---

### Step 7: Runtime Controller (Terminal 2)

**File:** `control_plane/5_controller.py`

**Purpose:** Real-time monitoring and control of P4 switch via P4Runtime API.

**How to run:**

**IMPORTANT**: Run the controller in a **separate terminal** AFTER Mininet is running.

```bash
# Open a NEW terminal (Terminal 2)
cd /tutorials/exercises/p4-pca-dt/control_plane

# Start controller (connects to running Mininet switch)
python3 5_controller.py

# With custom gRPC address
python3 5_controller.py --grpc-addr localhost:9559
```

**Execution Order:**
1. **Terminal 1**: Start Mininet (`sudo make run`) ← FIRST
2. **Terminal 2**: Start controller (`python3 5_controller.py`) ← SECOND

**Features:**
- Loads table entries via P4Runtime
- Monitors packet digests (classification results)
- Logs predictions to `logs/predictions.csv`
- Real-time traffic classification monitoring

---

## Complete Pipeline Examples

### Example 1: Quick Start with Sample PCAP Files

```bash
# Navigate to project directory (must be in /tutorials/exercises/)
cd /tutorials/exercises/p4-pca-dt/control_plane

# Step 1: Extract features from all PCAP files
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# Step 2: Train PCA and generate entries (auto-select components)
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv

# Step 3: Train decision tree
python3 3_dt_training_model.py

# Step 4: Generate DT table entries
python3 4_dt_generating_entries.py

# Step 5: Generate P4 code (auto-detect components)
python3 5_generating_p4_code.py

# Step 6: Compile P4 program
cd ..
make clean
make

# Step 7: Run Mininet (Terminal 1)
sudo make run
# Keep this terminal running with Mininet

# Step 8: In a NEW terminal (Terminal 2), run controller
# Open new terminal, then:
cd /tutorials/exercises/p4-pca-dt/control_plane
python3 5_controller.py
```

### Example 2: Live Network Capture

```bash
cd control_plane

# Capture Skype traffic (1000 packets)
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label skype --output dataset/skype_live.csv

# Capture Webex traffic (60 seconds)
sudo python3 1_data_extraction.py --mode live --interface eth0 --duration 60 --label webex --output dataset/webex_live.csv

# Capture WhatsApp traffic
sudo python3 1_data_extraction.py --mode live --interface wlan0 --count 500 --label whatsapp --output dataset/whatsapp_live.csv

# Combine captures into one dataset
cat dataset/skype_live.csv dataset/webex_live.csv dataset/whatsapp_live.csv > dataset/combined.csv

# Continue with PCA training...
python3 2_pca_generating_entries.py --dataset dataset/combined.csv
```

### Example 3: Custom Number of PCA Components

```bash
cd control_plane

# Step 1: Extract features
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# Step 2: Use 3 PCA components instead of auto-detect
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv --n-components 3

# Step 3: Train DT
python3 3_dt_training_model.py

# Step 4: Generate DT entries
python3 4_dt_generating_entries.py

# Step 5: Generate P4 code (auto-detected from params file)
python3 5_generating_p4_code.py

# Verify the generated P4 code has 3 pca_component tables
grep "pca_component" ../basic.p4
```

### Manual Sequential Execution

```bash
# Ensure you're in the correct directory
cd /tutorials/exercises/p4-pca-dt/control_plane

# Extract → PCA → DT → Generate → Compile
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv && \
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv && \
python3 3_dt_training_model.py && \
python3 4_dt_generating_entries.py && \
python3 5_generating_p4_code.py && \
cd .. && make clean && make

# Check outputs
ls -lh control_plane/dataset/
ls -lh control_plane/tables/
ls -lh control_plane/model/
ls -lh basic.p4 basic.json

# Terminal 1: Start Mininet
sudo make run

# Terminal 2 (new terminal): Start controller
cd /tutorials/exercises/p4-pca-dt/control_plane
python3 5_controller.py
```

---

## Project Structure

```
p4-pca-dt/
├── control_plane/                    # Main pipeline directory
│   ├── 1_data_extraction.py          # Feature extraction (PCAP/live)
│   ├── 2_pca_generating_entries.py   # PCA training & P4 entry generation
│   ├── 3_dt_training_model.py        # Decision tree training
│   ├── 4_dt_generating_entries.py    # DT P4 entry generation
│   ├── 5_generating_p4_code.py       # P4 code generator (scalable)
│   ├── 5_controller.py               # P4Runtime controller
│   │
│   ├── dataset/                      # Feature datasets
│   │   └── dataset.csv               # (Step 1 output)
│   │
│   ├── model/                        # Trained models
│   │   └── dt.model                  # (Step 3 output)
│   │
│   ├── tables/                       # P4 table entries and configs
│   │   ├── s1-commands.txt           # (Step 2 output) PCA tables
│   │   ├── dt_commands.txt           # (Step 4 output) DT table
│   │   ├── pca_encoding_params.json  # (Step 2) PCA parameters
│   │   ├── pca_integer_mapping.csv   # (Step 2) Feature→PCA codes
│   │   ├── pca_metrics.json          # (Step 2) PCA metrics
│   │   ├── dt_metrics.json           # (Step 3) DT metrics
│   │   ├── dt_tree.txt               # (Step 4) Tree visualization
│   │   └── dt_if_rules.txt           # (Step 4) IF-THEN rules
│   │
│   ├── pcaps/                        # Input PCAP files
│   │   ├── skype.v1.pcap
│   │   ├── skype.v2.pcap
│   │   ├── webex.v1.pcap
│   │   ├── whatsapp.v1.pcap
│   │   └── ...
│   │
│   ├── logs/                         # Runtime logs
│   │   └── predictions.csv           # (Controller output)
│   │
│   └── __pycache__/
│
├── p4/                               # P4Runtime gRPC bindings
│   ├── v1/                           # P4Runtime v1
│   ├── config/                       # P4 config proto
│   ├── server/                       # Server proto
│   ├── bm/                           # Behavioral model
│   └── tmp/
│
├── basic.p4                          # Generated P4 program (Step 5)
├── basic.p4info                      # P4Info file (from compilation)
├── basic.json                        # BMv2 JSON (compiled P4)
├── Makefile                          # Build automation
├── README.md                         # This file
├── topology.json                     # Mininet topology
├── s1-runtime.json                   # Runtime configuration
└── complete_pipeline.sh              # Automated pipeline script
```

---

## File Descriptions

### Input Files
- **`control_plane/pcaps/*.pcap`**: Network traffic captures with labeled filenames
- **`topology.json`**: Mininet network topology (switches, hosts, links)

### Generated Files (Pipeline Outputs)
| File | Generator | Purpose |
|------|-----------|---------|
| `dataset/dataset.csv` | Step 1 | Raw features (IAT, PacketLength, DiffLength) |
| `tables/pca_encoding_params.json` | Step 2 | PCA parameters for quantization |
| `tables/pca_integer_mapping.csv` | Step 2 | Training data with PCA codes |
| `tables/pca_metrics.json` | Step 2 | PCA explained variance |
| `tables/s1-commands.txt` | Step 2 | P4 table entries for PCA |
| `model/dt.model` | Step 3 | Trained decision tree (pickle) |
| `tables/dt_metrics.json` | Step 3 | Classification metrics |
| `tables/dt_commands.txt` | Step 4 | P4 table entries for DT |
| `tables/dt_tree.txt` | Step 4 | Tree visualization |
| `basic.p4` | Step 5 | P4 program (auto-generated) |
| `basic.json` | Compilation | BMv2 executable |
| `basic.p4info` | Compilation | P4Runtime schema |

### Configuration Files
- **`Makefile`**: Build rules for P4 compilation and Mininet launch
- **`s1-runtime.json`**: Switch runtime config (forwarding rules)
- **`topology.json`**: Network topology definition

---

## Key Concepts

### Feature Extraction
- **IAT (Inter-Arrival Time)**: Time difference between consecutive packets in nanoseconds
- **Packet Length**: Total packet size including Ethernet header
- **Diff Length**: Difference from previous packet length + 65535 (offset to avoid negatives)

### PCA (Principal Component Analysis)
- **Purpose**: Dimensionality reduction of raw features
- **Input**: 3 raw features (IAT, PacketLength, DiffLength)
- **Output**: N principal components (typically 2-5, auto-selected for 95% variance)
- **Quantization**: Float PCA values → 16-bit integers (0–65535) for P4 table matching
- **P4 Tables**: `pca_component1`, `pca_component2`, ..., `pca_componentN`
  - **Keys**: 3 raw features (range matching)
  - **Action**: Set PCA code for that component

### Decision Tree Classification
- **Input**: N PCA integer codes (pc1_code, pc2_code, ...)
- **Output**: Traffic class label (0=skype, 1=webex, 2=whatsapp, etc.)
- **P4 Table**: `ml_code`
  - **Keys**: All PCA codes (range matching)
  - **Action**: `set_result` with class label
- **Logic**: Multi-dimensional range partitioning

### P4 Range Matching
- **Format**: `table_add <table> <action> <range1> <range2> ... => <param> <priority>`
- **Range**: `min->max` (inclusive on both ends)
- **Example**: `0->15665196544` matches IAT from 0 to 15665196544 ns
- **Priority**: Lower numbers = higher priority (evaluated first)

### Label Encoding
- Traffic class names (strings) → Numeric codes (integers)
- Encoding is alphabetical by default
- Example mapping:
  ```
  skype     → 0
  webex     → 1
  whatsapp  → 2
  ```
- Stored in P4 metadata field: `inference_result_t ml_result` (8-bit)

### Scalability Features
- **Auto-detection**: Pipeline automatically detects PCA components
- **Dynamic P4 generation**: Adapts to any number of components
- **Live capture**: Can train on real-time traffic
- **Configurable**: Adjust variance threshold, tree depth, etc.

---

## P4 Program Architecture

### Data Plane Flow

```
Packet Arrives
    ↓
[Parser] Extract headers (Ethernet, IPv4, TCP/UDP)
    ↓
[Ingress Processing]
    │
    ├─→ Extract raw features:
    │   ├─ get_iat()       → meta.iat (from timestamp registers)
    │   ├─ get_pkt_len()   → meta.pkt_len (packet length)
    │   └─ get_diff_len()  → meta.diffLen (from length registers)
    │
    ├─→ PCA Transformation (range-based lookup):
    │   ├─ pca_component1.apply() → meta.pc1_code
    │   ├─ pca_component2.apply() → meta.pc2_code
    │   └─ ... (for N components)
    │
    ├─→ Classification:
    │   └─ ml_code.apply() → meta.ml_result (class label)
    │
    ├─→ Send digest to controller (optional monitoring)
    │
    └─→ Forward packet: ipv4_lpm.apply()
    ↓
[Egress Processing] (pass-through)
    ↓
[Deparser] Emit headers
    ↓
Packet Forwarded
```

### Register State
- **`last_ts_reg`**: Stores last packet timestamp (for IAT calculation)
- **`last_len_reg`**: Stores last packet length (for DiffLength calculation)
- Each register holds 1 element (per-switch state, not per-flow)

### Tables
1. **`ipv4_lpm`**: Standard IPv4 forwarding (LPM on destination IP)
2. **`pca_component1` ... `pca_componentN`**: PCA transformation
   - Match: (iat, pkt_len, diffLen) as ranges
   - Action: Set pcN_code
3. **`ml_code`**: Decision tree classification
   - Match: (pc1_code, pc2_code, ..., pcN_code) as ranges
   - Action: Set ml_result

### Metadata Fields
```p4
struct metadata {
    bit<16> srcPort;
    bit<16> dstPort;
    
    // Raw features
    feature1_t iat;       // 64-bit
    feature2_t pkt_len;   // 16-bit
    feature3_t diffLen;   // 32-bit
    
    // PCA codes (dynamically generated)
    pca_code_t pc1_code;  // 16-bit
    pca_code_t pc2_code;  // 16-bit
    // ... up to pcN_code
    
    // Result
    inference_result_t ml_result;  // 8-bit
}
```

---

## Advanced Usage

### Custom Feature Engineering

Edit `1_data_extraction.py` to add new features:

```python
# In extract_features_from_pcap() function
feature = {
    "IAT": iat,
    "PacketLength": ip_len,
    "DiffLength": diff_len,
    "CustomFeature": calculate_custom_feature(packet),  # Add here
}
```

Then update P4 program generator to include new feature in match keys.

### Multiple Traffic Classes

The pipeline supports any number of classes:
- Add PCAP files with different labels: `class1.v1.pcap`, `class2.v1.pcap`, etc.
- Pipeline automatically detects all unique labels
- Generates appropriate classifier with N classes

### Fine-Tuning Decision Tree

Edit `3_dt_training_model.py`:

```python
dt_classifier = DecisionTreeClassifier(
    max_depth=15,           # Limit tree depth
    min_samples_split=10,   # Minimum samples to split node
    min_samples_leaf=5,     # Minimum samples in leaf
    random_state=42
)
```

### Custom P4 Templates

Modify `5_generating_p4_code.py` to customize:
- Header definitions
- Parser logic
- Additional tables
- Custom actions

---

## Example Datasets

### Included PCAP Files
- **Skype**: VoIP traffic (multiple versions)
- **Webex**: Video conferencing traffic
- **WhatsApp**: Messaging and VoIP traffic

### Collecting Your Own Data

```bash
# Capture during specific application usage
sudo tcpdump -i eth0 -w control_plane/pcaps/myapp.v1.pcap

# Filter by port
sudo tcpdump -i eth0 port 443 -w control_plane/pcaps/https.v1.pcap

# Capture for specific duration
timeout 60 sudo tcpdump -i eth0 -w control_plane/pcaps/sample.v1.pcap
```

---

## Dependencies & Library Versions

### Required Libraries

**Python Packages:**
```bash
pip install -r requirements.txt
```

**System Packages:**
```bash
# Packet capture tools
sudo apt-get install tshark wireshark tcpdump

# Build tools
sudo apt-get install build-essential cmake

# Python development
sudo apt-get install python3-dev python3-pip
```

**P4 Tools:**
```bash
# P4 compiler (p4c)
# Install from: https://github.com/p4lang/p4c

# Behavioral Model v2 (BMv2)
# Install from: https://github.com/p4lang/behavioral-model

# Mininet
sudo apt-get install mininet

# P4 utilities
git clone https://github.com/p4lang/tutorials.git
```

### Verify Installation

```bash
# Python packages
python3 -c "import numpy, pandas, sklearn, scipy, pyshark, scapy; print('All packages OK')"

# P4 tools
p4c --version
simple_switch --version
mn --version

# Network tools
tshark --version
tcpdump --version
```

### Tested Versions

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.8+ | Required for type hints |
| numpy | 1.21+ | Array operations |
| pandas | 1.3+ | DataFrame support |
| scikit-learn | 1.0+ | PCA and DecisionTree |
| pyshark | 0.4.5+ | PCAP parsing |
| p4c | 1.2.0+ | P4_16 compiler |
| BMv2 | 1.15+ | Behavioral model |
| Mininet | 2.3.0+ | Network emulation |

---

## Common Commands Reference

### Data Collection

```bash
# List network interfaces
ip link show

# Capture live traffic
sudo tcpdump -i eth0 -w output.pcap

# View PCAP info
tcpdump -r file.pcap -n | head

# Extract with tshark
tshark -r file.pcap -T fields -e ip.src -e ip.dst
```

### Pipeline Execution

```bash
# Extract features from PCAP
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# Live capture
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label myapp

# Train PCA (auto-detect components)
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv

# Train PCA (manual components)
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv --n-components 2

# Train decision tree
python3 3_dt_training_model.py

# Generate DT table entries
python3 4_dt_generating_entries.py

# Generate P4 code (auto-detected)
python3 5_generating_p4_code.py

# Custom output path
python3 5_generating_p4_code.py --output ../basic.p4
```

### P4 Compilation & Execution

```bash
# Ensure correct directory
cd /tutorials/exercises/p4-pca-dt

# Clean build
make clean

# Compile P4
make

# Terminal 1: Run Mininet (start FIRST)
sudo make run

# Terminal 2: Run controller (start AFTER Mininet is running)
cd /tutorials/exercises/p4-pca-dt/control_plane
python3 5_controller.py

# Alternative: Load table entries manually via CLI
simple_switch_CLI < control_plane/tables/s1-commands.txt

# View P4 tables
echo "table_dump MyIngress.pca_component1" | simple_switch_CLI

# Check counters
echo "counter_read MyIngress.packet_counter 0" | simple_switch_CLI
```

### Debugging

```bash
# Check generated files
ls -lh control_plane/dataset/
ls -lh control_plane/tables/
ls -lh control_plane/model/

# View metrics
cat control_plane/tables/pca_metrics.json | python3 -m json.tool
cat control_plane/tables/dt_metrics.json | python3 -m json.tool

# Count table entries
wc -l control_plane/tables/s1-commands.txt
wc -l control_plane/tables/dt_commands.txt

# Verify PCA components
grep "n_components" control_plane/tables/pca_encoding_params.json

# Check P4 tables
grep "table " basic.p4
```

---


