# PCA with Decision Tree Traffic Classifier in P4 Switch

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

**Extracts:** IAT (Inter-Arrival Time), Packet Length, Diff Length

**Note:** The feature extractor automatically supports both `.pcap` and `.pcapng` file formats. Labels are auto-extracted from filenames (format: `<label>.v<version>.pcap` or `.pcapng`)

---

### Step 2: Train PCA and Generate Entries

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Auto-detect number of components (95% variance)
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv

# Or specify number of components
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv --n-components 2
```

**Output:** PCA parameters, integer codes, and P4 table commands

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

**Output:** `basic.p4` (auto-generated for any number of PCA components)

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
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv && \
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv && \
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

### Live Capture Pipeline

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Capture different traffic types
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label skype --output dataset/skype.csv
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label webex --output dataset/webex.csv
sudo python3 1_data_extraction.py --mode live --interface eth0 --count 1000 --label whatsapp --output dataset/whatsapp.csv

# Combine datasets
cat dataset/skype.csv dataset/webex.csv dataset/whatsapp.csv > dataset/combined.csv

# Train pipeline on combined data
python3 2_pca_generating_entries.py --dataset dataset/combined.csv && \
python3 3_dt_training_model.py && \
python3 4_dt_generating_entries.py && \
python3 5_generating_p4_code.py && \
cd .. && make clean && make
```

### Custom PCA Components

```bash
cd /tutorials/exercises/p4-pca-dt/control_plane

# Extract features
python3 1_data_extraction.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv

# Use 3 PCA components
python3 2_pca_generating_entries.py --dataset dataset/dataset.csv --n-components 3

# Continue with training
python3 3_dt_training_model.py && \
python3 4_dt_generating_entries.py && \
python3 5_generating_p4_code.py && \
cd .. && make clean && make
```

---

## Recent Improvements

### Flexible Label Support
- **Controller (6_controller.py)** now dynamically loads class labels from the trained model instead of hardcoding them
- The `load_class_labels()` function extracts labels from `model/dt.model` at runtime
- **Benefit:** No code modifications needed when changing dataset labels - any classification labels work automatically

### Multi-Format Capture Support
- **Feature Extractor (1_data_extraction.py)** now supports both `.pcap` and `.pcapng` file formats
- Automatically globs for both formats when processing directories
- **Benefit:** Works with any modern packet capture format without reconfiguration

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
