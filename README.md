# p4sec — In-Network IoT Attack Detection on P4 Programmable Data Planes

Reference implementation accompanying the paper
**"In-Network IoT Attack Detection Using Principal Component Analysis and Machine Learning Models on P4 Programmable Data Planes."**

The pipeline trains a Decision Tree (DT) or Random Forest (RF) classifier on
bidirectional flow features and compiles it into range-match P4 tables that run
end-to-end on the data plane. Dimensionality reduction is applied as a single
range-match preprocessing stage; PCA is the primary reduction method evaluated
in the paper, with LDA and Autoencoder also supported as alternatives.

The system classifies four CIC-IoT 2023 categories: **Benign**, **DoS**,
**Brute Force**, and **Reconnaissance**.

---

## Pipeline Overview

```
Raw PCAP files (control_plane/pcaps/)
    │
    ▼
[1] Feature Extraction        →  1_extract_dataset.py
    │   (canonical 5-tuple, register-based per-flow accumulation,
    │    20 bidirectional flow features)
    ▼
dataset/dataset.csv
    │
    ▼
[2] Dimensionality Reduction  →  2_pca_generate_entries.py   (primary)
    │                            2_lda_generate_entries.py
    │                            2_autoencoder_generate_entries.py
    │   (K-component projection, B-bit quantization, range-match
    │    surrogate DTR per component)
    ▼
tables/reduction_config.json + tables/s1-commands.txt (transform entries)
tables/transform_mapping.csv  (K quantized codes + Label)
    │
    ▼
[3] Classifier Training       →  3_train_model.py  -m {dt|rf}
    │
    ▼
model/{dt|rf}.model + tables/model_metrics.json
    │
    ▼
[4] Classifier Table Generation → 4_generate_model_entries.py  -m {dt|rf}
    │   (range-match entries: ml_code for DT, rf_tree_i + rf_vote_classify for RF)
    ▼
tables/s1-commands.txt (appended with classifier entries)
    │
    ▼
[5] P4 Code Generation        →  5_generating_p4_code.py  -m {dt|rf}
    │   (emits both BMv2 basic.p4 and Tofino p4sec_tofino.p4)
    ▼
../basic.p4   +   ../p4sec_tofino.p4
    │
    ▼
[6] Compile & deploy on BMv2  →  make clean && make run
    │
    ▼
[7] Runtime controller         →  6_controller.py
        (loads s1-commands.txt, listens for digests, logs predictions)
```

---

## Prerequisites

The project expects the standard `p4lang/tutorials` layout:

```bash
cd /tutorials/exercises/
git clone https://github.com/vafekt/p4sec.git
cd p4sec
```

Install requirements:

```bash
pip install -r requirements.txt
sudo apt-get install tshark wireshark tcpdump
```

P4 toolchain (BMv2, p4c, Mininet) — follow https://github.com/p4lang/tutorials.

---

## Step 1 — Feature Extraction

```bash
cd control_plane
python3 1_extract_dataset.py --mode pcap --pcap-dir pcaps --output dataset/dataset.csv
```

The 20 bidirectional flow features extracted (paper Table 2):

| Category | Feature | Bits |
|---|---|---|
| Protocol | Protocol | 8 |
| Ports | SrcPort, DstPort | 16, 16 |
| Timing | Duration, MaxIAT, MinIAT | 48, 48, 48 |
| Volume | FwdPktCount, BwdPktCount, FwdBytes, BwdBytes, FwdMaxPktLen, BwdMaxPktLen | 32×4, 16×2 |
| TCP Flags | FlagsSyn, FlagsAck, FlagsFin, FlagsRst, FlagsPsh, UrgCount | 32×6 |
| Window | MaxWinSize, InitFwdWinBytes | 16, 16 |

Per-flow state is held in CRC16-indexed register arrays; a CRC32-indexed Bloom
filter detects index collisions. A flow is finalized on TCP FIN/RST or after a
20-second idle timeout, after which it proceeds to classification.

---

## Step 2 — Dimensionality Reduction

Choose one of the three methods. The paper's headline results use **PCA**.

```bash
# PCA (primary — paper Section 4)
python3 2_pca_generate_entries.py --components 7 --bits 32

# LDA
python3 2_lda_generate_entries.py --components 3 --bits 16

# Autoencoder
python3 2_autoencoder_generate_entries.py --components 7 --bits 16
```

Each method:
- Fits the projection on the raw integer features.
- Quantises every component score to B bits.
- Trains a Decision Tree Regressor (DTR) per component to approximate the
  projection. Each leaf stores the mean quantized code over its training
  samples.
- Emits the DTR as range-match P4 entries (one entry per leaf per component).
- Writes `tables/reduction_config.json` so steps 3 / 4 / 5 detect the method
  automatically.

The paper evaluates **24 PCA configurations**: k ∈ {5,6,7,8,9,10} components,
b ∈ {16, 32} bits, classifier ∈ {DT, RF}.

---

## Step 3 — Classifier Training

```bash
python3 3_train_model.py -m dt          # Decision Tree
python3 3_train_model.py -m rf -n 4     # Random Forest, 4 trees
```

Trains on the K quantized codes produced in step 2. Reports 80/20 holdout
metrics, then retrains on 100 % of data for deployment.

Output:
- `model/{dt|rf}.model`
- `tables/model_metrics.json`
- `tables/model_params.json` (RF only — vote-packing parameters)

---

## Step 4 — Classifier Table Generation

```bash
python3 4_generate_model_entries.py -m dt
python3 4_generate_model_entries.py -m rf
```

DT: the tree is encoded as a single `ml_code` range-match table; one entry per
leaf, each covering the per-feature ranges along the root-to-leaf path.

RF: each tree becomes its own range-match table `rf_tree_i` writing
⌈log₂C⌉ vote bits, and the final `rf_vote_classify` exact-match table resolves
the majority class from the concatenated vote register.

---

## Step 5 — P4 Code Generation

```bash
python3 5_generating_p4_code.py -m dt   # or -m rf
```

Emits two P4 programs:
- `../basic.p4` — BMv2 v1model (the primary evaluation target).
- `../p4sec_tofino.p4` — Intel Tofino (TNA architecture; discussion in paper
  Section 5, end-to-end compilation still under development).

Both files include: parser (Ethernet / ARP / IPv4 / TCP / UDP / ICMP),
canonical 5-tuple normalisation, CRC16/CRC32 hashes, register-based feature
accumulation, the trained reduction tables, the trained classifier tables,
and the digest of finalised flows to the controller.

---

## Step 6 — Compile & Run on BMv2

```bash
cd /tutorials/exercises/p4sec
make clean
make run
```

This brings up the BMv2 switch via the standard `p4lang/tutorials` harness
together with two Mininet hosts (`h1`, `h2`).

---

## Step 7 — Runtime Controller

In a second terminal:

```bash
cd /tutorials/exercises/p4sec/control_plane
python3 6_controller.py
```

The controller:
1. Loads `tables/s1-commands.txt` into the BMv2 switch via
   `simple_switch_CLI` (chunked for large RF vote tables).
2. Subscribes to flow digests emitted by `basic.p4` when a flow finalises
   (FIN/RST or idle timeout).
3. Maps each digest into `logs/predictions.csv` together with the predicted
   class label.

---

## Dataset

We evaluate on **CIC-IoT 2023**, restricted to four classes (Benign, DoS,
Brute Force, Reconnaissance). PCAPs go in `control_plane/pcaps/<class>/`;
the extractor derives labels from the directory name.

---

## Repository Layout

```
basic.p4                   Generated BMv2 program (overwritten by step 5)
p4sec_tofino.p4            Generated Tofino program (overwritten by step 5)
Makefile                   Standard p4lang/tutorials make wrapper
s1-runtime.json            BMv2 switch runtime descriptor
topology.json              Mininet topology (h1 ↔ s1 ↔ h2)

control_plane/
    1_extract_dataset.py            Feature extraction (PCAP → CSV)
    2_pca_generate_entries.py       PCA reduction + DTR surrogate
    2_lda_generate_entries.py       LDA reduction + DTR surrogate
    2_autoencoder_generate_entries.py  Autoencoder reduction + DTR surrogate
    3_train_model.py                DT / RF training
    4_generate_model_entries.py     Range-match classifier entries
    5_generating_p4_code.py         BMv2 + Tofino code synthesis
    6_controller.py                 Runtime: load rules, log digests
    pipeline_utils.py               Shared feature & config helpers
    filter_pcaps.py                 Optional PCAP preprocessing
```

---

## Citation

If you use this code, please cite the accompanying paper.
