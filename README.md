# p4sec: In-Network IoT Attack Detection on a P4 Programmable Data Plane

Reference implementation for the paper
*"In-Network IoT Attack Detection Using Principal Component Analysis and
Machine Learning Models on P4 Programmable Data Planes."*

The pipeline trains a Decision Tree (DT) or Random Forest (RF) classifier on
20 bidirectional flow features and compiles it into range-match P4 tables that
run end-to-end on the BMv2 data plane. PCA is the primary dimensionality
reduction method evaluated in the paper; LDA and Autoencoder are also
supported as drop-in alternatives.

The classifier targets four CIC-IoT 2023 categories: **Benign**, **DoS**,
**Brute Force**, and **Reconnaissance**.

## Pipeline Overview

```
PCAP files (control_plane/CIC-IoT/)
    |
    v
[1] Feature extraction  ->  1_extract_dataset.py
    (canonical 5-tuple, register-based per-flow state,
     20 bidirectional flow features)
    |
    v
dataset/dataset.csv
    |
    v
[2] Dimensionality reduction
    PCA   ->  2_pca_generate_entries.py   (primary)
    LDA   ->  2_lda_generate_entries.py
    AE    ->  2_autoencoder_generate_entries.py
    (K-component projection, B-bit quantization, DT regressor surrogate
     emitted as range-match P4 entries)
    |
    v
tables/reduction_config.json
tables/transform_mapping.csv   (K quantized codes + Label)
tables/s1-commands.txt         (transform entries)
    |
    v
[3] Classifier training  ->  3_train_model.py  -m {dt|rf}
    |
    v
model/{dt|rf}.model
tables/model_metrics.json
    |
    v
[4] Classifier table generation  ->  4_generate_model_entries.py  -m {dt|rf}
    (ml_code for DT; rf_tree_i + rf_vote_classify for RF)
    |
    v
tables/s1-commands.txt   (appended with classifier entries)
    |
    v
[5] P4 code generation  ->  5_generating_p4_code.py  -m {dt|rf}
    |
    v
../basic.p4
    |
    v
[6] Compile and deploy on BMv2  ->  make clean && make run
    |
    v
[7] Runtime controller  ->  6_controller.py
    (loads s1-commands.txt, listens for digests, logs predictions)
```

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

P4 toolchain (BMv2, p4c, Mininet): follow https://github.com/p4lang/tutorials.

## Dataset

Place CIC-IoT 2023 PCAP files in `control_plane/CIC-IoT/`. One PCAP per
class is sufficient; the label is derived from the filename prefix:

```
control_plane/CIC-IoT/Benign.v1.pcap
control_plane/CIC-IoT/DoS.v1.pcap
control_plane/CIC-IoT/BruteForce.v1.pcap
control_plane/CIC-IoT/Reconnaissance.v1.pcap
```

## Step 1: Feature Extraction

```bash
cd control_plane
python3 1_extract_dataset.py --mode pcap --pcap-dir CIC-IoT --output dataset/dataset.csv
```

The 20 bidirectional flow features extracted (paper Table 2):

| Category | Feature | Bits |
|---|---|---|
| Protocol | Protocol | 8 |
| Ports | SrcPort, DstPort | 16, 16 |
| Timing | Duration, MaxIAT, MinIAT | 48, 48, 48 |
| Volume | FwdPktCount, BwdPktCount, FwdBytes, BwdBytes, FwdMaxPktLen, BwdMaxPktLen | 32x4, 16x2 |
| TCP Flags | FlagsSyn, FlagsAck, FlagsFin, FlagsRst, FlagsPsh, UrgCount | 32x6 |
| Window | MaxWinSize, InitFwdWinBytes | 16, 16 |

Per-flow state is held in CRC16-indexed register arrays; a CRC32-indexed Bloom
filter detects index collisions. A flow is finalized on TCP FIN/RST or after a
20-second idle timeout, after which it proceeds to classification.

## Step 2: Dimensionality Reduction

Choose one of the three methods. The paper's headline results use **PCA**.

```bash
# PCA (primary, paper Section 4)
python3 2_pca_generate_entries.py --components 7 --bits 32

# LDA
python3 2_lda_generate_entries.py --components 3 --bits 16

# Autoencoder
python3 2_autoencoder_generate_entries.py --components 7 --bits 16
```

Each method:
* Fits the projection on the raw integer features.
* Quantises every component score to B bits.
* Trains a Decision Tree Regressor (DTR) per component to approximate the
  projection. Each DTR leaf stores the mean quantized code over the training
  samples that fall into it.
* Emits the DTR as range-match P4 entries (one entry per leaf per component).
* Writes `tables/reduction_config.json` so steps 3, 4, 5 detect the method
  automatically.

The paper evaluates **24 PCA configurations**: k in {5, 6, 7, 8, 9, 10}
components, b in {16, 32} bits, classifier in {DT, RF}.

## Step 3: Classifier Training

```bash
python3 3_train_model.py -m dt          # Decision Tree
python3 3_train_model.py -m rf -n 4     # Random Forest, 4 trees
```

Trains on the K quantized codes produced in step 2. Reports 80/20 holdout
metrics, then retrains on 100% of the data for deployment.

Output:
* `model/{dt|rf}.model`
* `tables/model_metrics.json`
* `tables/model_params.json` (RF only, vote-packing parameters)

## Step 4: Classifier Table Generation

```bash
python3 4_generate_model_entries.py -m dt
python3 4_generate_model_entries.py -m rf
```

DT: the tree is encoded as a single `ml_code` range-match table; one entry per
leaf, each covering the per-feature ranges along the root-to-leaf path.

RF: each tree becomes its own range-match table `rf_tree_i` writing
ceil(log2(C)) vote bits, and the final `rf_vote_classify` exact-match table
resolves the majority class from the concatenated vote register.

## Step 5: P4 Code Generation

```bash
python3 5_generating_p4_code.py -m dt   # or -m rf
```

Emits `../basic.p4`: parser (Ethernet, ARP, IPv4, TCP, UDP, ICMP), canonical
5-tuple normalisation, CRC16/CRC32 hashes, register-based feature
accumulation, the trained reduction tables, the trained classifier tables,
and a digest of finalised flows to the controller.

## Step 6: Compile and Run on BMv2

```bash
cd /tutorials/exercises/p4sec
make clean
make run
```

This brings up the BMv2 switch via the standard `p4lang/tutorials` harness
together with two Mininet hosts (`h1`, `h2`).

## Step 7: Runtime Controller

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
3. Writes each digest into `logs/predictions.csv` together with the
   predicted class label.

## End-to-End Verification on BMv2

To validate the deployed pipeline on real CIC-IoT traffic rather than
relying on the offline holdout split:

1. Run steps 1 through 5 once with the CIC-IoT PCAPs in place.
2. `make run` to bring up the BMv2 switch.
3. In a second terminal, start the controller from step 7.
4. From a Mininet host, replay one of the labelled PCAPs into the switch,
   for example:

   ```bash
   mininet> h1 tcpreplay -i h1-eth0 /tutorials/exercises/p4sec/control_plane/CIC-IoT/DoS.v1.pcap
   ```

5. Stop the controller and compare `logs/predictions.csv` against the
   ground-truth label encoded in the PCAP filename.

## Repository Layout

```
basic.p4                       Generated BMv2 program (overwritten by step 5)
Makefile                       Standard p4lang/tutorials make wrapper
s1-runtime.json                BMv2 switch runtime descriptor
topology.json                  Mininet topology (h1, s1, h2)

control_plane/
    1_extract_dataset.py            Feature extraction (PCAP to CSV)
    2_pca_generate_entries.py       PCA reduction + DTR surrogate
    2_lda_generate_entries.py       LDA reduction + DTR surrogate
    2_autoencoder_generate_entries.py  Autoencoder reduction + DTR surrogate
    3_train_model.py                DT / RF training
    4_generate_model_entries.py     Range-match classifier entries
    5_generating_p4_code.py         BMv2 P4 code synthesis
    6_controller.py                 Runtime: load rules, log digests
    pipeline_utils.py               Shared feature and config helpers
    CIC-IoT/                        Place CIC-IoT 2023 PCAP files here
```

## Citation

If you use this code, please cite the accompanying paper.
