# In-Network IoT Attack Detection Using Principal Component Analysis and Machine Learning Models on P4 Programmable Data Planes

Reference implementation for the paper
*"In-Network IoT Attack Detection Using Principal Component Analysis and
Machine Learning Models on P4 Programmable Data Planes."*

The pipeline extracts 20 bidirectional flow features at line rate, optionally
projects them onto a K-dimensional PCA subspace with B-bit quantisation, and
classifies finalised flows with a Decision Tree (DT) or Random Forest (RF)
encoded entirely as range-match P4 tables on the BMv2 v1model target. PCA
and the classifier share the same encoding, so each leaf becomes a single
range-match rule with per-feature ranges.

Four CIC-IoT 2023 categories are targeted: **Benign**, **DoS**,
**Brute Force**, and **Reconnaissance**.

LDA and Autoencoder are kept in the repository as drop-in alternative
reduction methods; the paper itself evaluates PCA only.

## Pipeline Overview

```
PCAP files (control_plane/CIC-IoT/)
    |
    v
[1] Feature extraction (1_extract_dataset.py)
    canonical 5-tuple, CRC-indexed per-flow registers,
    20 bidirectional flow features (paper Table 2)
    |
    v
dataset/dataset.csv
    |
    v
[2] Dimensionality reduction
    PCA  ->  2_pca_generate_entries.py          (paper, primary)
    raw  ->  2_raw_features.py                   (paper Section 4.2 baseline)
    LDA  ->  2_lda_generate_entries.py           (extra)
    AE   ->  2_autoencoder_generate_entries.py   (extra)
    K-component projection, B-bit quantization, DT regressor surrogate
    emitted as range-match P4 entries
    |
    v
tables/reduction_config.json
tables/transform_mapping.csv   (K codes + Label, or quantised raw features + Label)
tables/s1-commands.txt         (transform entries; empty for raw mode)
    |
    v
[3] Classifier training (3_train_model.py -m {dt|rf})
    |
    v
model/{dt|rf}.model
tables/model_metrics.json
    |
    v
[4] Classifier table generation (4_generate_model_entries.py -m {dt|rf})
    DT  -> ml_code range-match table
    RF  -> rf_tree_i range-match tables + rf_vote_classify exact table
    |
    v
tables/s1-commands.txt   (appended with classifier entries)
    |
    v
[5] P4 code generation (5_generating_p4_code.py -m {dt|rf})
    |
    v
../basic.p4
    |
    v
[6] Compile and deploy on BMv2 (make clean && make run)
    |
    v
[7] Runtime controller (6_controller.py)
    loads s1-commands.txt, listens for digests, logs predictions
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
| Timing | Duration, MaxIAT | 48, 48 |
| Volume | FwdPktCount, BwdPktCount, FwdBytes, BwdBytes, FwdMaxPktLen, BwdMaxPktLen | 32x4, 16x2 |
| TCP Flags | FlagsSyn, FlagsAck, FlagsFin, FlagsRst, FlagsPsh | 32x5 |
| Window | MaxWinSize, InitFwdWinBytes | 16, 16 |

Per-flow state is held in CRC16-indexed register arrays; a CRC32-indexed Bloom
filter detects index collisions. A flow is finalised on TCP FIN/RST, after a
20-second idle timeout, or via the amortised scan-and-drain action.

The two cross-flow features from the paper's Table 2 (`FlowCountPerSrc`,
`SynCountPerDst`) are intentionally omitted from this implementation:
their per-IP registers accumulate across the entire switch lifetime, which
makes per-pcap replay tests order-dependent on BMv2 without provided no
hardware register reset path.

## Step 2: Dimensionality Reduction

The paper's headline configuration is **PCA with k=7, b=32, DT**, which
delivers the peak BMv2 macro F1 of 97.07% on 118,588 entries.

```bash
# PCA (paper, primary). Defaults are the paper's recommended k=7, b=32.
python3 2_pca_generate_entries.py --components 7 --bits 32

# Raw-feature baseline (paper Section 4.2 — no PCA)
python3 2_raw_features.py

# Alternative reduction methods (not in paper, kept as extras)
python3 2_lda_generate_entries.py --components 3 --bits 16
python3 2_autoencoder_generate_entries.py --components 7 --bits 16
```

The PCA step:
* Fits the projection on the raw integer features.
* Quantises each component score to B bits using
  `code_j = clamp(round((pc_j - min_j)/range_j * (2^B-1)), 0, 2^B-1)`.
* Trains a Decision Tree Regressor (DTR) to approximate the projection.
  Each DTR leaf stores the mean quantised code over its training samples.
* Emits the DTR as range-match P4 entries (one entry per leaf per component).
* Writes `tables/reduction_config.json` so steps 3, 4, 5 detect the method
  automatically.

The paper evaluates **24 PCA configurations**: k in {5, 6, 7, 8, 9, 10},
b in {16, 32}, classifier in {DT, RF}. Macro F1 ranges from 95.45% to 97.07%
on the BMv2 data plane across these settings. The raw-feature DT baseline
reaches 97.09% on 436 entries with a 552-bit composite key.

## Step 3: Classifier Training

```bash
python3 3_train_model.py -m dt          # Decision Tree
python3 3_train_model.py -m rf -n 4     # Random Forest, 4 trees (paper setting)
```

Trains on the K quantised codes produced in step 2 (or on the 20 quantised
raw features in raw mode). Reports 80/20 stratified holdout metrics, then
retrains on 100% of the data for deployment.

Output:
* `model/{dt|rf}.model`
* `tables/model_metrics.json`
* `tables/model_params.json` (RF only, vote-packing parameters)

## Step 4: Classifier Table Generation

```bash
python3 4_generate_model_entries.py -m dt
python3 4_generate_model_entries.py -m rf
```

DT: the tree is encoded as a single `ml_code` range-match table; one entry
per leaf, each covering the per-feature ranges along the root-to-leaf path.

RF: each tree becomes its own range-match table `rf_tree_i` writing
ceil(log2(C)) vote bits, and the final `rf_vote_classify` exact-match table
resolves the majority class from the concatenated vote register.

## Step 5: P4 Code Generation

```bash
python3 5_generating_p4_code.py -m dt   # or -m rf
```

Emits `../basic.p4`: parser (Ethernet, ARP, IPv4, TCP, UDP, ICMP), canonical
5-tuple normalisation, CRC16/CRC32 hashes, per-flow and per-IP register
accumulation, the trained reduction tables (in PCA mode), the trained
classifier tables, and a digest of finalised flows to the controller.

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
2. Subscribes to flow digests emitted by `basic.p4` when a flow finalises.
3. Writes each digest into `logs/predictions.csv` together with the
   predicted class label.

## End-to-End Verification on BMv2

To validate the deployed pipeline on real traffic rather than relying on
the offline holdout split:

1. Run steps 1 to 5 once with the PCAPs in place.
2. `make run` to bring up the BMv2 switch.
3. In a second terminal, start the controller from step 7.
4. From a Mininet host, replay one of the labelled PCAPs into the switch:

   ```bash
   mininet> h1 ip link set eth0 mtu 9000        # avoid PF_PACKET EMSGSIZE on large frames
   mininet> h1 tcpreplay -i eth0 -p 200 /tutorials/exercises/p4sec/control_plane/CIC-IoT/DoS.v1.pcap
   ```

5. Optional: send drain-trigger packets (source IP 10.255.255.254) to
   finalise stale flows still sitting in the per-flow registers without
   waiting for the 20-second idle timeout. A pre-built helper pcap can be
   generated with scapy:

   ```python
   from scapy.all import Ether, IP, UDP, wrpcap
   pkts = [Ether()/IP(src='10.255.255.254', dst='10.255.255.253')/UDP(sport=1000+i, dport=2000)
           for i in range(2048)]
   wrpcap('drain.pcap', pkts)
   ```

   Each drain-trigger packet sweeps 32 register slots, so 2,048 packets
   cover all 65,536 slots.

6. Stop the controller and compare `logs/predictions.csv` against the
   ground-truth label encoded in the PCAP filename.

### Why live BMv2 accuracy can lag offline holdout accuracy

Two unavoidable sources of noise mean that line-rate accuracy on BMv2 will
typically be a few percentage points below the offline holdout number, even
with a model that scores 100% on the test split:

1. **`tcpreplay` cannot reproduce the original packet timing.** The trained
   model sees `Duration`, `MaxIAT` (and, internally, the inter-packet
   gaps that influence flow-end decisions) at the resolution BMv2 measures
   them: microseconds (multiplied to nanoseconds). `tcpreplay` re-emits
   packets at a software-scheduled rate (`-p`, `--mbps`, or
   `--topspeed`), so the inter-packet gaps the switch records are *not*
   the gaps the offline extractor saw when reading the same pcap directly.
   For flows whose decision boundary depends on timing features at the
   sub-millisecond range, a few-microsecond shift is enough to push a
   sample to the wrong side of a range-match rule.

2. **PCAPs contain background / mixed traffic.** Public attack pcaps are
   captures of *complete* sessions, not curated single-class flows. A
   "BruteForce" pcap contains the brute-force flows plus the ARP, DNS,
   ICMP and TCP control-plane traffic that ran alongside them. The
   filename label is applied to every flow in the file, so a few
   benign-looking ACK/FIN exchanges inside the BruteForce pcap legitimately
   get classified as Benign (or as the nearest neighbouring class) at line
   rate, even though they carry the BruteForce label in the dataset.

Both effects show up most strongly on small or visually-similar classes
(Evasion in `AttackIDS/`, BruteForce-vs-DoS in `CIC-IoT/`). They do **not**
indicate a bug in the pipeline: the same trained model still produces 100%
agreement when fed its own training PCA codes back through the DT.

## Bundled Example PCAPs

Two folders of labelled captures ship with the repo so you can run the
pipeline end-to-end without sourcing a dataset:

| Folder | Classes | Approx. size | Source |
|---|---|---|---|
| `control_plane/CIC-IoT/` | Benign, BruteForce, DoS, Reconnaissance | ~24 MB (not committed; place files manually) | CIC-IoT 2023 subset |
| `control_plane/AttackIDS/` | Access, CC, Discovery, Evasion | ~480 KB (committed) | AttackIDS demo captures |

`AttackIDS/` is small enough to ship in-tree as a runnable example. To use
it, point step 1 at that directory:

```bash
python3 1_extract_dataset.py --mode pcap --pcap-dir AttackIDS --output dataset/dataset.csv
```

The 14 captures yield ~176 labelled flows. On the live BMv2 switch this
pipeline reproduces a measured **97.47% overall accuracy** (Access 100% /
CC 96.70% / Discovery 95.83%), within a couple of percentage points of
the offline holdout result, with the gap attributable to the two effects
described above.

## Repository Layout

```
basic.p4                       Generated BMv2 program (overwritten by step 5)
Makefile                       Standard p4lang/tutorials make wrapper
s1-runtime.json                BMv2 switch runtime descriptor
topology.json                  Mininet topology (h1, s1, h2)

control_plane/
    1_extract_dataset.py             Feature extraction (PCAP to CSV)
    2_pca_generate_entries.py        PCA reduction + DTR surrogate (paper, primary)
    2_raw_features.py                Raw-feature baseline (paper Section 4.2)
    2_lda_generate_entries.py        LDA reduction (extra)
    2_autoencoder_generate_entries.py  Autoencoder reduction (extra)
    3_train_model.py                 DT / RF training
    4_generate_model_entries.py      Range-match classifier entries
    5_generating_p4_code.py          BMv2 P4 code synthesis
    6_controller.py                  Runtime: load rules, log digests
    pipeline_utils.py                Shared feature and config helpers
    AttackIDS/                       Bundled example PCAPs (Access/CC/Discovery/Evasion)
    CIC-IoT/                         Place CIC-IoT 2023 PCAP files here
```

## Citation

If you use this code, please cite the accompanying paper.
