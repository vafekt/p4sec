# In-Network IoT Intrusion Detection with PCA and ML on P4

Reference implementation for *"In-Network IoT Intrusion Detection Using Principal
Component Analysis and Machine Learning Models on P4 Programmable Data Planes"*
(MIST 2026, ESORICS workshop).

The switch extracts 20 bidirectional flow features at line rate, projects them onto
a K-dimensional PCA subspace, and classifies each finished flow with a Decision Tree
or Random Forest — all as range-match tables on BMv2. Target tier is an IoT gateway
(BMv2 / P4Pi on Raspberry Pi).

LDA and Autoencoder scripts are included as drop-in alternatives; the paper uses PCA.

---

## Setup

The repo must live inside the `p4lang/tutorials` tree:

```bash
cd tutorials/exercises/
git clone https://github.com/vafekt/p4sec.git && cd p4sec
pip install -r requirements.txt
sudo apt-get install tshark tcpdump tcpreplay
```

P4 toolchain (p4c, BMv2, Mininet): see https://github.com/p4lang/tutorials.

## Dataset

Drop one PCAP per class into a folder. The label is the filename prefix:

```
control_plane/CIC-IoT/Benign.v1.pcap
control_plane/CIC-IoT/DoS.v1.pcap
control_plane/CIC-IoT/BruteForce.v1.pcap
control_plane/CIC-IoT/Reconnaissance.v1.pcap
```

`control_plane/AttackIDS/` ships with the repo and works out of the box.

## Run the pipeline

```bash
cd control_plane
python3 1_extract_dataset.py --mode pcap --pcap-dir CIC-IoT   # -> dataset/dataset.csv
python3 2_pca_generate_entries.py                             # PCA transform tables
python3 3_train_model.py        --model-type rf               # -> model/rf.model
python3 4_generate_model_entries.py --model-type rf           # -> tables/s1-commands.txt
python3 5_generating_p4_code.py     --model-type rf           # -> basic.p4
```

Steps 2–5 must be re-run together. Changing the reduction step without
regenerating the classifier tables makes every flow classify as class 0.

Swap `rf` for `dt` to use a Decision Tree. Alternatives: `2_raw_features.py`
(no PCA baseline), `2_lda_generate_entries.py`, `2_autoencoder_generate_entries.py`.

## Deploy

```bash
cd ..            # back to p4sec/
make run         # compiles basic.p4, starts BMv2 + Mininet
```

In a second terminal:

```bash
cd control_plane && python3 6_controller.py
```

The controller installs the tables, subscribes to digests, prints each classified
flow, and writes `logs/predictions.csv`.

## Verify on live traffic

From the Mininet CLI:

```bash
mininet> h1 ip link set eth0 mtu 9000
mininet> h1 tcpreplay -i eth0 -p 300 control_plane/AttackIDS/CC.v1.pcap
```

Flows finalise after a 20 s idle timeout. To flush them immediately, send packets
from source IP `10.255.255.254` — each sweeps 32 register slots, so 2048 packets
cover all 65536.

Then compare `logs/predictions.csv` against the label in the PCAP filename.

**Live accuracy runs a few points below the offline holdout, by design not by bug.**
`tcpreplay` cannot reproduce original packet timing, so `Duration` and `MaxIAT` shift
by microseconds and flows near a range boundary fall the other way. Public PCAPs also
mix background ARP/DNS/ICMP into every file, and the filename label is applied to all
of it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `p4c-bm2-ss: libboost_iostreams.so... not found` | `apt install libboost-all-dev` |
| `Conditional execution in actions unsupported` | p4c too old; needs ≥ 1.2.5 |
| `Descriptors cannot be created directly` | `pip install "protobuf<4"` |
| `No module named 'p4.tmp'` | Put the repo dir first on `PYTHONPATH`; the pip `p4runtime` package shadows the bundled `p4/` |
| `make run` exits immediately | The Mininet CLI needs a live stdin — do not redirect from `/dev/null` |
| Most `table_add` lines rejected | `basic.p4` and `tables/s1-commands.txt` are out of sync; re-run steps 2–5 |
