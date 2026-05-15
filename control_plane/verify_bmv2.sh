#!/usr/bin/env bash
#
# End-to-end BMv2 verification using the same CIC-IoT PCAPs the model was
# trained on. Boots simple_switch_grpc with veth pairs, loads the compiled
# table entries, runs the controller, replays each labelled PCAP via
# tcpreplay, and prints per-class macro-F1 accuracy.
#
# Run from the control_plane/ directory AFTER you have already produced
# basic.p4, tables/s1-commands.txt, model/dt.model via steps 1-5.
#
#   bash verify_bmv2.sh
#
# Requires sudo (to create veth pairs and run simple_switch_grpc).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PCAP_DIR="$SCRIPT_DIR/CIC-IoT"
BASIC_JSON="/tmp/p4sec_basic.json"
P4INFO="/tmp/p4sec_basic.p4info.txtpb"
SWITCH_LOG="/tmp/p4sec_s1.log"
CTRL_LOG="$SCRIPT_DIR/logs/verify_controller.log"
PRED_CSV="$SCRIPT_DIR/logs/predictions.csv"
LABEL_CSV="$SCRIPT_DIR/logs/verify_labels.csv"

mkdir -p "$SCRIPT_DIR/logs"

echo "==> Compiling basic.p4"
p4c-bm2-ss --p4v 16 --p4runtime-files "$P4INFO" -o "$BASIC_JSON" \
    "$PROJECT_DIR/basic.p4"

cleanup() {
    sudo pkill -f 6_controller.py 2>/dev/null || true
    sudo pkill -f simple_switch_grpc 2>/dev/null || true
    sudo ip link delete veth0 2>/dev/null || true
    sudo ip link delete veth2 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Creating veth pairs (veth0 <-> veth1, veth2 <-> veth3)"
cleanup
sudo ip link add name veth0 type veth peer name veth1
sudo ip link add name veth2 type veth peer name veth3
for v in veth0 veth1 veth2 veth3; do
    sudo ip link set dev "$v" up
    sudo ethtool -K "$v" tx off rx off sg off tso off ufo off gso off gro off lro off 2>/dev/null || true
    sudo sysctl -w net.ipv6.conf.${v}.disable_ipv6=1 >/dev/null || true
done

echo "==> Starting simple_switch_grpc on veth1 (port 1) and veth3 (port 2)"
sudo simple_switch_grpc \
    -i 1@veth1 -i 2@veth3 \
    --log-file "$SWITCH_LOG" \
    --thrift-port 9090 \
    --nanolog ipc:///tmp/bm-0-log.ipc \
    -- --grpc-server-addr 127.0.0.1:50051 \
    "$BASIC_JSON" > /dev/null 2>&1 &
sleep 3

echo "==> Starting controller (writes $PRED_CSV)"
rm -f "$PRED_CSV"
cd "$SCRIPT_DIR"
nohup python3 6_controller.py > "$CTRL_LOG" 2>&1 &
sleep 10

echo "==> Replaying each PCAP through veth0 (label is captured per replay)"
echo "label,start_row" > "$LABEL_CSV"
for pcap in "$PCAP_DIR"/*.pcap; do
    label="$(basename "$pcap" | cut -d. -f1)"
    start_row=$(wc -l < "$PRED_CSV" 2>/dev/null || echo 0)
    echo "$label,$start_row" >> "$LABEL_CSV"
    echo "    --> $label  ($(basename "$pcap"))  predictions-before=$start_row"
    sudo tcpreplay -i veth0 --topspeed --quiet "$pcap"
    sleep 3
done

echo "==> Waiting for in-flight flows to drain (idle timeout = 20 s)"
sleep 25

echo "==> Stopping controller and switch"
sudo pkill -f 6_controller.py 2>/dev/null || true
sleep 1
sudo pkill -f simple_switch_grpc 2>/dev/null || true
sleep 2

echo "==> Computing per-class accuracy"
python3 - "$PRED_CSV" "$LABEL_CSV" << 'PYEOF'
import csv, sys
from collections import Counter

pred_csv, label_csv = sys.argv[1], sys.argv[2]

# Read label boundaries: each row says "first prediction row index for this label"
boundaries = []
with open(label_csv) as f:
    next(f)
    for line in f:
        lbl, idx = line.strip().split(',')
        boundaries.append((lbl, int(idx)))

# Read predictions
rows = []
with open(pred_csv) as f:
    header = next(f).strip().split(',')
    label_idx = header.index('class_label')
    for line in f:
        cols = line.strip().split(',')
        rows.append(cols[label_idx])

# Assign each row to the label whose boundary it falls within
results = {}
for i, (lbl, start) in enumerate(boundaries):
    end = boundaries[i+1][1] - 1 if i + 1 < len(boundaries) else len(rows)
    segment = rows[start:end]
    correct = sum(1 for p in segment if p == lbl)
    total = len(segment)
    results[lbl] = (correct, total, Counter(segment))

print("\n" + "="*70)
print(f"{'Class':16s}  {'Correct':>8s}  {'Total':>8s}  {'Accuracy':>10s}  Prediction breakdown")
print("="*70)
total_correct = 0
total_count = 0
for lbl, (c, t, br) in results.items():
    acc = c/t if t else 0.0
    print(f"{lbl:16s}  {c:>8d}  {t:>8d}  {acc*100:>9.2f}%   {dict(br)}")
    total_correct += c
    total_count += t

overall = total_correct / total_count if total_count else 0.0
print("="*70)
print(f"Overall BMv2 accuracy: {overall*100:.2f}%  ({total_correct}/{total_count})")
print(f"Target: >= 97.00%")
print(f"Result: {'PASS' if overall >= 0.97 else 'FAIL'}")
PYEOF
