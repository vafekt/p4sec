#!/bin/bash
# Automated P4sec demo runner
# - Pane 0 (left):  make clean && make run  (mininet + P4 switch)
# - Pane 1 (right): 6_controller.py
#
# tcpreplay and ip-link-set are run DIRECTLY in this script with sudo —
# s1-eth1 is a veth in the host root namespace so no mininet CLI needed.
# This removes all async completion-detection complexity.

set -euo pipefail

command -v tmux      &>/dev/null || { echo "ERROR: tmux not found.  sudo apt install tmux"; exit 1; }
command -v tcpreplay &>/dev/null || { echo "ERROR: tcpreplay not found.  sudo apt install tcpreplay"; exit 1; }

SESSION="p4sec-demo"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
CONTROLLER_LOG="/tmp/p4sec_controller.log"
METADATA_TSV="/tmp/p4sec_metadata.tsv"
PREDICTIONS_CSV="$WORKDIR/control_plane/logs/predictions.csv"
PCAP_DIR="$WORKDIR/control_plane/pcaps/CICIoT"
RESULTS_DIR="$WORKDIR/results/CICIoT_$(date +%Y%m%d_%H%M%S)"
MININET_PANE="$SESSION:0.0"
CTRL_PANE="$SESSION:0.1"

# Seconds to wait after tcpreplay returns for remaining flows to time out
# and be written to predictions.csv  (P4 flow timeout = 20 s).
FLOW_WAIT=30

# ── helpers ────────────────────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

csv_rows() {
    local n
    n=$(wc -l < "$PREDICTIONS_CSV" 2>/dev/null || echo 0)
    echo $(( n > 0 ? n - 1 : 0 ))
}

true_label_of() { echo "$1" | cut -d'.' -f1; }

show_ctrl_tail() {
    local n="${1:-5}"
    if [[ -s "$CONTROLLER_LOG" ]]; then
        tail -n "$n" "$CONTROLLER_LOG" | sed 's/^/      /'
    else
        echo "    [controller log empty]"
    fi
}

# Wait until predictions.csv stops growing for 4 × 5 s = 20 s.
drain_controller_queue() {
    local stable=0 prev; prev=$(csv_rows)
    log "  Draining controller queue..."
    while (( stable < 4 )); do
        sleep 5
        local cur; cur=$(csv_rows)
        if (( cur == prev )); then
            stable=$(( stable + 1 ))   # avoid (( stable++ )) — returns 0 on first iter → set -e kills script
            log "  [drain ${stable}/4] stable at $cur rows"
        else
            stable=0
            log "  [drain] growing: $prev → $cur (+$(( cur - prev )))"
            prev=$cur
        fi
    done
    log "  Drained — $( csv_rows ) rows total."
}

check_controller_alive() {
    local c
    c=$(tmux capture-pane -t "$CTRL_PANE" -p 2>/dev/null || true)
    if echo "$c" | grep -qiE 'traceback|errno|exception'; then
        log "  WARNING: controller pane shows an error:"
        echo "$c" | grep -iE 'traceback|errno|exception' | tail -3 | sed 's/^/    /'
    fi
}

# ── main ───────────────────────────────────────────────────────────────────────

# Cache sudo credentials up front so tcpreplay/ip-link-set don't prompt mid-run
log "Caching sudo credentials (you may be asked for your password once)..."
sudo -v

# Kill any leftover session
tmux kill-session -t "$SESSION" 2>/dev/null && log "Killed previous tmux session."
rm -f "$CONTROLLER_LOG" "$METADATA_TSV"

log "Creating tmux session '$SESSION'..."
log "(Tip: run  tmux attach -t $SESSION  in another terminal to watch live)"
tmux new-session -d -s "$SESSION" -x 240 -y 60
tmux split-window -h -t "$SESSION"

# ── Pane 0: mininet ────────────────────────────────────────────────────────────
log "Pane 0: make clean && make run ..."
tmux send-keys -t "$MININET_PANE" "cd '$WORKDIR' && make clean && make run" Enter

log "Waiting 15 s for mininet to initialise..."
sleep 15

# ── Pane 1: controller ─────────────────────────────────────────────────────────
log "Pane 1: starting controller  (log → $CONTROLLER_LOG)"
tmux send-keys -t "$CTRL_PANE" \
    "cd '$WORKDIR/control_plane' && PYTHONUNBUFFERED=1 python3 -u ./6_controller.py 2>&1 | tee '$CONTROLLER_LOG'" Enter

# ── Wait for rules ─────────────────────────────────────────────────────────────
log "Waiting for controller to install rules..."
RULES_TIMEOUT=600
elapsed=0
last_report=0
while (( elapsed < RULES_TIMEOUT )); do
    if grep -q "Rules installed" "$CONTROLLER_LOG" 2>/dev/null; then
        log "Rules installed."
        break
    fi
    if (( elapsed - last_report >= 30 )); then
        log "  Still loading rules... (${elapsed}s)"
        show_ctrl_tail 3
        last_report=$elapsed
    fi
    sleep 5; (( elapsed += 5 ))
done
if (( elapsed >= RULES_TIMEOUT )); then
    log "ERROR: rules not installed within ${RULES_TIMEOUT}s — aborting."
    cat "$CONTROLLER_LOG" 2>/dev/null | tail -20 | sed 's/^/  /'
    exit 1
fi
show_ctrl_tail 6
sleep 2

# ── Set MTU (run directly — s1-eth1 is in the host root namespace) ────────────
log "Setting s1-eth1 MTU to 65535..."
sudo ip link set s1-eth1 mtu 65535
log "MTU set."

# ── Collect pcap list ──────────────────────────────────────────────────────────
mapfile -t PCAPS < <(find "$PCAP_DIR" -maxdepth 1 \( -name "*.pcap" -o -name "*.pcapng" \) | sort)
(( ${#PCAPS[@]} > 0 )) || { log "ERROR: no pcap files in $PCAP_DIR"; exit 1; }

log "Will replay ${#PCAPS[@]} pcap file(s):"
printf '  %s\n' "${PCAPS[@]}"

# ── Replay loop ────────────────────────────────────────────────────────────────
: > "$METADATA_TSV"

for pcap in "${PCAPS[@]}"; do
    pcap_name=$(basename "$pcap")
    true_label=$(true_label_of "$pcap_name")
    start_row=$(csv_rows)

    log "─── [$true_label] $pcap_name  (start_row=$start_row)"

    # Run tcpreplay DIRECTLY in this shell — synchronous, no detection tricks.
    # sudo is needed to send raw packets on the veth interface.
    sudo tcpreplay -i s1-eth1 --timer=gtod "$pcap"

    log "  tcpreplay done. Waiting ${FLOW_WAIT}s for flows to expire..."
    fw_waited=0
    fw_prev=$(csv_rows)
    while (( fw_waited < FLOW_WAIT )); do
        sleep 5; (( fw_waited += 5 ))
        fw_cur=$(csv_rows)
        log "  [flow-wait ${fw_waited}/${FLOW_WAIT}s] predictions: $fw_cur rows (+$(( fw_cur - fw_prev )))"
        fw_prev=$fw_cur
        check_controller_alive
    done

    drain_controller_queue

    end_row=$(csv_rows)
    log "  Captured $(( end_row - start_row )) predictions (rows $start_row–$end_row)"
    printf '%s\t%s\t%d\t%d\n' "$pcap_name" "$true_label" "$start_row" "$end_row" >> "$METADATA_TSV"
    log "─── Done: $pcap_name"
    sleep 1
done

# ── Metrics ────────────────────────────────────────────────────────────────────
log "All files replayed. Computing metrics..."
python3 "$WORKDIR/analyze_results.py" "$PREDICTIONS_CSV" "$METADATA_TSV" "$RESULTS_DIR" \
    "$WORKDIR/control_plane/logs/run_metadata.json"

log "Done. Mininet still running — attach with:  tmux attach -t $SESSION"
