#!/bin/bash
# Run P4 switch controller with PCAP replay for testing
# This script:
# 1. Cleans and compiles the P4 code
# 2. Starts the BMv2 switch with the controller
# 3. Replays a PCAP file through the switch with proper timing
# 4. Outputs results to predictions.csv

set -e

PCAP_FILE="${1:-control_plane/pcaps/Mu-IoT/Benign.v1.pcap}"
INTERFACE="${2:-s1-eth1}"
TIMEOUT_AFTER_PCAP="${3:-15}"  # seconds to wait after PCAP finishes before shutting down

if [ ! -f "$PCAP_FILE" ]; then
    echo "ERROR: PCAP file not found: $PCAP_FILE"
    exit 1
fi

echo "=== P4 Switch + PCAP Replay Testing ==="
echo "PCAP File: $PCAP_FILE"
echo "Interface: $INTERFACE"
echo ""

# Step 1: Clean and compile
echo "[1] Compiling P4 code..."
make clean && make > /dev/null 2>&1
echo "    Done"

# Step 2: Start make run in the background (starts switch and controller)
echo "[2] Starting BMv2 switch and controller..."
timeout 120 make run > /tmp/switch.log 2>&1 &
MAKE_PID=$!
sleep 3  # Give switch time to start

# Step 3: Replay PCAP with timing preservation
echo "[3] Replaying PCAP with tcpreplay (preserving timing)..."
if command -v tcpreplay &> /dev/null; then
    # Use multiplier=1 to preserve original PCAP timing
    # This ensures Duration, MaxIAT, MinIAT match the offline extraction
    tcpreplay --interface="$INTERFACE" --multiplier=1 "$PCAP_FILE" 2>/dev/null || true
    echo "    PCAP replay complete"
else
    echo "    WARNING: tcpreplay not installed - cannot replay PCAP"
    echo "    Install with: sudo apt-get install tcpreplay"
fi

# Step 4: Wait for switch to process and timeout flows
echo "[4] Waiting for controller to flush remaining flows..."
sleep "$TIMEOUT_AFTER_PCAP"

# Step 5: Gracefully stop the switch
echo "[5] Stopping switch and controller..."
kill $MAKE_PID 2>/dev/null || true
wait $MAKE_PID 2>/dev/null || true

sleep 1

# Step 6: Show results
echo ""
echo "=== Results ==="
if [ -f "control_plane/logs/predictions.csv" ]; then
    NUM_FLOWS=$(($(wc -l < control_plane/logs/predictions.csv) - 1))
    echo "Flows captured: $NUM_FLOWS"
    echo "Predictions saved to: control_plane/logs/predictions.csv"
    
    # Compare with dataset if available
    if [ -f "control_plane/dataset/recon_fresh.csv" ]; then
        NUM_DATASET=$(($(wc -l < control_plane/dataset/recon_fresh.csv) - 1))
        MATCH_PERCENT=$(echo "scale=1; 100*$NUM_FLOWS/$NUM_DATASET" | bc)
        echo "Dataset flows: $NUM_DATASET"
        echo "Match: $MATCH_PERCENT%"
    fi
else
    echo "ERROR: No predictions.csv found"
fi

echo ""
echo "Done!"
