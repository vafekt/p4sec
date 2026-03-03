#!/bin/bash
#
# Script to replay PCAP files to P4 switch with PRESERVED TIMING
# This ensures Duration, MaxIAT, and MinIAT match training values
#
# Usage:
#   ./send_pcap.sh <pcap_file> <interface> [options]
#
# Examples:
#   ./send_pcap.sh control_plane/pcaps/TCP-DDoS.v1.pcap h1-eth0
#   ./send_pcap.sh control_plane/pcaps/Benign.v1.pcap h1-eth0 --multiplier=1
#   ./send_pcap.sh control_plane/pcaps/Recon.v1.pcap h1-eth0 --mbps=10
#

if [ $# -lt 2 ]; then
    echo "Usage: $0 <pcap_file> <interface> [tcpreplay_options]"
    echo ""
    echo "Common options:"
    echo "  --multiplier=1      Replay at original speed (DEFAULT - preserves timing)"
    echo "  --multiplier=0.5    Replay at half speed"
    echo "  --multiplier=2      Replay at 2x speed"
    echo "  --mbps=10           Replay at fixed 10 Mbps"
    echo "  --pps=1000          Replay at fixed 1000 packets/sec"
    echo "  --topspeed          Replay as fast as possible (BREAKS TIMING FEATURES)"
    echo ""
    echo "To preserve timing features (Duration, MaxIAT, MinIAT):"
    echo "  YOU MUST USE: --multiplier=1 (or close to 1)"
    exit 1
fi

PCAP_FILE=$1
INTERFACE=$2
shift 2
EXTRA_ARGS="$@"

# Check if tcpreplay is installed
if ! command -v tcpreplay &> /dev/null; then
    echo "ERROR: tcpreplay is not installed"
    echo "Install it with:"
    echo "  Ubuntu/Debian: sudo apt-get install tcpreplay"
    echo "  CentOS/RHEL:   sudo yum install tcpreplay"
    echo "  macOS:         brew install tcpreplay"
    exit 1
fi

# Check if pcap file exists
if [ ! -f "$PCAP_FILE" ]; then
    echo "ERROR: PCAP file not found: $PCAP_FILE"
    exit 1
fi

# Default to --multiplier=1 if no speed option provided
if [[ ! "$EXTRA_ARGS" =~ (--multiplier|--mbps|--pps|--topspeed) ]]; then
    echo "No timing option specified, using --multiplier=1 (original speed)"
    EXTRA_ARGS="--multiplier=1 $EXTRA_ARGS"
fi

echo "=========================================="
echo "Replaying PCAP with preserved timing"
echo "=========================================="
echo "PCAP file:  $PCAP_FILE"
echo "Interface:  $INTERFACE"
echo "Options:    $EXTRA_ARGS"
echo "=========================================="
echo ""

# Replay the pcap file
sudo tcpreplay --intf1=$INTERFACE $EXTRA_ARGS "$PCAP_FILE"

echo ""
echo "Replay complete!"
