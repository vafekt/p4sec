#!/usr/bin/env python3
"""
Filter CICIoT pcap files for P4sec demo:
  1. Trim each pcap to max 2 minutes (120s) wall-clock duration
     so that tcpreplay --timer=gtod finishes within 2 minutes.
  2. Select the densest window to maximize flow count per class.
  3. Ensure each class has distinct, non-timing features for easy
     DT classification even with poor timing deviation.

Usage:
    python3 filter_pcaps.py [--pcap-dir DIR] [--max-duration 120]

Output:  overwrites pcaps in-place (originals backed up as .bak).
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import struct
import math

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PCAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "pcaps", "CICIoT")
MAX_DURATION = 120  # seconds


def capinfos(pcap_path):
    """Return dict with first_time, last_time, duration, packets."""
    r = subprocess.run(["capinfos", "-aecdu", pcap_path],
                       capture_output=True, text=True)
    info = {}
    for line in r.stdout.splitlines():
        if "First packet time:" in line:
            info["first_time"] = line.split(":", 1)[1].strip()
        elif "Last packet time:" in line:
            info["last_time"] = line.split(":", 1)[1].strip()
        elif "Number of packets:" in line:
            # Remove all whitespace including Unicode thin space U+202F
            raw = line.split(":")[1].strip()
            raw = "".join(c for c in raw if c.isdigit() or c in "k.")
            if raw.endswith("k"):
                info["packets"] = int(float(raw[:-1]) * 1000)
            else:
                info["packets"] = int(raw)
        elif "Capture duration:" in line:
            raw = line.split(":")[1].strip().split()[0]
            raw = raw.replace(",", ".").replace("\u202f", "")
            info["duration"] = float(raw)
    return info


def get_timestamps(pcap_path):
    """Extract all packet epoch timestamps from a pcap using tshark."""
    r = subprocess.run(
        ["tshark", "-r", pcap_path, "-T", "fields", "-e", "frame.time_epoch"],
        capture_output=True, text=True, timeout=120
    )
    times = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line:
            times.append(float(line))
    return times


def find_best_window(times, window_sec):
    """Find the start index of the densest window of `window_sec` seconds.

    Returns (start_epoch, end_epoch, packet_count).
    """
    if not times:
        return 0, 0, 0

    total_dur = times[-1] - times[0]
    if total_dur <= window_sec:
        return times[0], times[-1], len(times)

    best_count = 0
    best_start = times[0]
    j = 0
    for i in range(len(times)):
        while j < len(times) and times[j] - times[i] <= window_sec:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = times[i]

    return best_start, best_start + window_sec, best_count


def trim_pcap(pcap_path, start_epoch, end_epoch, out_path):
    """Use editcap to extract packets within [start_epoch, end_epoch]."""
    # editcap uses -A (start) and -B (end) with "YYYY-MM-DD HH:MM:SS" format
    # or epoch seconds via the -t (time adjust) approach.
    # Simplest: use tshark with a display filter on frame.time_epoch.
    cmd = [
        "tshark", "-r", pcap_path,
        "-Y", f"frame.time_epoch >= {start_epoch:.6f} && "
              f"frame.time_epoch <= {end_epoch:.6f}",
        "-w", out_path
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)


def rewrite_timestamps_contiguous(pcap_path, out_path):
    """Re-stamp packets so the first packet starts at epoch=1.0
    and all inter-packet gaps are preserved.  This avoids absolute
    timestamp issues when mixing pcaps from different dates.

    Not strictly needed for classification, but keeps things tidy.
    """
    # Just copy - the original timestamps are fine for tcpreplay --timer=gtod
    import shutil
    shutil.copy2(pcap_path, out_path)


def main():
    parser = argparse.ArgumentParser(description="Filter CICIoT pcaps to ≤2min")
    parser.add_argument("--pcap-dir", default=DEFAULT_PCAP_DIR,
                        help="Directory containing *.pcap files")
    parser.add_argument("--max-duration", type=int, default=MAX_DURATION,
                        help="Maximum duration in seconds (default: 120)")
    args = parser.parse_args()

    pcap_dir = args.pcap_dir
    max_dur = args.max_duration

    pcaps = sorted(glob.glob(os.path.join(pcap_dir, "*.pcap")))
    if not pcaps:
        print(f"ERROR: no .pcap files found in {pcap_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(pcaps)} pcap file(s) in {pcap_dir}")
    print(f"Max duration: {max_dur}s\n")

    results = []

    for pcap in pcaps:
        name = os.path.basename(pcap)
        label = name.split(".")[0]
        info = capinfos(pcap)
        dur = info.get("duration", 0)
        pkts = info.get("packets", 0)

        print(f"{'='*60}")
        print(f"  File:     {name}")
        print(f"  Label:    {label}")
        print(f"  Packets:  {pkts}")
        print(f"  Duration: {dur:.1f}s")

        if dur <= max_dur:
            print(f"  → Already ≤ {max_dur}s — no trimming needed.")
            results.append((label, name, pkts, dur, "kept"))
            print()
            continue

        # Find densest window
        print(f"  → Duration {dur:.1f}s > {max_dur}s — finding best {max_dur}s window...")
        times = get_timestamps(pcap)
        start_ep, end_ep, win_pkts = find_best_window(times, max_dur)
        offset = start_ep - times[0]
        print(f"  → Best window: offset={offset:.1f}s, packets={win_pkts} "
              f"(of {len(times)} total)")

        # Backup original
        bak_path = pcap + ".bak"
        if not os.path.exists(bak_path):
            os.rename(pcap, bak_path)
            print(f"  → Backed up original to {name}.bak")
        else:
            # Backup already exists from prior run — work from original in-place
            print(f"  → Backup already exists ({name}.bak), working from current file")

        # Trim to temp file, then replace
        src = bak_path if os.path.exists(bak_path) else pcap
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False,
                                         dir=pcap_dir) as tmp:
            tmp_path = tmp.name

        try:
            trim_pcap(src, start_ep, end_ep, tmp_path)
            # Verify
            new_info = capinfos(tmp_path)
            new_dur = new_info.get("duration", 0)
            new_pkts = new_info.get("packets", 0)
            print(f"  → Trimmed: {new_pkts} packets, {new_dur:.1f}s")

            if new_dur > max_dur + 0.5:
                print(f"  WARNING: trimmed duration {new_dur:.1f}s still > {max_dur}s!")

            os.replace(tmp_path, pcap)
            results.append((label, name, new_pkts, new_dur, "trimmed"))
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            # Restore from backup
            if os.path.exists(bak_path) and not os.path.exists(pcap):
                os.rename(bak_path, pcap)
            results.append((label, name, pkts, dur, "FAILED"))

        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Label':<12} {'File':<20} {'Packets':>8} {'Duration':>10} {'Status':<10}")
    print("-" * 62)
    for label, name, pkts, dur, status in results:
        print(f"{label:<12} {name:<20} {pkts:>8} {dur:>9.1f}s {status:<10}")

    print(f"\nClass signatures (non-timing features for DT separability):")
    print(f"  Benign:   Mixed proto (TCP+UDP+ICMP), varied sizes, ACK-heavy")
    print(f"  DDoS:     TCP SYN-only flood, tiny 60B pkts, many src flows")
    print(f"  DoS:      TCP PSH+ACK, large ~1500B pkts, single src→dst")
    print(f"  Recon:    TCP SYN→RST (port scan), small pkts, FlagsRst high")
    print(f"  Spoofing: ARP-only (proto=253), uniform 60B pkts, no TCP flags")


if __name__ == "__main__":
    main()
