#!/usr/bin/env python3
"""
Network Traffic Flow Feature Extractor
Extracts flow-based features from PCAP files or live network capture.

Features per flow (bidirectional, canonical key):
  IAT        - Sum of inter-arrival times >> 2 (matches P4 register approximation)
  Duration   - Time from first to last packet (nanoseconds)
  PktCount   - Total packet count (both directions)
  SrcPort    - Canonical source port (lower-endpoint side)
  DstPort    - Canonical destination port
  TotalBytes - Total bytes (both directions)
  FlagsSyn, FlagsAck, FlagsFin, FlagsRst - TCP flags aggregated over both directions

Canonical key logic (mirrors P4 switch exactly):
  The flow key is normalised so that both A->B and B->A packets are merged
  into the same flow entry, identical to how the P4 switch aggregates them.
  Rule: compare (src_ip, src_port) vs (dst_ip, dst_port) as tuples of integers.
    - If (src_ip_int, src_port) <= (dst_ip_int, dst_port): keep as-is
    - Otherwise: swap src/dst fields
"""

import os
import glob
import ipaddress
import pyshark
import csv
import argparse
import logging
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FLOW_TIMEOUT_NS = 120 * 1_000_000_000  # 120 seconds in nanoseconds


# ---------------------------------------------------------------------------
# Canonical key helper — MUST match P4 compute_flow_hash() logic exactly
# ---------------------------------------------------------------------------

def make_canonical_key(src_ip, src_port, dst_ip, dst_port, protocol):
    """
    Normalise a 5-tuple so that both directions of a flow map to the same key.

    Mirrors the P4 switch logic:
      if src_ip < dst_ip  →  keep order
      if src_ip > dst_ip  →  swap
      if equal IPs        →  compare ports; lower port = canonical src

    Returns (canon_src_ip, canon_src_port, canon_dst_ip, canon_dst_port, protocol)
    where all IP values remain as strings (for CSV output) but comparison is numeric.
    """
    src_int = int(ipaddress.ip_address(src_ip))
    dst_int = int(ipaddress.ip_address(dst_ip))

    if src_int < dst_int:
        return src_ip, src_port, dst_ip, dst_port, protocol
    elif src_int > dst_int:
        return dst_ip, dst_port, src_ip, src_port, protocol
    else:
        # Same IP (e.g. loopback): lower port = canonical src
        if src_port <= dst_port:
            return src_ip, src_port, dst_ip, dst_port, protocol
        else:
            return dst_ip, dst_port, src_ip, src_port, protocol


# ---------------------------------------------------------------------------
# Feature finalisation (shared by PCAP and live modes)
# ---------------------------------------------------------------------------

def _finalize_flow(flow_key, flow_data, label):
    """
    Convert accumulated flow state into a feature dict.
    flow_key = (canon_src_ip, canon_src_port, canon_dst_ip, canon_dst_port, protocol)
    """
    timestamps = flow_data['timestamps']
    if len(timestamps) < 2:
        return None

    lengths = flow_data['lengths']

    # Duration: first to last packet (nanoseconds)
    duration_ns = timestamps[-1] - timestamps[0]

    # IAT: sum of inter-arrival times, right-shifted by 2 to match P4 approximation.
    # P4 register stores sum_iat then reads meta.iat = sum_iat >> 2.
    sum_iat_ns = sum(timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps)))
    approx_iat_ns = sum_iat_ns >> 2

    total_bytes = sum(lengths)
    pkt_count   = len(timestamps)

    feature = {
        "IAT":        approx_iat_ns,
        "Duration":   duration_ns,
        "PktCount":   pkt_count,
        "SrcPort":    flow_key[1],   # canonical src port
        "DstPort":    flow_key[3],   # canonical dst port
        "TotalBytes": total_bytes,
        "FlagsSyn":   flow_data['flags_syn'],
        "FlagsAck":   flow_data['flags_ack'],
        "FlagsFin":   flow_data['flags_fin'],
        "FlagsRst":   flow_data['flags_rst'],
    }
    if label is not None:
        feature["Label"] = label
    return feature


def _empty_flow_state():
    return {
        'timestamps': [],
        'lengths':    [],
        'flags_syn':  0,
        'flags_ack':  0,
        'flags_fin':  0,
        'flags_rst':  0,
    }


# ---------------------------------------------------------------------------
# PCAP file extraction
# ---------------------------------------------------------------------------

def extract_features_from_pcap(pcap_path, label=None):
    """
    Extract bidirectional flow features from a PCAP/PCAPNG file.

    Both directions of a flow (A->B and B->A) are merged into a single flow
    entry using the same canonical key logic as the P4 switch.  The output
    feature set therefore matches what the switch sees at classification time.
    """
    features_list = []
    cap = pyshark.FileCapture(pcap_path)

    flows = defaultdict(lambda: None)
    packet_count = 0

    try:
        for packet in cap:
            try:
                if not hasattr(packet, 'ip'):
                    continue

                src_ip   = packet.ip.src
                dst_ip   = packet.ip.dst
                protocol = int(packet.ip.proto)

                if protocol == 6:   # TCP
                    src_port      = int(packet.tcp.srcport)
                    dst_port      = int(packet.tcp.dstport)
                    flags_syn     = int(packet.tcp.flags_syn)   if hasattr(packet.tcp, 'flags_syn')   else 0
                    flags_ack     = int(packet.tcp.flags_ack)   if hasattr(packet.tcp, 'flags_ack')   else 0
                    flags_fin     = int(packet.tcp.flags_fin)   if hasattr(packet.tcp, 'flags_fin')   else 0
                    flags_rst     = int(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
                elif protocol == 17:  # UDP
                    src_port  = int(packet.udp.srcport)
                    dst_port  = int(packet.udp.dstport)
                    flags_syn = flags_ack = flags_fin = flags_rst = 0
                else:
                    continue

                # --- Canonical key (matches P4 switch) ---
                c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto = \
                    make_canonical_key(src_ip, src_port, dst_ip, dst_port, protocol)
                flow_key = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto)

                timestamp_ns = int(float(packet.sniff_timestamp) * 1_000_000_000)
                pkt_len      = int(packet.length)

                # Initialise or timeout-reset
                if flows[flow_key] is None:
                    flows[flow_key] = _empty_flow_state()
                else:
                    state = flows[flow_key]
                    if state['timestamps'] and \
                            (timestamp_ns - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                        feat = _finalize_flow(flow_key, state, label)
                        if feat:
                            features_list.append(feat)
                        flows[flow_key] = _empty_flow_state()

                state = flows[flow_key]
                state['timestamps'].append(timestamp_ns)
                state['lengths'].append(pkt_len)
                state['flags_syn'] = max(state['flags_syn'], flags_syn)
                state['flags_ack'] = max(state['flags_ack'], flags_ack)
                state['flags_fin'] = max(state['flags_fin'], flags_fin)
                state['flags_rst'] = max(state['flags_rst'], flags_rst)

                packet_count += 1
                if packet_count % 1000 == 0:
                    logger.info(f"Processed {packet_count} packets, "
                                f"{len(flows)} flows from {os.path.basename(pcap_path)}")

            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
                continue
    finally:
        cap.close()

    # Finalise remaining flows
    for flow_key, state in flows.items():
        if state is not None:
            feat = _finalize_flow(flow_key, state, label)
            if feat:
                features_list.append(feat)

    logger.info(f"Extracted {len(features_list)} flows from {packet_count} packets in {pcap_path}")
    return features_list


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_to_csv(features_list, output_path, mode='w', write_header=True):
    """Write feature dicts to CSV."""
    if not features_list:
        logger.warning("No features to write")
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(features_list[0].keys())
    with open(output_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(features_list)

    logger.info(f"Wrote {len(features_list)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Multi-file PCAP processing helpers
# ---------------------------------------------------------------------------

def extract_base_label(filename):
    """Remove extension and version suffix (.v1, .v2 …) from filename."""
    base  = os.path.splitext(filename)[0]
    parts = base.split('.')
    for i, part in enumerate(parts):
        if part.startswith('v') and part[1:].isdigit():
            return '.'.join(parts[:i])
    return base


def process_single_pcap(pcap_path, output_csv, label=None):
    """Process one PCAP file and write to CSV."""
    logger.info(f"Processing PCAP file: {pcap_path}")
    features = extract_features_from_pcap(pcap_path, label=label)
    write_to_csv(features, output_csv, mode='w', write_header=True)


def process_pcap_folder(folder_path, output_csv):
    """Process all .pcap/.pcapng files in a folder into a single labeled CSV."""
    pcap_files  = sorted(glob.glob(os.path.join(folder_path, "*.pcap")))
    pcapng_files = sorted(glob.glob(os.path.join(folder_path, "*.pcapng")))
    all_files   = pcap_files + pcapng_files

    if not all_files:
        logger.warning(f"No .pcap or .pcapng files found in {folder_path}")
        return

    logger.info(f"Found {len(all_files)} capture files in {folder_path}")

    header_written = False
    for pcap_path in all_files:
        filename = os.path.basename(pcap_path)
        label    = extract_base_label(filename)
        logger.info(f"Processing {pcap_path}  (label={label})")
        features = extract_features_from_pcap(pcap_path, label=label)
        mode = 'w' if not header_written else 'a'
        write_to_csv(features, output_csv, mode=mode, write_header=not header_written)
        header_written = True

    logger.info(f"Finished processing. Output: {output_csv}")


# ---------------------------------------------------------------------------
# Live capture extraction
# ---------------------------------------------------------------------------

def extract_features_from_interface(interface, output_csv, label=None,
                                     packet_count=1000, duration=None):
    """
    Extract bidirectional flow features from live network capture.
    Uses the same canonical key and feature computation as the PCAP mode
    so that training data and live/replay inference data are consistent.
    """
    logger.info(f"Starting live capture on interface: {interface}")
    logger.info(f"Target packets: {packet_count}, "
                f"Duration: {duration if duration else 'unlimited'}")

    flows         = defaultdict(lambda: None)
    features_list = []
    cap           = pyshark.LiveCapture(interface=interface)
    captured      = 0
    start_time    = time.time()

    try:
        for packet in cap.sniff_continuously():
            try:
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit of {duration}s reached")
                    break
                if captured >= packet_count:
                    logger.info(f"Captured {captured} packets")
                    break

                if not hasattr(packet, 'ip'):
                    continue

                src_ip   = packet.ip.src
                dst_ip   = packet.ip.dst
                protocol = int(packet.ip.proto)

                if protocol == 6:
                    src_port  = int(packet.tcp.srcport)
                    dst_port  = int(packet.tcp.dstport)
                    flags_syn = int(packet.tcp.flags_syn)   if hasattr(packet.tcp, 'flags_syn')   else 0
                    flags_ack = int(packet.tcp.flags_ack)   if hasattr(packet.tcp, 'flags_ack')   else 0
                    flags_fin = int(packet.tcp.flags_fin)   if hasattr(packet.tcp, 'flags_fin')   else 0
                    flags_rst = int(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
                elif protocol == 17:
                    src_port  = int(packet.udp.srcport)
                    dst_port  = int(packet.udp.dstport)
                    flags_syn = flags_ack = flags_fin = flags_rst = 0
                else:
                    continue

                # --- Canonical key ---
                c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto = \
                    make_canonical_key(src_ip, src_port, dst_ip, dst_port, protocol)
                flow_key = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto)

                timestamp_ns = int(float(packet.sniff_timestamp) * 1_000_000_000)
                pkt_len      = int(packet.length)

                if flows[flow_key] is None:
                    flows[flow_key] = _empty_flow_state()
                else:
                    state = flows[flow_key]
                    if state['timestamps'] and \
                            (timestamp_ns - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                        feat = _finalize_flow(flow_key, state, label)
                        if feat:
                            features_list.append(feat)
                        flows[flow_key] = _empty_flow_state()

                state = flows[flow_key]
                state['timestamps'].append(timestamp_ns)
                state['lengths'].append(pkt_len)
                state['flags_syn'] = max(state['flags_syn'], flags_syn)
                state['flags_ack'] = max(state['flags_ack'], flags_ack)
                state['flags_fin'] = max(state['flags_fin'], flags_fin)
                state['flags_rst'] = max(state['flags_rst'], flags_rst)

                captured += 1
                if captured % 100 == 0:
                    logger.info(f"Captured {captured} packets, {len(flows)} active flows…")

            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("\nCapture interrupted by user")
    finally:
        cap.close()

    for flow_key, state in flows.items():
        if state is not None:
            feat = _finalize_flow(flow_key, state, label)
            if feat:
                features_list.append(feat)

    logger.info(f"Extracted {len(features_list)} flows from {captured} packets")

    if features_list:
        write_to_csv(features_list, output_csv, mode='w', write_header=True)
    else:
        logger.warning("No flows captured")

    return features_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract bidirectional flow features from PCAP/PCAPNG files or live capture'
    )
    parser.add_argument('--mode', choices=['pcap', 'live'], required=True,
                        help='Processing mode: pcap (file) or live (interface)')
    parser.add_argument('--input',    help='Input PCAP file (single file mode)')
    parser.add_argument('--pcap-dir', default='pcaps',
                        help='Directory with PCAP/PCAPNG files (default: pcaps)')
    parser.add_argument('--interface', help='Network interface for live capture')
    parser.add_argument('--count',    type=int, default=1000,
                        help='Packets to capture in live mode (default: 1000)')
    parser.add_argument('--duration', type=int,
                        help='Duration in seconds for live capture (optional)')
    parser.add_argument('--label',    help='Label for captured flows in live mode')
    parser.add_argument('--output',   default='dataset/dataset.csv',
                        help='Output CSV file (default: dataset/dataset.csv)')

    args = parser.parse_args()

    if args.mode == 'pcap':
        if args.input:
            label    = os.path.splitext(os.path.basename(args.input))[0]
            features = extract_features_from_pcap(args.input, label=label)
            write_to_csv(features, args.output, mode='w', write_header=True)
        else:
            process_pcap_folder(args.pcap_dir, args.output)

    elif args.mode == 'live':
        if not args.interface:
            logger.error("--interface is required for live capture mode")
            parser.print_help()
            return
        extract_features_from_interface(
            interface=args.interface,
            output_csv=args.output,
            label=args.label,
            packet_count=args.count,
            duration=args.duration,
        )


if __name__ == '__main__':
    main()