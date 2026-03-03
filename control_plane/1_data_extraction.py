#!/usr/bin/env python3
"""
Network Traffic Flow Feature Extractor
Extracts flow-based features from PCAP files or live network capture.

Output columns per flow (bidirectional, canonical key):

  Identifier columns (not used as ML features):
    SrcIP     - Canonical source IP
    DstIP     - Canonical destination IP
    SrcPort   - Canonical source port
    DstPort   - Canonical destination port
    Protocol  - IP protocol number (6=TCP, 17=UDP)

  ML feature columns — match EXACTLY what the P4 switch computes and
  sends to the pca_component* tables (same names, same order):
    Duration     - Time from first to last packet (nanoseconds)
    MaxIAT       - Maximum inter-arrival time across consecutive packets (ns)
    UrgCount     - Number of packets with URG flag set
    FwdPktCount  - Packet count in forward (canonical) direction
    BwdPktCount  - Packet count in backward (reverse) direction
    FwdBytes     - Total bytes in forward direction
    BwdBytes     - Total bytes in backward direction
    MaxWinSize   - Maximum TCP window size observed in flow (0 for UDP)

Direction convention (mirrors P4 compute_flow_hash() exactly):
  "Forward"  (is_reverse=0) = packets whose src_ip is the numerically smaller IP
  "Backward" (is_reverse=1) = packets from the other endpoint.
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
    Returns (canon_src_ip, canon_src_port, canon_dst_ip, canon_dst_port,
             protocol, is_reverse).
    """
    src_int = int(ipaddress.ip_address(src_ip))
    dst_int = int(ipaddress.ip_address(dst_ip))

    if src_int < dst_int:
        return src_ip, src_port, dst_ip, dst_port, protocol, False
    elif src_int > dst_int:
        return dst_ip, dst_port, src_ip, src_port, protocol, True
    else:
        if src_port <= dst_port:
            return src_ip, src_port, dst_ip, dst_port, protocol, False
        else:
            return dst_ip, dst_port, src_ip, src_port, protocol, True


# ---------------------------------------------------------------------------
# Feature finalisation
# ---------------------------------------------------------------------------

def _finalize_flow(flow_key, flow_data, label):
    """
    Convert accumulated flow state into a feature dict.
    flow_key = (canon_src_ip, canon_src_port, canon_dst_ip, canon_dst_port, protocol)
    """
    timestamps = flow_data['timestamps']
    if len(timestamps) < 2:
        return None

    duration_ns = timestamps[-1] - timestamps[0]

    # MaxIAT: maximum consecutive inter-arrival time (mirrors P4 reg_max_iat)
    max_iat_ns = 0
    for i in range(1, len(timestamps)):
        iat = timestamps[i] - timestamps[i - 1]
        if iat > max_iat_ns:
            max_iat_ns = iat

    feature = {
        # --- Identifier columns (not ML features) ---
        "SrcIP":       flow_key[0],
        "DstIP":       flow_key[2],
        "SrcPort":     flow_key[1],
        "DstPort":     flow_key[3],
        "Protocol":    flow_key[4],
        # --- ML features in P4 table key order ---
        "Duration":    duration_ns,
        "MaxIAT":      max_iat_ns,
        "UrgCount":    flow_data['urg_count'],
        "FwdPktCount": flow_data['fwd_pkt_count'],
        "BwdPktCount": flow_data['bwd_pkt_count'],
        "FwdBytes":    flow_data['fwd_bytes'],
        "BwdBytes":    flow_data['bwd_bytes'],
        "MaxWinSize":  flow_data['max_win_size'],
    }
    if label is not None:
        feature["Label"] = label
    return feature


def _empty_flow_state():
    return {
        'timestamps':    [],
        'syn_count':     0,
        'ack_count':     0,
        'fin_count':     0,
        'rst_count':     0,
        'urg_count':     0,
        'fwd_pkt_count': 0,
        'bwd_pkt_count': 0,
        'fwd_bytes':     0,
        'bwd_bytes':     0,
        'hdr_len':       0,
        'max_win_size':  0,
    }


def _update_flow_state(state, timestamp_ns, pkt_len, pkt_hdr_len,
                        is_reverse, flags_syn, flags_ack, flags_fin,
                        flags_rst, flags_urg, win_size):
    """Accumulate a single packet into the flow state."""
    state['timestamps'].append(timestamp_ns)
    state['syn_count'] += flags_syn
    state['ack_count'] += flags_ack
    state['fin_count'] += flags_fin
    state['rst_count'] += flags_rst
    state['urg_count'] += flags_urg
    state['hdr_len']   += pkt_hdr_len
    if win_size > state['max_win_size']:
        state['max_win_size'] = win_size

    if is_reverse:
        state['bwd_pkt_count'] += 1
        state['bwd_bytes']     += pkt_len
    else:
        state['fwd_pkt_count'] += 1
        state['fwd_bytes']     += pkt_len


# ---------------------------------------------------------------------------
# Packet field extraction helper
# ---------------------------------------------------------------------------

def _extract_packet_fields(packet):
    """Extract all needed fields from a pyshark packet. Returns dict or None."""
    if not hasattr(packet, 'ip'):
        return None

    src_ip   = packet.ip.src
    dst_ip   = packet.ip.dst
    protocol = int(packet.ip.proto)
    ip_hdr_len = int(packet.ip.hdr_len) if hasattr(packet.ip, 'hdr_len') else 20

    if protocol == 6:   # TCP
        src_port    = int(packet.tcp.srcport)
        dst_port    = int(packet.tcp.dstport)
        flags_syn   = int(packet.tcp.flags_syn)   if hasattr(packet.tcp, 'flags_syn')   else 0
        flags_ack   = int(packet.tcp.flags_ack)   if hasattr(packet.tcp, 'flags_ack')   else 0
        flags_fin   = int(packet.tcp.flags_fin)   if hasattr(packet.tcp, 'flags_fin')   else 0
        flags_rst   = int(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
        flags_urg   = int(packet.tcp.flags_urg)   if hasattr(packet.tcp, 'flags_urg')   else 0
        l4_hdr_len  = int(packet.tcp.hdr_len)     if hasattr(packet.tcp, 'hdr_len')     else 20
        # window_size_value is the unscaled value (matches the raw TCP header field)
        win_size    = int(packet.tcp.window_size_value) if hasattr(packet.tcp, 'window_size_value') else \
                      (int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0)
    elif protocol == 17:  # UDP
        src_port    = int(packet.udp.srcport)
        dst_port    = int(packet.udp.dstport)
        flags_syn = flags_ack = flags_fin = flags_rst = flags_urg = 0
        l4_hdr_len  = 8
        win_size    = 0
    else:
        return None

    return {
        'src_ip': src_ip, 'dst_ip': dst_ip, 'protocol': protocol,
        'src_port': src_port, 'dst_port': dst_port,
        'flags_syn': flags_syn, 'flags_ack': flags_ack,
        'flags_fin': flags_fin, 'flags_rst': flags_rst,
        'flags_urg': flags_urg,
        'pkt_hdr_len': ip_hdr_len + l4_hdr_len,
        'pkt_len': int(packet.length),
        'timestamp_ns': int(float(packet.sniff_timestamp) * 1_000_000_000),
        'win_size': win_size,
    }


# ---------------------------------------------------------------------------
# PCAP file extraction
# ---------------------------------------------------------------------------

def extract_features_from_pcap(pcap_path, label=None):
    features_list = []
    cap = pyshark.FileCapture(pcap_path)
    flows = defaultdict(lambda: None)
    packet_count = 0

    try:
        for packet in cap:
            try:
                fields = _extract_packet_fields(packet)
                if fields is None:
                    continue

                c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, is_reverse = \
                    make_canonical_key(fields['src_ip'], fields['src_port'],
                                       fields['dst_ip'], fields['dst_port'],
                                       fields['protocol'])
                flow_key = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto)

                if flows[flow_key] is None:
                    flows[flow_key] = _empty_flow_state()
                else:
                    state = flows[flow_key]
                    if state['timestamps'] and \
                            (fields['timestamp_ns'] - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                        feat = _finalize_flow(flow_key, state, label)
                        if feat:
                            features_list.append(feat)
                        flows[flow_key] = _empty_flow_state()

                _update_flow_state(flows[flow_key], fields['timestamp_ns'],
                                   fields['pkt_len'], fields['pkt_hdr_len'],
                                   is_reverse, fields['flags_syn'], fields['flags_ack'],
                                   fields['flags_fin'], fields['flags_rst'],
                                   fields['flags_urg'], fields['win_size'])

                packet_count += 1
                if packet_count % 1000 == 0:
                    logger.info(f"Processed {packet_count} packets, "
                                f"{len(flows)} flows from {os.path.basename(pcap_path)}")
            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
                continue
    finally:
        cap.close()

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


def extract_base_label(filename):
    base  = os.path.splitext(filename)[0]
    parts = base.split('.')
    for i, part in enumerate(parts):
        if part.startswith('v') and part[1:].isdigit():
            return '.'.join(parts[:i])
    return base


def process_pcap_folder(folder_path, output_csv):
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


def extract_features_from_interface(interface, output_csv, label=None,
                                     packet_count=1000, duration=None):
    logger.info(f"Starting live capture on interface: {interface}")
    flows         = defaultdict(lambda: None)
    features_list = []
    cap           = pyshark.LiveCapture(interface=interface)
    captured      = 0
    start_time    = time.time()

    try:
        for packet in cap.sniff_continuously():
            try:
                if duration and (time.time() - start_time) >= duration:
                    break
                if captured >= packet_count:
                    break
                fields = _extract_packet_fields(packet)
                if fields is None:
                    continue

                c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, is_reverse = \
                    make_canonical_key(fields['src_ip'], fields['src_port'],
                                       fields['dst_ip'], fields['dst_port'],
                                       fields['protocol'])
                flow_key = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto)

                if flows[flow_key] is None:
                    flows[flow_key] = _empty_flow_state()
                else:
                    state = flows[flow_key]
                    if state['timestamps'] and \
                            (fields['timestamp_ns'] - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                        feat = _finalize_flow(flow_key, state, label)
                        if feat:
                            features_list.append(feat)
                        flows[flow_key] = _empty_flow_state()

                _update_flow_state(flows[flow_key], fields['timestamp_ns'],
                                   fields['pkt_len'], fields['pkt_hdr_len'],
                                   is_reverse, fields['flags_syn'], fields['flags_ack'],
                                   fields['flags_fin'], fields['flags_rst'],
                                   fields['flags_urg'], fields['win_size'])
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract bidirectional flow features from PCAP/PCAPNG files or live capture')
    parser.add_argument('--mode', choices=['pcap', 'live'], required=True)
    parser.add_argument('--input',    help='Input PCAP file (single file mode)')
    parser.add_argument('--pcap-dir', default='pcaps')
    parser.add_argument('--interface', help='Network interface for live capture')
    parser.add_argument('--count',    type=int, default=1000)
    parser.add_argument('--duration', type=int)
    parser.add_argument('--label',    help='Label for captured flows in live mode')
    parser.add_argument('--output',   default='dataset/dataset.csv')
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
            return
        extract_features_from_interface(
            interface=args.interface, output_csv=args.output,
            label=args.label, packet_count=args.count, duration=args.duration)


if __name__ == '__main__':
    main()