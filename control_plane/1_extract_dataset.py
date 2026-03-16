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
  sends as a digest when a flow ends (same names, same order):
    Duration     - Time from first to last packet (nanoseconds)
    MaxIAT       - Maximum inter-arrival time across consecutive packets (ns)
    UrgCount     - Number of packets with URG flag set
    FwdPktCount  - Packet count in forward (canonical) direction
    BwdPktCount  - Packet count in backward (reverse) direction
    FwdBytes     - Total bytes in forward direction
    BwdBytes     - Total bytes in backward direction
    MaxWinSize   - Maximum TCP window size observed in flow (0 for UDP)
    FlagsSyn     - Number of packets with SYN flag set (0 for UDP)
    FlagsAck     - Number of packets with ACK flag set (0 for UDP)
    FlagsFin     - Number of packets with FIN flag set (0 for UDP)
    FlagsRst     - Number of packets with RST flag set (0 for UDP)

NOTE: Output is one row PER FLOW, emitted when a flow ends:
  - TCP FIN or RST flag is seen (flow closed)
  - Flow has been idle for 120 s (timeout)
  - End of PCAP file
This mirrors P4's classify-and-digest behavior which is gated on
meta.flow_ended == 1, so training data and live inference see the
same complete-flow feature vectors.
Only flows with at least 2 packets are written (Duration > 0).

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
import sys
import logging
import time
from collections import defaultdict
from decimal import Decimal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FLOW_TIMEOUT_NS = 120 * 1_000_000_000  # 120 seconds in nanoseconds

LOGO = """---------------------------------------------------------------------------
------PPPPPPPP------4444------SSSSSSSS------EEEEEEEE------CCCCCCCC---------
------PP-----PP----44--44----SS-------------EE------------CC---------------
------PP-----PP---44---44----SS-------------EE------------CC---------------
------PPPPPPPP---44----44-----SSSSSS--------EEEEEEE-------CC---------------
------PP---------444444444----------SS------EE------------CC---------------
------PP---------------44-----------SS------EE------------CC---------------
------PP---------------44----SSSSSSSS-------EEEEEEEE------CCCCCCCC---------
---------------------------------------------------------------------------"""


class P4secArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        if file is None:
            file = sys.stdout
        print(LOGO, file=file)
        super().print_help(file)


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
# Flow state management  (mirrors P4 registers + update_flow_state action)
# ---------------------------------------------------------------------------

def _finalize_flow(flow_key, flow_data, label):
    """
    Convert completed flow state into a feature dict.
    Mirrors P4's classify-and-digest step that runs when flow_ended==1.
    Only emits a row when the flow has at least 2 packets (Duration > 0).
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
        'SrcIP':       flow_key[0],
        'DstIP':       flow_key[2],
        'SrcPort':     flow_key[1],
        'DstPort':     flow_key[3],
        'Protocol':    flow_key[4],
        # --- ML features in P4 table key order ---
        'Duration':    duration_ns,
        'MaxIAT':      max_iat_ns,
        'UrgCount':    flow_data['urg_count'],
        'FwdPktCount': flow_data['fwd_pkt_count'],
        'BwdPktCount': flow_data['bwd_pkt_count'],
        'FwdBytes':    flow_data['fwd_bytes'],
        'BwdBytes':    flow_data['bwd_bytes'],
        'MaxWinSize':  flow_data['max_win_size'],
        'FlagsSyn':    flow_data['flags_syn_count'],
        'FlagsAck':    flow_data['flags_ack_count'],
        'FlagsFin':    flow_data['flags_fin_count'],
        'FlagsRst':    flow_data['flags_rst_count'],
    }
    if label is not None:
        feature['Label'] = label
    return feature


def _empty_flow_state():
    """
    Initialise a blank flow state — mirrors P4 register initial values (all zero).
    """
    return {
        'timestamps':     [],
        'urg_count':      0,
        'fwd_pkt_count':  0,
        'bwd_pkt_count':  0,
        'fwd_bytes':      0,
        'bwd_bytes':      0,
        'max_win_size':   0,
        'flags_syn_count': 0,  # count of packets with SYN set
        'flags_ack_count': 0,  # count of packets with ACK set
        'flags_fin_count': 0,  # count of packets with FIN set (also triggers flow end)
        'flags_rst_count': 0,  # count of packets with RST set (also triggers flow end)
    }


def _update_flow_state(state, timestamp_ns, pkt_len, is_reverse,
                        flags_syn, flags_ack, flags_fin, flags_rst, flags_urg, win_size):
    """Accumulate a single packet into the flow state. Returns True if the flow
    should be finalized after this packet (FIN or RST seen), False otherwise."""
    state['timestamps'].append(timestamp_ns)
    state['urg_count'] += flags_urg
    if win_size > state['max_win_size']:
        state['max_win_size'] = win_size
    # Accumulate flag counts (mirrors P4 register += 1 logic)
    state['flags_syn_count'] += flags_syn
    state['flags_ack_count'] += flags_ack
    state['flags_fin_count'] += flags_fin
    state['flags_rst_count'] += flags_rst

    if is_reverse:
        state['bwd_pkt_count'] += 1
        state['bwd_bytes']     += pkt_len
    else:
        state['fwd_pkt_count'] += 1
        state['fwd_bytes']     += pkt_len

    # Signal flow-end when TCP FIN or RST is seen (mirrors P4 flow_ended logic)
    return bool(flags_fin or flags_rst)


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

    if protocol == 6:   # TCP
        src_port    = int(packet.tcp.srcport)
        dst_port    = int(packet.tcp.dstport)
        flags_syn   = int(packet.tcp.flags_syn)   if hasattr(packet.tcp, 'flags_syn')   else 0
        flags_ack   = int(packet.tcp.flags_ack)   if hasattr(packet.tcp, 'flags_ack')   else 0
        flags_fin   = int(packet.tcp.flags_fin)   if hasattr(packet.tcp, 'flags_fin')   else 0
        flags_rst   = int(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
        flags_urg   = int(packet.tcp.flags_urg)   if hasattr(packet.tcp, 'flags_urg')   else 0
        # window_size_value is the unscaled value (matches the raw TCP header field)
        win_size    = int(packet.tcp.window_size_value) if hasattr(packet.tcp, 'window_size_value') else \
                      (int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0)
    elif protocol == 17:  # UDP
        src_port    = int(packet.udp.srcport)
        dst_port    = int(packet.udp.dstport)
        flags_syn = flags_ack = flags_fin = flags_rst = flags_urg = 0
        win_size    = 0
    else:
        return None

    return {
        'src_ip': src_ip, 'dst_ip': dst_ip, 'protocol': protocol,
        'src_port': src_port, 'dst_port': dst_port,
        'flags_syn': flags_syn, 'flags_ack': flags_ack,
        'flags_fin': flags_fin, 'flags_rst': flags_rst,
        'flags_urg': flags_urg,
        'pkt_len': int(packet.length),
        # P4 ingress_global_timestamp has microsecond granularity (*1000 → ns).
        # Truncate to microseconds here so training features match inference features
        # exactly. Using Decimal avoids float precision loss for large Unix timestamps.
        'timestamp_ns': int(Decimal(str(packet.sniff_timestamp)) * 1_000_000) * 1000,
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

                state = flows[flow_key]

                # Timeout: finalize the previous flow, then reset
                if state is not None and state['timestamps'] and \
                        (fields['timestamp_ns'] - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                    feat = _finalize_flow(flow_key, state, label)
                    if feat:
                        features_list.append(feat)
                    flows[flow_key] = _empty_flow_state()
                elif state is None:
                    flows[flow_key] = _empty_flow_state()

                # Accumulate packet; check if flow ends (FIN/RST)
                flow_ended = _update_flow_state(
                    flows[flow_key],
                    fields['timestamp_ns'], fields['pkt_len'], is_reverse,
                    fields['flags_syn'], fields['flags_ack'],
                    fields['flags_fin'], fields['flags_rst'],
                    fields['flags_urg'], fields['win_size'])

                if flow_ended:
                    feat = _finalize_flow(flow_key, flows[flow_key], label)
                    if feat:
                        features_list.append(feat)
                    flows[flow_key] = None   # reset slot for next flow on same 5-tuple

                packet_count += 1
                if packet_count % 1000 == 0:
                    logger.info(f"Processed {packet_count} packets, "
                                f"{len(flows)} flows from {os.path.basename(pcap_path)}")
            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
                continue
    finally:
        cap.close()

    # Finalize all open flows at end of file (mirrors controller reading
    # registers at capture end, or flow aging in production)
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

                state = flows[flow_key]

                # Timeout: finalize the previous flow, then reset
                if state is not None and state['timestamps'] and \
                        (fields['timestamp_ns'] - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                    feat = _finalize_flow(flow_key, state, label)
                    if feat:
                        features_list.append(feat)
                    flows[flow_key] = _empty_flow_state()
                elif state is None:
                    flows[flow_key] = _empty_flow_state()

                # Accumulate packet; check if flow ends (FIN/RST)
                flow_ended = _update_flow_state(
                    flows[flow_key],
                    fields['timestamp_ns'], fields['pkt_len'], is_reverse,
                    fields['flags_syn'], fields['flags_ack'],
                    fields['flags_fin'], fields['flags_rst'],
                    fields['flags_urg'], fields['win_size'])

                if flow_ended:
                    feat = _finalize_flow(flow_key, flows[flow_key], label)
                    if feat:
                        features_list.append(feat)
                    flows[flow_key] = None

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

    # Finalize all open flows at end of capture
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
    parser = P4secArgumentParser(
        description='Extract bidirectional flow features from PCAP/PCAPNG files or live capture',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Mode-specific options:\n"
            "  pcap : --input or --pcap-dir\n"
            "  live : --interface (required), --count, --duration, --label\n"
        )
    )
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
