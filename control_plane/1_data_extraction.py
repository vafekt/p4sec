#!/usr/bin/env python3
"""
Network Traffic Flow Feature Extractor
Extracts flow-based features from PCAP files or live network capture:
  1. IAT - Average Inter-Arrival Time within a flow (nanoseconds)
  2. Duration - Total duration of the flow (nanoseconds)
  3. Source Port - Source port of the flow
  4. Destination Port - Destination port of the flow
  5. Total Bytes - Total bytes transferred in the flow
  6. Flags - TCP flags (SYN, ACK, FIN, RST) aggregated over the flow. Each flag is represented as one feature (0 or 1).

Supports both PCAP file processing and live interface capture.
Aggregates packets into flows using 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).
"""

import os
import glob
import pyshark
import csv
import argparse
import logging
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FLOW_TIMEOUT_NS = 120 * 1_000_000_000  # 120 seconds in nanoseconds - timeout for inactive flows


def extract_features_from_pcap(pcap_path, label=None):
    """
    Extract flow-based features from a PCAP file.
    
    Aggregates packets into flows using 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).
    For each flow, extracts:
      - IAT: Average Inter-Arrival Time between packets in the flow
      - Duration: Time from first to last packet in flow
      - SrcPort: Source port
      - DstPort: Destination port
      - TotalBytes: Total bytes in the flow
      - Flags: Aggregated TCP flags (SYN, ACK, FIN, RST)
    
    Returns list of feature dictionaries (one per flow).
    """
    features_list = []
    cap = pyshark.FileCapture(pcap_path)
    
    # Dictionary to store flow state: flow_key -> {packets_info}
    flows = defaultdict()
    flow_count = 0
    packet_count = 0

    def finalize_flow(flow_key, flow_data):
        if len(flow_data['timestamps']) < 2:
            return None

        timestamps = flow_data['timestamps']
        lengths = flow_data['lengths']

        # Duration: time from first to last packet (in nanoseconds)
        duration_ns = timestamps[-1] - timestamps[0]

        # IAT: sum of inter-arrival times >> 2 (matches P4 calculation)
        if len(timestamps) > 1:
            sum_iat_ns = 0
            for i in range(1, len(timestamps)):
                sum_iat_ns += (timestamps[i] - timestamps[i - 1])
            # Right-shift by 2 to match P4 approximation
            approx_iat_ns = sum_iat_ns >> 2
        else:
            approx_iat_ns = 0

        # Total bytes (match P4: sum of packet lengths as reported)
        total_bytes = sum(lengths)

        feature = {
            "IAT": approx_iat_ns,
            "Duration": duration_ns,
            "SrcPort": flow_key[2],
            "DstPort": flow_key[3],
            "TotalBytes": total_bytes,
            "FlagsSyn": flow_data['flags_syn'],
            "FlagsAck": flow_data['flags_ack'],
            "FlagsFin": flow_data['flags_fin'],
            "FlagsRst": flow_data['flags_rst'],
        }

        if label is not None:
            feature["Label"] = label

        return feature
    
    try:
        for packet in cap:
            try:
                # Extract IP layer information
                if not hasattr(packet, 'ip'):
                    continue
                
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                protocol = int(packet.ip.proto)
                
                # Extract port information (TCP/UDP only)
                if protocol == 6:  # TCP
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                    tcp_flags_syn = int(packet.tcp.flags_syn) if hasattr(packet.tcp, 'flags_syn') else 0
                    tcp_flags_ack = int(packet.tcp.flags_ack) if hasattr(packet.tcp, 'flags_ack') else 0
                    tcp_flags_fin = int(packet.tcp.flags_fin) if hasattr(packet.tcp, 'flags_fin') else 0
                    tcp_flags_rst = int(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
                elif protocol == 17:  # UDP
                    src_port = int(packet.udp.srcport)
                    dst_port = int(packet.udp.dstport)
                    tcp_flags_syn = 0
                    tcp_flags_ack = 0
                    tcp_flags_fin = 0
                    tcp_flags_rst = 0
                else:
                    continue
                
                # Create flow key (5-tuple)
                flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                
                # Get packet timestamp and length
                timestamp_ns = int(float(packet.sniff_timestamp) * 1_000_000_000)
                pkt_len = int(packet.length)
                
                # Initialize or update flow (apply flow timeout in nanoseconds)
                if flow_key not in flows:
                    flows[flow_key] = {
                        'packets': [],
                        'timestamps': [],
                        'lengths': [],
                        'flags_syn': 0,
                        'flags_ack': 0,
                        'flags_fin': 0,
                        'flags_rst': 0,
                    }
                    flow_count += 1
                else:
                    if flows[flow_key]['timestamps']:
                        last_ts = flows[flow_key]['timestamps'][-1]
                        if (timestamp_ns - last_ts) > FLOW_TIMEOUT_NS:
                            finalized = finalize_flow(flow_key, flows[flow_key])
                            if finalized:
                                features_list.append(finalized)
                            flows[flow_key] = {
                                'packets': [],
                                'timestamps': [],
                                'lengths': [],
                                'flags_syn': 0,
                                'flags_ack': 0,
                                'flags_fin': 0,
                                'flags_rst': 0,
                            }

                flows[flow_key]['packets'].append(pkt_len)
                flows[flow_key]['timestamps'].append(timestamp_ns)
                flows[flow_key]['lengths'].append(pkt_len)
                flows[flow_key]['flags_syn'] = max(flows[flow_key]['flags_syn'], tcp_flags_syn)
                flows[flow_key]['flags_ack'] = max(flows[flow_key]['flags_ack'], tcp_flags_ack)
                flows[flow_key]['flags_fin'] = max(flows[flow_key]['flags_fin'], tcp_flags_fin)
                flows[flow_key]['flags_rst'] = max(flows[flow_key]['flags_rst'], tcp_flags_rst)
                
                packet_count += 1
                if packet_count % 1000 == 0:
                    logger.info(f"Processed {packet_count} packets, {len(flows)} flows from {os.path.basename(pcap_path)}")
                    
            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
                continue
    finally:
        cap.close()
    
    # Convert remaining flows to feature records
    for flow_key, flow_data in flows.items():
        finalized = finalize_flow(flow_key, flow_data)
        if finalized:
            features_list.append(finalized)
    
    logger.info(f"Extracted {len(features_list)} flows from {packet_count} packets in {pcap_path}")
    return features_list


def write_to_csv(features_list, output_path, mode='w', write_header=True):
    """Write features to CSV file."""
    if not features_list:
        logger.warning("No features to write")
        return
    
    # Ensure output directory exists
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
    """Extract base label from filename, removing version suffixes like .v1, .v2, etc."""
    base = os.path.splitext(filename)[0]  # Remove .pcap extension
    # Remove version suffix (.v1, .v2, etc.)
    parts = base.split('.')
    # Keep only the part before the version (v followed by digits)
    for i, part in enumerate(parts):
        if part.startswith('v') and part[1:].isdigit():
            return '.'.join(parts[:i])
    return base


def extract_base_label(filename):
    """Extract base label from filename, removing version suffixes like .v1, .v2, etc."""
    base = os.path.splitext(filename)[0]  # Remove .pcap extension
    # Remove version suffix (.v1, .v2, etc.)
    parts = base.split('.')
    # Keep only the part before the version (v followed by digits)
    for i, part in enumerate(parts):
        if part.startswith('v') and part[1:].isdigit():
            return '.'.join(parts[:i])
    return base


def process_single_pcap(pcap_path, output_csv, label=None):
    """Process a single PCAP file and write to CSV."""
    logger.info(f"Processing PCAP file: {pcap_path}")
    
    features = extract_features_from_pcap(pcap_path, label=label)
    write_to_csv(features, output_csv, mode='w', write_header=True)


def process_pcap_folder(folder_path, output_csv):
    """Process all .pcap and .pcapng files in a folder and write combined labeled CSV."""
    pcap_files = sorted(glob.glob(os.path.join(folder_path, "*.pcap")))
    pcapng_files = sorted(glob.glob(os.path.join(folder_path, "*.pcapng")))
    all_files = pcap_files + pcapng_files
    
    if not all_files:
        logger.warning(f"No .pcap or .pcapng files found in {folder_path}")
        return
    
    logger.info(f"Found {len(all_files)} capture files in {folder_path} ({len(pcap_files)} .pcap, {len(pcapng_files)} .pcapng)")
    
    header_written = False
    
    for pcap_path in all_files:
        # Use filename (without extension and version suffix) as label
        filename = os.path.basename(pcap_path)
        label = extract_base_label(filename)
        logger.info(f"Processing {pcap_path} (label={label})")
        
        features = extract_features_from_pcap(pcap_path, label=label)
        
        # Write with header only on first file
        mode = 'w' if not header_written else 'a'
        write_to_csv(features, output_csv, mode=mode, write_header=not header_written)
        header_written = True
    
    logger.info(f"Finished processing all capture files. Output: {output_csv}")


def extract_features_from_interface(interface, output_csv, label=None, packet_count=1000, duration=None):
    """
    Extract flow-based features from live network capture on specified interface.
    
    Args:
        interface: Network interface name (e.g., 'eth0', 'wlan0')
        output_csv: Output CSV file path
        label: Optional label for captured flows
        packet_count: Number of packets to capture (default: 1000)
        duration: Optional time limit in seconds (default: None - use packet_count)
    """
    logger.info(f"Starting live capture on interface: {interface}")
    logger.info(f"Target packets: {packet_count}, Duration: {duration if duration else 'unlimited'}")
    
    flows = defaultdict()
    features_list = []
    cap = pyshark.LiveCapture(interface=interface)
    
    captured = 0
    start_time = time.time()

    def finalize_flow(flow_key, flow_data):
        if len(flow_data['timestamps']) < 2:
            return None

        timestamps = flow_data['timestamps']
        lengths = flow_data['lengths']

        duration_ns = timestamps[-1] - timestamps[0]

        if len(timestamps) > 1:
            sum_iat_ns = 0
            for i in range(1, len(timestamps)):
                sum_iat_ns += (timestamps[i] - timestamps[i - 1])
            approx_iat_ns = sum_iat_ns >> 2
        else:
            approx_iat_ns = 0

        total_bytes = sum(lengths)

        feature = {
            "IAT": approx_iat_ns,
            "Duration": duration_ns,
            "SrcPort": flow_key[2],
            "DstPort": flow_key[3],
            "TotalBytes": total_bytes,
            "FlagsSyn": flow_data['flags_syn'],
            "FlagsAck": flow_data['flags_ack'],
            "FlagsFin": flow_data['flags_fin'],
            "FlagsRst": flow_data['flags_rst'],
        }

        if label is not None:
            feature["Label"] = label

        return feature
    
    try:
        for packet in cap.sniff_continuously():
            try:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit of {duration}s reached")
                    break
                
                # Check packet count limit
                if captured >= packet_count:
                    logger.info(f"Captured {captured} packets")
                    break
                
                # Extract IP layer information
                if not hasattr(packet, 'ip'):
                    continue
                
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                protocol = int(packet.ip.proto)
                
                # Extract port information (TCP/UDP only)
                if protocol == 6:  # TCP
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                    tcp_flags_syn = int(packet.tcp.flags_syn) if hasattr(packet.tcp, 'flags_syn') else 0
                    tcp_flags_ack = int(packet.tcp.flags_ack) if hasattr(packet.tcp, 'flags_ack') else 0
                    tcp_flags_fin = int(packet.tcp.flags_fin) if hasattr(packet.tcp, 'flags_fin') else 0
                    tcp_flags_rst = int(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
                elif protocol == 17:  # UDP
                    src_port = int(packet.udp.srcport)
                    dst_port = int(packet.udp.dstport)
                    tcp_flags_syn = 0
                    tcp_flags_ack = 0
                    tcp_flags_fin = 0
                    tcp_flags_rst = 0
                else:
                    continue
                
                # Create flow key (5-tuple)
                flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                
                # Get packet timestamp and length
                timestamp_ns = int(float(packet.sniff_timestamp) * 1_000_000_000)
                pkt_len = int(packet.length)
                
                # Initialize or update flow (apply flow timeout in nanoseconds)
                if flow_key not in flows:
                    flows[flow_key] = {
                        'packets': [],
                        'timestamps': [],
                        'lengths': [],
                        'flags_syn': 0,
                        'flags_ack': 0,
                        'flags_fin': 0,
                        'flags_rst': 0,
                    }
                else:
                    if flows[flow_key]['timestamps']:
                        last_ts = flows[flow_key]['timestamps'][-1]
                        if (timestamp_ns - last_ts) > FLOW_TIMEOUT_NS:
                            finalized = finalize_flow(flow_key, flows[flow_key])
                            if finalized:
                                features_list.append(finalized)
                            flows[flow_key] = {
                                'packets': [],
                                'timestamps': [],
                                'lengths': [],
                                'flags_syn': 0,
                                'flags_ack': 0,
                                'flags_fin': 0,
                                'flags_rst': 0,
                            }

                flows[flow_key]['packets'].append(pkt_len)
                flows[flow_key]['timestamps'].append(timestamp_ns)
                flows[flow_key]['lengths'].append(pkt_len)
                flows[flow_key]['flags_syn'] = max(flows[flow_key]['flags_syn'], tcp_flags_syn)
                flows[flow_key]['flags_ack'] = max(flows[flow_key]['flags_ack'], tcp_flags_ack)
                flows[flow_key]['flags_fin'] = max(flows[flow_key]['flags_fin'], tcp_flags_fin)
                flows[flow_key]['flags_rst'] = max(flows[flow_key]['flags_rst'], tcp_flags_rst)
                
                captured += 1
                
                if captured % 100 == 0:
                    logger.info(f"Captured {captured} packets, {len(flows)} flows...")
                    
            except Exception as e:
                logger.debug(f"Error processing packet: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("\nCapture interrupted by user")
    finally:
        cap.close()
    
    # Convert remaining flows to feature records
    for flow_key, flow_data in flows.items():
        finalized = finalize_flow(flow_key, flow_data)
        if finalized:
            features_list.append(finalized)
    
    logger.info(f"Extracted {len(features_list)} flows from {captured} packets")
    
    # Write to CSV
    if features_list:
        write_to_csv(features_list, output_csv, mode='w', write_header=True)
    else:
        logger.warning("No flows captured")
    
    return features_list


def main():
    parser = argparse.ArgumentParser(
        description='Extract flow-based features from PCAP/PCAPNG files or live capture'
    )
    parser.add_argument(
        '--mode',
        choices=['pcap', 'live'],
        required=True,
        help='Processing mode: pcap (file processing) or live (interface capture)'
    )
    parser.add_argument(
        '--input',
        help='Input PCAP file (for single file processing)'
    )
    parser.add_argument(
        '--pcap-dir',
        default='pcaps',
        help='Directory containing PCAP/PCAPNG files (default: pcaps)'
    )
    parser.add_argument(
        '--interface',
        help='Network interface for live capture (e.g., eth0, wlan0)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1000,
        help='Number of packets to capture in live mode (default: 1000)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Duration in seconds for live capture (optional)'
    )
    parser.add_argument(
        '--label',
        help='Label for captured flows in live mode'
    )
    parser.add_argument(
        '--output',
        default='dataset/dataset.csv',
        help='Output CSV file (default: dataset/dataset.csv)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'pcap':
        if args.input:
            # Single PCAP file
            label = os.path.splitext(os.path.basename(args.input))[0]
            features = extract_features_from_pcap(args.input, label=label)
            write_to_csv(features, args.output, mode='w', write_header=True)
        else:
            # Folder of PCAP files
            process_pcap_folder(args.pcap_dir, args.output)
    
    elif args.mode == 'live':
        if not args.interface:
            logger.error("--interface is required for live capture mode")
            parser.print_help()
            return
        
        # Live capture from network interface
        extract_features_from_interface(
            interface=args.interface,
            output_csv=args.output,
            label=args.label,
            packet_count=args.count,
            duration=args.duration
        )


if __name__ == '__main__':
    main()
