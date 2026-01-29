#!/usr/bin/env python3
"""
Simplified Network Traffic Feature Extractor
Extracts 3 simple features from PCAP files or live network capture:
  1. IAT (Inter Arrival Time) - time difference between consecutive packets (nanoseconds)
  2. Packet Length - size of current packet
  3. Diff Length - difference between current and previous packet length

Supports both PCAP file processing and live interface capture.
"""

import os
import glob
import pyshark
import csv
import argparse
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features_from_pcap(pcap_path, label=None):
    """
    Extract IAT, packet length, and diff length from a PCAP file.
    
    For each packet:
      - IAT: Inter Arrival Time (nanoseconds) between consecutive packets
      - PacketLength: Size of the current packet
      - DiffLength: Difference in packet length from previous packet
    
    Returns list of feature dictionaries.
    """
    features_list = []
    cap = pyshark.FileCapture(pcap_path)
    
    last_ts = 0
    last_len = 0
    packet_count = 0
    
    try:
        for packet in cap:
            try:
                # Get timestamp and packet length
                ts = float(packet.sniff_timestamp)
                ip_len = int(packet.length)
                
                # Convert timestamp to nanoseconds
                ts_ns = int(ts * 1000000 * 1000)
                
                # First packet - initialize baseline
                if last_ts == 0:
                    last_ts = ts_ns
                    last_len = ip_len
                    packet_count += 1
                    continue
                
                # Calculate IAT (Inter Arrival Time)
                iat = ts_ns - last_ts
                
                # Skip packets with negative IAT (out of order)
                if iat < 0:
                    logger.warning(f"Ignoring unordered packet at {ts}")
                    continue
                
                # Calculate diff length: difference from previous packet
                diff_len = ip_len - last_len
                diff_len += 0xFFFF  # Avoid negative values (add 65535)
                
                # Create feature record
                feature = {
                    "IAT": iat,
                    "PacketLength": ip_len,
                    "DiffLength": diff_len
                }
                
                if label is not None:
                    feature["Label"] = label
                
                features_list.append(feature)
                
                # Update for next iteration
                last_ts = ts_ns
                last_len = ip_len
                packet_count += 1
                
                if packet_count % 1000 == 0:
                    logger.info(f"Processed {packet_count} packets from {os.path.basename(pcap_path)}")
                    
            except Exception as e:
                logger.warning(f"Error processing packet: {e}")
                continue
    finally:
        cap.close()
    
    logger.info(f"Extracted {len(features_list)} features from {pcap_path}")
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


def process_single_pcap(pcap_path, output_csv, label=None):
    """Process a single PCAP file and write to CSV."""
    logger.info(f"Processing PCAP file: {pcap_path}")
    
    features = extract_features_from_pcap(pcap_path, label=label)
    write_to_csv(features, output_csv, mode='w', write_header=True)


def process_pcap_folder(folder_path, output_csv):
    """Process all .pcap files in a folder and write combined labeled CSV."""
    pcap_files = sorted(glob.glob(os.path.join(folder_path, "*.pcap")))
    
    if not pcap_files:
        logger.warning(f"No .pcap files found in {folder_path}")
        return
    
    logger.info(f"Found {len(pcap_files)} PCAP files in {folder_path}")
    
    header_written = False
    
    for pcap_path in pcap_files:
        # Use filename (without extension and version suffix) as label
        filename = os.path.basename(pcap_path)
        label = extract_base_label(filename)
        logger.info(f"Processing {pcap_path} (label={label})")
        
        features = extract_features_from_pcap(pcap_path, label=label)
        
        # Write with header only on first file
        mode = 'w' if not header_written else 'a'
        write_to_csv(features, output_csv, mode=mode, write_header=not header_written)
        header_written = True
    
    logger.info(f"Finished processing all PCAP files. Output: {output_csv}")


def extract_features_from_interface(interface, output_csv, label=None, packet_count=1000, duration=None):
    """
    Extract features from live network capture on specified interface.
    
    Args:
        interface: Network interface name (e.g., 'eth0', 'wlan0')
        output_csv: Output CSV file path
        label: Optional label for captured packets
        packet_count: Number of packets to capture (default: 1000)
        duration: Optional time limit in seconds (default: None - use packet_count)
    """
    logger.info(f"Starting live capture on interface: {interface}")
    logger.info(f"Target packets: {packet_count}, Duration: {duration if duration else 'unlimited'}")
    
    features_list = []
    cap = pyshark.LiveCapture(interface=interface)
    
    last_ts = 0
    last_len = 0
    captured = 0
    start_time = time.time()
    
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
                
                # Get timestamp and packet length
                ts = float(packet.sniff_timestamp)
                ip_len = int(packet.length)
                
                # Convert timestamp to nanoseconds
                ts_ns = int(ts * 1000000 * 1000)
                
                # First packet - initialize baseline
                if last_ts == 0:
                    last_ts = ts_ns
                    last_len = ip_len
                    captured += 1
                    continue
                
                # Calculate IAT (Inter Arrival Time)
                iat = ts_ns - last_ts
                
                # Skip packets with negative IAT (out of order)
                if iat < 0:
                    logger.warning(f"Ignoring unordered packet at {ts}")
                    continue
                
                # Calculate diff length: difference from previous packet
                diff_len = ip_len - last_len
                diff_len += 0xFFFF  # Avoid negative values (add 65535)
                
                # Create feature record
                feature = {
                    "IAT": iat,
                    "PacketLength": ip_len,
                    "DiffLength": diff_len
                }
                
                if label is not None:
                    feature["Label"] = label
                
                features_list.append(feature)
                
                # Update for next iteration
                last_ts = ts_ns
                last_len = ip_len
                captured += 1
                
                if captured % 100 == 0:
                    logger.info(f"Captured {captured} packets...")
                    
            except Exception as e:
                logger.warning(f"Error processing packet: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("\nCapture interrupted by user")
    finally:
        cap.close()
    
    logger.info(f"Extracted {len(features_list)} features from live capture")
    
    # Write to CSV
    if features_list:
        write_to_csv(features_list, output_csv, mode='w', write_header=True)
    else:
        logger.warning("No features captured")
    
    return features_list


def main():
    parser = argparse.ArgumentParser(
        description='Extract IAT, Packet Length, and Diff Length features from PCAP files or live capture'
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
        help='Directory containing PCAP files (default: pcaps)'
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
        help='Label for captured packets in live mode'
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
            process_single_pcap(args.input, args.output, label=label)
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
