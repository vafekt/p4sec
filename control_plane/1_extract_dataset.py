#!/usr/bin/env python3
"""
Network Traffic Flow Feature Extractor
Extracts flow-based features from PCAP files or live network capture.

Output columns per flow (bidirectional, canonical key):

  ML feature columns — all 20 match EXACTLY what the P4 switch computes and
  sends as a digest when a flow ends (same names, same order).
  SrcIP, DstIP are stored as 32-bit integer representations of the IP address.
  SrcMAC, DstMAC are stored as 48-bit integer representations.
  Protocol, SrcPort, DstPort are part of the 20-feature vector; PCA/LDA/AE
  operate on the full feature set.
    SrcIP              - Canonical source IP as 32-bit integer
    DstIP              - Canonical destination IP as 32-bit integer
    SrcMAC             - Canonical source MAC as 48-bit integer
    DstMAC             - Canonical destination MAC as 48-bit integer
    Protocol           - IP protocol number (6=TCP, 17=UDP, 1=ICMP, 253=ARP pseudo)
    SrcPort            - Canonical source port (bit<16>)
    DstPort            - Canonical destination port (bit<16>)
    Duration           - Time from first to last packet (nanoseconds)
    MaxIAT             - Maximum inter-arrival time across consecutive packets (ns)
    MinIAT             - Minimum inter-arrival time across consecutive packets (ns)
    FwdPktCount        - Packet count in forward (canonical) direction
    BwdPktCount        - Packet count in backward (reverse) direction
    FwdBytes           - Total bytes in forward direction
    BwdBytes           - Total bytes in backward direction
    MaxWinSize         - Maximum TCP window size observed in flow (0 for non-TCP)
    FlagsSyn           - Number of packets with SYN flag set (0 for non-TCP)
    FlagsAck           - Number of packets with ACK flag set (0 for non-TCP)
    FlagsFin           - Number of packets with FIN flag set (0 for non-TCP)
    FlagsRst           - Number of packets with RST flag set (0 for non-TCP)
    FwdMaxPktLen       - Maximum packet length in forward direction (bytes)
    BwdMaxPktLen       - Maximum packet length in backward direction (bytes)
    FlagsPsh           - Number of packets with PSH flag set (0 for non-TCP)
    UrgCount           - Number of packets with URG flag set (0 for non-TCP)
    InitFwdWinBytes    - TCP window size of the first forward-direction packet (0 for non-TCP)

NOTE: Output is one row PER FLOW, emitted when a flow ends:
  - TCP FIN or RST flag is seen (flow closed)
  - Flow has been idle for 20 s (inactive timeout — matches P4 FLOW_TIMEOUT)
  - Flow has been active for 60 s (active timeout — matches P4 ACTIVE_TIMEOUT)
  - End of PCAP file
This mirrors P4's classify-and-digest behavior which is gated on
meta.flow_ended == 1, so training data and live inference see the
same complete-flow feature vectors.
Flows with 1 or more packets are written (matches Zeek, NetFlow, CICFlowMeter).

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
import struct
import crcmod.predefined
from decimal import Decimal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FLOW_TIMEOUT_NS = 20 * 1_000_000_000        # 20s idle timeout in nanoseconds (matches P4 FLOW_TIMEOUT)
ACTIVE_TIMEOUT_NS = 60 * 1_000_000_000      # 60s active timeout — force-export long-lived flows
MAX_REGISTER_ENTRIES = 65536            # matches P4 MAX_REGISTER_ENTRIES (CRC16 → 65536 slots)

# CRC functions matching BMv2's HashAlgorithm.crc16 / crc32
_crc16_fn = crcmod.predefined.mkCrcFun('crc-16')
_crc32_fn = crcmod.predefined.mkCrcFun('crc-32')


def _compute_flow_hashes(c_src_ip, c_dst_ip, c_src_port, c_dst_port, proto):
    """Compute CRC16 and CRC32 hashes matching P4's compute_flow_hash() exactly.

    P4 concatenates {canon_src_ip(32), canon_dst_ip(32), canon_src_port(16),
    canon_dst_port(16), protocol(8)} in big-endian order (13 bytes), then:
      h16 = crc16(data) % MAX_REGISTER_ENTRIES
      h32 = crc32(data) % MAX_REGISTER_ENTRIES
    """
    src_int = int(ipaddress.ip_address(c_src_ip))
    dst_int = int(ipaddress.ip_address(c_dst_ip))
    data = struct.pack('!IIHHB', src_int, dst_int, c_src_port, c_dst_port, proto)
    h16 = _crc16_fn(data) % MAX_REGISTER_ENTRIES
    h32 = _crc32_fn(data) % MAX_REGISTER_ENTRIES
    return h16, h32


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
# MAC address helper
# ---------------------------------------------------------------------------

def mac_to_int(mac_str):
    """Convert MAC address string 'aa:bb:cc:dd:ee:ff' to 48-bit integer."""
    return int(mac_str.replace(':', '').replace('-', ''), 16)


# ---------------------------------------------------------------------------
# Canonical key helper — MUST match P4 compute_flow_hash() logic exactly
# ---------------------------------------------------------------------------

def make_canonical_key(src_ip, src_port, dst_ip, dst_port, protocol,
                       src_mac='00:00:00:00:00:00', dst_mac='00:00:00:00:00:00'):
    """
    Normalise a 5-tuple so that both directions of a flow map to the same key.
    Returns (canon_src_ip, canon_src_port, canon_dst_ip, canon_dst_port,
             protocol, is_reverse, canon_src_mac, canon_dst_mac).
    """
    src_int = int(ipaddress.ip_address(src_ip))
    dst_int = int(ipaddress.ip_address(dst_ip))

    if src_int < dst_int:
        return src_ip, src_port, dst_ip, dst_port, protocol, False, src_mac, dst_mac
    elif src_int > dst_int:
        return dst_ip, dst_port, src_ip, src_port, protocol, True, dst_mac, src_mac
    else:
        if src_port <= dst_port:
            return src_ip, src_port, dst_ip, dst_port, protocol, False, src_mac, dst_mac
        else:
            return dst_ip, dst_port, src_ip, src_port, protocol, True, dst_mac, src_mac


# ---------------------------------------------------------------------------
# Flow state management  (mirrors P4 registers + update_flow_state action)
# ---------------------------------------------------------------------------

def _finalize_flow(flow_key, flow_data, label):
    """
    Convert completed flow state into a feature dict.
    Mirrors P4's classify-and-digest step that runs when flow_ended==1.
    Only emits a row when the flow has at least 2 packets (Duration > 0).
    flow_key = (canon_src_ip, canon_src_port, canon_dst_ip, canon_dst_port,
                protocol, canon_src_mac, canon_dst_mac)
    """
    timestamps = flow_data['timestamps']
    if len(timestamps) < 1:
        return None

    duration_ns = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0

    # MaxIAT / MinIAT: consecutive inter-arrival time bounds
    # (mirrors P4 reg_max_iat / reg_min_iat).
    # MinIAT is 0 when fewer than 2 packets are seen.
    max_iat_ns = 0
    min_iat_ns = 0
    for i in range(1, len(timestamps)):
        iat = timestamps[i] - timestamps[i - 1]
        if iat > max_iat_ns:
            max_iat_ns = iat
        if min_iat_ns == 0 or iat < min_iat_ns:
            min_iat_ns = iat

    src_ip_int = int(ipaddress.ip_address(flow_key[0]))
    dst_ip_int = int(ipaddress.ip_address(flow_key[2]))
    src_mac_int = mac_to_int(flow_key[5])
    dst_mac_int = mac_to_int(flow_key[6])

    feature = {
        'SrcIP':       src_ip_int,
        'DstIP':       dst_ip_int,
        'SrcMAC':      src_mac_int,
        'DstMAC':      dst_mac_int,
        'SrcPort':     flow_key[1],
        'DstPort':     flow_key[3],
        'Protocol':    flow_key[4],
        # --- ML features in P4 table key order ---
        'Duration':    duration_ns,
        'MaxIAT':      max_iat_ns,
        'MinIAT':      min_iat_ns,
        'FwdPktCount': flow_data['fwd_pkt_count'],
        'BwdPktCount': flow_data['bwd_pkt_count'],
        'FwdBytes':    flow_data['fwd_bytes'],
        'BwdBytes':    flow_data['bwd_bytes'],
        'MaxWinSize':      flow_data['max_win_size'],
        'FlagsSyn':        flow_data['flags_syn_count'],
        'FlagsAck':        flow_data['flags_ack_count'],
        'FlagsFin':        flow_data['flags_fin_count'],
        'FlagsRst':        flow_data['flags_rst_count'],
        'FwdMaxPktLen':    flow_data['fwd_max_pkt_len'],
        'BwdMaxPktLen':    flow_data['bwd_max_pkt_len'],
        'FlagsPsh':        flow_data['flags_psh_count'],
        'UrgCount':        flow_data['flags_urg_count'],
        'InitFwdWinBytes': flow_data['init_fwd_win'],
    }
    if label is not None:
        feature['Label'] = label
    return feature


def _empty_flow_state():
    """
    Initialise a blank flow state — mirrors P4 register initial values (all zero).
    """
    return {
        'timestamps':        [],
        'fwd_pkt_count':     0,
        'bwd_pkt_count':     0,
        'fwd_bytes':         0,
        'bwd_bytes':         0,
        'max_win_size':      0,
        'flags_syn_count':   0,  # count of packets with SYN set
        'flags_ack_count':   0,  # count of packets with ACK set
        'flags_fin_count':   0,  # count of packets with FIN set (also triggers flow end)
        'flags_rst_count':   0,  # count of packets with RST set (also triggers flow end)
        'fwd_max_pkt_len':   0,
        'bwd_max_pkt_len':   0,
        'flags_psh_count':   0,
        'flags_urg_count':   0,  # count of packets with URG set
        'init_fwd_win':      0,  # window size of first forward packet (TCP only)
    }


def _update_flow_state(state, timestamp_ns, pkt_len, is_reverse,
                        flags_syn, flags_ack, flags_fin, flags_rst, flags_urg,
                        flags_psh, win_size):
    """Accumulate a single packet into the flow state. Returns True if the flow
    should be finalized after this packet (FIN or RST seen), False otherwise."""
    # ── Quantize timestamp to microsecond resolution ────────────────────
    # BMv2's ingress_global_timestamp is in microseconds; P4 converts via
    # current_time = current_time_us * 1000.  All P4 timestamps are therefore
    # multiples of 1000 ns.  Truncate here so Duration / MaxIAT
    # in the training data match what the P4 switch will actually compute.
    timestamp_ns = (timestamp_ns // 1000) * 1000

    state['timestamps'].append(timestamp_ns)

    # ── Per-direction counters ────────────────────────────────────────
    if is_reverse:
        state['bwd_pkt_count'] += 1
        state['bwd_bytes']     += pkt_len
        if pkt_len > state['bwd_max_pkt_len']:
            state['bwd_max_pkt_len'] = pkt_len
    else:
        # Capture initial forward window on the very first fwd packet
        if state['fwd_pkt_count'] == 0 and state['init_fwd_win'] == 0:
            state['init_fwd_win'] = win_size
        state['fwd_pkt_count'] += 1
        state['fwd_bytes']     += pkt_len
        if pkt_len > state['fwd_max_pkt_len']:
            state['fwd_max_pkt_len'] = pkt_len

    # ── Window / flag counts ─────────────────────────────────────────
    if win_size > state['max_win_size']:
        state['max_win_size'] = win_size
    state['flags_syn_count'] += flags_syn
    state['flags_ack_count'] += flags_ack
    state['flags_fin_count'] += flags_fin
    state['flags_rst_count'] += flags_rst
    state['flags_psh_count'] += flags_psh
    state['flags_urg_count'] += flags_urg

    # Signal flow-end when TCP FIN or RST is seen (mirrors P4 flow_ended logic)
    return bool(flags_fin or flags_rst)


# ---------------------------------------------------------------------------
# Packet field extraction helper
# ---------------------------------------------------------------------------

def _flag(val):
    """Parse a pyshark TCP flag field to 0 or 1.

    Depending on tshark version, flag fields may be returned as:
      '0' / '1'         (older tshark)
      'False' / 'True'  (newer tshark with boolean field types)
    """
    s = str(val).strip()
    return 1 if s in ('1', 'True') else 0


def _extract_packet_fields(packet):
    """Extract all needed fields from a pyshark packet. Returns dict or None.

    Supported protocols:
      - TCP  (IP proto 6)
      - UDP  (IP proto 17)
      - ICMP (IP proto 1)  — uses icmp.type/code as pseudo src/dst port
      - ARP  (etherType 0x0806) — uses spa/tpa as IPs, oper as pseudo src_port,
            protocol encoded as 253 (reserved) to keep the 5-tuple schema intact
    """
    # ── Extract Ethernet MAC addresses ─────────────────────────────────
    try:
        src_mac = packet.eth.src
        dst_mac = packet.eth.dst
    except AttributeError:
        src_mac = '00:00:00:00:00:00'
        dst_mac = '00:00:00:00:00:00'

    # ── ARP (no IP header, carried directly over Ethernet) ───────────────
    if hasattr(packet, 'arp') and not hasattr(packet, 'ip'):
        try:
            src_ip   = packet.arp.src_proto_ipv4
            dst_ip   = packet.arp.dst_proto_ipv4
            src_port = int(packet.arp.opcode)   # 1=request, 2=reply
            dst_port = 0
            # ARP IPv4 payload is 28 bytes (fixed); mirrors P4's meta.pkt_len = 32w28
            pkt_len  = 28
        except Exception:
            return None
        return {
            'src_ip': src_ip, 'dst_ip': dst_ip, 'protocol': 253,
            'src_mac': src_mac, 'dst_mac': dst_mac,
            'src_port': src_port, 'dst_port': dst_port,
            'flags_syn': 0, 'flags_ack': 0, 'flags_fin': 0,
            'flags_rst': 0, 'flags_urg': 0, 'flags_psh': 0,
            'pkt_len': pkt_len,
            'timestamp_ns': int(Decimal(str(packet.sniff_timestamp)) * 1_000_000) * 1000,
            'win_size': 0,
        }

    if not hasattr(packet, 'ip'):
        return None

    src_ip   = packet.ip.src
    dst_ip   = packet.ip.dst
    protocol = int(packet.ip.proto)

    if protocol == 6:   # TCP
        try:
            src_port = int(str(packet.tcp.srcport))
            dst_port = int(str(packet.tcp.dstport))
        except (ValueError, TypeError):
            return None   # malformed port field — skip packet
        flags_syn   = _flag(packet.tcp.flags_syn)   if hasattr(packet.tcp, 'flags_syn')   else 0
        flags_ack   = _flag(packet.tcp.flags_ack)   if hasattr(packet.tcp, 'flags_ack')   else 0
        flags_fin   = _flag(packet.tcp.flags_fin)   if hasattr(packet.tcp, 'flags_fin')   else 0
        flags_rst   = _flag(packet.tcp.flags_reset) if hasattr(packet.tcp, 'flags_reset') else 0
        flags_urg   = _flag(packet.tcp.flags_urg)   if hasattr(packet.tcp, 'flags_urg')   else 0
        flags_psh   = _flag(packet.tcp.flags_push)  if hasattr(packet.tcp, 'flags_push')  else 0
        # window_size_value is the unscaled value (matches the raw TCP header field)
        win_size    = int(packet.tcp.window_size_value) if hasattr(packet.tcp, 'window_size_value') else \
                      (int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else 0)
    elif protocol == 17:  # UDP
        try:
            src_port = int(str(packet.udp.srcport))
            dst_port = int(str(packet.udp.dstport))
        except (ValueError, TypeError):
            return None
        flags_syn = flags_ack = flags_fin = flags_rst = flags_urg = flags_psh = 0
        win_size    = 0
    elif protocol == 1:  # ICMP — use type/code as pseudo ports
        src_port  = int(packet.icmp.type) if hasattr(packet, 'icmp') else 0
        dst_port  = int(packet.icmp.code) if hasattr(packet, 'icmp') else 0
        flags_syn = flags_ack = flags_fin = flags_rst = flags_urg = flags_psh = 0
        win_size  = 0
    else:
        return None

    return {
        'src_ip': src_ip, 'dst_ip': dst_ip, 'protocol': protocol,
        'src_mac': src_mac, 'dst_mac': dst_mac,
        'src_port': src_port, 'dst_port': dst_port,
        'flags_syn': flags_syn, 'flags_ack': flags_ack,
        'flags_fin': flags_fin, 'flags_rst': flags_rst,
        'flags_urg': flags_urg, 'flags_psh': flags_psh,
        # Use IP total length (hdr.ipv4.totalLen) to match exactly what P4 counts.
        # packet.length includes the Ethernet frame header (~14 bytes) which P4 never sees.
        'pkt_len': int(packet.ip.len),
        # P4 ingress_global_timestamp has microsecond granularity (*1000 → ns).
        # Truncate to microseconds here so training features match inference features
        # exactly. Using Decimal avoids float precision loss for large Unix timestamps.
        'timestamp_ns': int(Decimal(str(packet.sniff_timestamp)) * 1_000_000) * 1000,
        'win_size': win_size,
    }


# ---------------------------------------------------------------------------
# PCAP file extraction
# ---------------------------------------------------------------------------

def extract_features_from_pcap(pcap_path, label=None, eof_flush=False):
    features_list = []
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    cap = pyshark.FileCapture(pcap_path, keep_packets=False)

    # P4-style hash-based flow table (65536 slots, indexed by CRC16)
    flow_states = [None] * MAX_REGISTER_ENTRIES
    flow_keys   = [None] * MAX_REGISTER_ENTRIES   # stored 5-tuple per slot
    bf1         = bytearray(MAX_REGISTER_ENTRIES)  # bloom filter 1 (indexed by h16)
    bf2         = bytearray(MAX_REGISTER_ENTRIES)  # bloom filter 2 (indexed by h32)
    reg_h2      = [0] * MAX_REGISTER_ENTRIES       # stored h32 per slot

    total_packets   = 0
    processed_count = 0
    skipped_proto   = 0
    collision_count = 0
    scan_idx        = 0   # scan_and_drain pointer (matches P4 reg_scan_idx)
    drain_count     = 0

    try:
        for packet in cap:
            total_packets += 1
            try:
                fields = _extract_packet_fields(packet)
                if fields is None:
                    skipped_proto += 1
                    continue

                c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, is_reverse, \
                    c_src_mac, c_dst_mac = \
                    make_canonical_key(fields['src_ip'], fields['src_port'],
                                       fields['dst_ip'], fields['dst_port'],
                                       fields['protocol'],
                                       fields['src_mac'], fields['dst_mac'])

                h16, h32 = _compute_flow_hashes(c_src_ip, c_dst_ip,
                                                 c_src_port, c_dst_port, proto)

                # Bloom filter collision detection (matches P4 exactly):
                # bf1 slot occupied but bf2 fingerprint absent → different flow
                if bf1[h16] == 1 and bf2[h32] == 0:
                    collision_count += 1
                    continue

                src_ip_int = int(ipaddress.ip_address(c_src_ip))
                dst_ip_int = int(ipaddress.ip_address(c_dst_ip))

                state = flow_states[h16]

                # Idle timeout: finalize the previous flow, then reset
                if state is not None and state['timestamps'] and \
                        (fields['timestamp_ns'] - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                    feat = _finalize_flow(flow_keys[h16], state, label)
                    if feat:
                        features_list.append(feat)
                    # Reset for new flow (P4 does NOT clear bf2[old_h32] on timeout)
                    flow_states[h16] = _empty_flow_state()
                    flow_keys[h16] = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, c_src_mac, c_dst_mac)
                    bf1[h16] = 1
                    bf2[h32] = 1
                    reg_h2[h16] = h32
                # Active timeout: flow running longer than 60s — export and restart
                elif state is not None and state['timestamps'] and \
                        (fields['timestamp_ns'] - state['timestamps'][0]) > ACTIVE_TIMEOUT_NS:
                    feat = _finalize_flow(flow_keys[h16], state, label)
                    if feat:
                        features_list.append(feat)
                    flow_states[h16] = _empty_flow_state()
                    flow_keys[h16] = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, c_src_mac, c_dst_mac)
                    bf1[h16] = 1
                    bf2[h32] = 1
                    reg_h2[h16] = h32
                elif state is None:
                    # Empty slot — start new flow
                    flow_states[h16] = _empty_flow_state()
                    flow_keys[h16] = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, c_src_mac, c_dst_mac)
                    bf1[h16] = 1
                    bf2[h32] = 1
                    reg_h2[h16] = h32

                # Accumulate packet; check if flow ends (FIN/RST)
                flow_ended = _update_flow_state(
                    flow_states[h16],
                    fields['timestamp_ns'], fields['pkt_len'], is_reverse,
                    fields['flags_syn'], fields['flags_ack'],
                    fields['flags_fin'], fields['flags_rst'],
                    fields['flags_urg'], fields['flags_psh'], fields['win_size'])

                if flow_ended:
                    feat = _finalize_flow(flow_keys[h16], flow_states[h16], label)
                    if feat:
                        features_list.append(feat)
                    # Clear slot (matches P4 FIN/RST register reset + bloom clear)
                    flow_states[h16] = None
                    flow_keys[h16] = None
                    bf1[h16] = 0
                    bf2[reg_h2[h16]] = 0
                    reg_h2[h16] = 0

                # ── scan_and_drain DISABLED during normal traffic ──
                # P4 only runs scan_and_drain on drain-trigger packets
                # (src_ip=10.255.255.254).  During normal pcap processing,
                # flows end via FIN/RST or timeout-on-arrival only.
                # Remaining flows are exported by eof_flush (equivalent
                # to P4's drain triggers).  This makes flow boundaries
                # deterministic and reproducible between Python and P4.

                processed_count += 1
                if processed_count % 1000 == 0:
                    active = sum(1 for s in flow_states if s is not None)
                    logger.info(f"Processed {processed_count}/{total_packets} packets, "
                                f"{active} active flows, {collision_count} collisions, "
                                f"{len(features_list)} completed from {os.path.basename(pcap_path)}")
            except Exception as e:
                logger.warning(f"Error processing packet #{total_packets}: {e}")
                continue
    finally:
        cap.close()

    if skipped_proto > 0:
        logger.info(f"Skipped {skipped_proto}/{total_packets} packets "
                    f"(unsupported protocol / non-IPv4)")
    if collision_count > 0:
        logger.info(f"Hash collisions: {collision_count} packets skipped "
                    f"(matches P4 bloom filter behavior)")
    if drain_count > 0:
        logger.info(f"Scan-and-drain: {drain_count} stale flows exported "
                    f"(matches P4 scan_and_drain action)")

    # EOF flush: finalize remaining open flows.
    # Default OFF (--eof-flush to enable) — P4 switch cannot know when the PCAP
    # ends so flows that are still active remain in registers indefinitely.
    # Disabling EOF flush makes training data match P4 inference exactly.
    eof_flushed = 0
    if eof_flush:
        for h16 in range(MAX_REGISTER_ENTRIES):
            if flow_states[h16] is not None:
                feat = _finalize_flow(flow_keys[h16], flow_states[h16], label)
                if feat:
                    features_list.append(feat)
                    eof_flushed += 1

    active_remaining = sum(1 for s in flow_states if s is not None)
    logger.info(f"Extracted {len(features_list)} flows from "
                f"{processed_count}/{total_packets} packets in {pcap_path}"
                f"{f' (eof_flush={eof_flushed})' if eof_flush else ''}"
                f"{f' ({active_remaining} open flows discarded — use --eof-flush to keep)' if active_remaining and not eof_flush else ''}")
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


def process_pcap_folder(folder_path, output_csv, eof_flush=False):
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
        features = extract_features_from_pcap(pcap_path, label=label, eof_flush=eof_flush)
        mode = 'w' if not header_written else 'a'
        write_to_csv(features, output_csv, mode=mode, write_header=not header_written)
        header_written = True
    logger.info(f"Finished processing. Output: {output_csv}")


def extract_features_from_interface(interface, output_csv, label=None,
                                     packet_count=1000, duration=None):
    logger.info(f"Starting live capture on interface: {interface}")
    features_list = []
    cap           = pyshark.LiveCapture(interface=interface)
    total_seen    = 0
    captured      = 0
    collision_count = 0
    start_time    = time.time()

    # P4-style hash-based flow table (65536 slots)
    flow_states = [None] * MAX_REGISTER_ENTRIES
    flow_keys   = [None] * MAX_REGISTER_ENTRIES
    bf1         = bytearray(MAX_REGISTER_ENTRIES)
    bf2         = bytearray(MAX_REGISTER_ENTRIES)
    reg_h2      = [0] * MAX_REGISTER_ENTRIES

    try:
        for packet in cap.sniff_continuously():
            total_seen += 1
            try:
                if duration and (time.time() - start_time) >= duration:
                    break
                if captured >= packet_count:
                    break
                fields = _extract_packet_fields(packet)
                if fields is None:
                    continue

                c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, is_reverse, \
                    c_src_mac, c_dst_mac = \
                    make_canonical_key(fields['src_ip'], fields['src_port'],
                                       fields['dst_ip'], fields['dst_port'],
                                       fields['protocol'],
                                       fields['src_mac'], fields['dst_mac'])

                h16, h32 = _compute_flow_hashes(c_src_ip, c_dst_ip,
                                                 c_src_port, c_dst_port, proto)

                # Bloom filter collision detection (matches P4)
                if bf1[h16] == 1 and bf2[h32] == 0:
                    collision_count += 1
                    continue

                src_ip_int = int(ipaddress.ip_address(c_src_ip))
                dst_ip_int = int(ipaddress.ip_address(c_dst_ip))

                state = flow_states[h16]

                # Idle timeout: finalize the previous flow, then reset
                if state is not None and state['timestamps'] and \
                        (fields['timestamp_ns'] - state['timestamps'][-1]) > FLOW_TIMEOUT_NS:
                    feat = _finalize_flow(flow_keys[h16], state, label)
                    if feat:
                        features_list.append(feat)
                    flow_states[h16] = _empty_flow_state()
                    flow_keys[h16] = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, c_src_mac, c_dst_mac)
                    bf1[h16] = 1
                    bf2[h32] = 1
                    reg_h2[h16] = h32
                # Active timeout: flow running longer than 60s — export and restart
                elif state is not None and state['timestamps'] and \
                        (fields['timestamp_ns'] - state['timestamps'][0]) > ACTIVE_TIMEOUT_NS:
                    feat = _finalize_flow(flow_keys[h16], state, label)
                    if feat:
                        features_list.append(feat)
                    flow_states[h16] = _empty_flow_state()
                    flow_keys[h16] = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, c_src_mac, c_dst_mac)
                    bf1[h16] = 1
                    bf2[h32] = 1
                    reg_h2[h16] = h32
                elif state is None:
                    flow_states[h16] = _empty_flow_state()
                    flow_keys[h16] = (c_src_ip, c_src_port, c_dst_ip, c_dst_port, proto, c_src_mac, c_dst_mac)
                    bf1[h16] = 1
                    bf2[h32] = 1
                    reg_h2[h16] = h32

                # Accumulate packet; check if flow ends (FIN/RST)
                flow_ended = _update_flow_state(
                    flow_states[h16],
                    fields['timestamp_ns'], fields['pkt_len'], is_reverse,
                    fields['flags_syn'], fields['flags_ack'],
                    fields['flags_fin'], fields['flags_rst'],
                    fields['flags_urg'], fields['flags_psh'], fields['win_size'])

                if flow_ended:
                    feat = _finalize_flow(flow_keys[h16], flow_states[h16], label)
                    if feat:
                        features_list.append(feat)
                    flow_states[h16] = None
                    flow_keys[h16] = None
                    bf1[h16] = 0
                    bf2[reg_h2[h16]] = 0
                    reg_h2[h16] = 0

                captured += 1
                if captured % 100 == 0:
                    active = sum(1 for s in flow_states if s is not None)
                    logger.info(f"Captured {captured}/{total_seen} packets, "
                                f"{active} active flows, {collision_count} collisions")
            except Exception as e:
                logger.warning(f"Error processing packet #{total_seen}: {e}")
                continue
    except KeyboardInterrupt:
        logger.info("\nCapture interrupted by user")
    finally:
        cap.close()

    # Finalize all open flows at end of capture
    for h16 in range(MAX_REGISTER_ENTRIES):
        if flow_states[h16] is not None:
            feat = _finalize_flow(flow_keys[h16], flow_states[h16], label)
            if feat:
                features_list.append(feat)

    logger.info(f"Extracted {len(features_list)} flows from {captured} packets"
                f" ({collision_count} collision-skipped)")
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
    parser.add_argument('--eof-flush', dest='eof_flush', action='store_true',
                        help='Finalize all open flows at end of each PCAP file.\n'
                             'Default OFF — scan_and_drain handles stale flows\n'
                             'the same way the P4 data plane does.')
    parser.add_argument('--no-eof-flush', dest='eof_flush', action='store_false',
                        help='Do NOT finalize open flows at end of each PCAP file (default).')
    parser.set_defaults(eof_flush=True)
    args = parser.parse_args()

    if args.mode == 'pcap':
        if args.input:
            if args.label:
                label = args.label
            else:
                label = extract_base_label(os.path.basename(args.input))
            features = extract_features_from_pcap(args.input, label=label, eof_flush=args.eof_flush)
            write_to_csv(features, args.output, mode='w', write_header=True)
        else:
            process_pcap_folder(args.pcap_dir, args.output, eof_flush=args.eof_flush)
    elif args.mode == 'live':
        if not args.interface:
            logger.error("--interface is required for live capture mode")
            return
        extract_features_from_interface(
            interface=args.interface, output_csv=args.output,
            label=args.label, packet_count=args.count, duration=args.duration)


if __name__ == '__main__':
    main()
