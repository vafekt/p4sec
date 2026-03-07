#!/usr/bin/env python3
"""
P4 Code Generator for Scalable PCA-based ML Classification with Flow-Based Features
Automatically generates basic.p4 with support for N PCA components and three
classifier back-ends:

  --model-type dt   DecisionTree   (default)   single ml_code table
  --model-type rf   RandomForest               N rf_tree_i tables + rf_vote_classify
  --model-type xgb  XGBoost                    N*K xgb_tree_i tables + xgb_classify

Reads model parameters from:
  tables/pca_encoding_params.json   (PCA metadata — always required)
  tables/rf_params.json             (RF  metadata — for --model-type rf)
  tables/xgb_params.json            (XGB metadata — for --model-type xgb)
"""

import json
import os
import argparse
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flow-based features being extracted
FLOW_FEATURES = [
    "Protocol",    # IP protocol number (6=TCP, 17=UDP)
    "Duration",    # Flow duration (ns)
    "MaxIAT",      # Maximum inter-arrival time (ns)
    "UrgCount",    # URG flag packet count
    "FwdPktCount", # Forward packet count
    "BwdPktCount", # Backward packet count
    "FwdBytes",    # Forward bytes
    "BwdBytes",    # Backward bytes
    "MaxWinSize",  # Maximum TCP window size
    "FlagsSyn",    # SYN flag packet count
    "FlagsAck",    # ACK flag packet count
    "FlagsFin",    # FIN flag packet count
    "FlagsRst",    # RST flag packet count
]

class P4CodeGenerator:
    def __init__(self, n_components=2, bits=16, output_file='basic.p4',
                 model_type='dt', rf_params=None, xgb_params=None,
                 n_registers=65536, flow_timeout_s=120):
        self.n_components    = n_components
        self.bits            = bits
        self.output_file     = output_file
        self.model_type      = model_type          # 'dt' | 'rf' | 'xgb'
        self.rf_params       = rf_params  or {}    # from rf_params.json
        self.xgb_params      = xgb_params or {}    # from xgb_params.json
        self.n_registers     = n_registers
        self.flow_timeout_ns = int(flow_timeout_s * 1_000_000_000)
        
    def generate_header(self):
        """Generate P4 file header with includes and constants."""
        return '''/* -*- P4_16 -*- */
/*
 * P4 Flow-Based ML Classification
 * Extracts flow-based features and applies PCA + Decision Tree classification
 */

#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x800;
const bit<8>  TYPE_TCP  = 6;
const bit<8>  TYPE_UDP  = 17;

const bit<32> NB_ENTRIES = ''' + str(self.n_registers) + ''';
const bit<32> MAX_REGISTER_ENTRIES = ''' + str(self.n_registers) + ''';

// Bloom filter for flow detection
#define BLOOM_FILTER_BIT_WIDTH 32
#define FLOW_TIMEOUT ''' + str(self.flow_timeout_ns) + '''  // ''' + str(self.flow_timeout_ns // 1_000_000_000) + ''' seconds in nanoseconds

// Macros for register operations
#define FIRST_INDEX ((bit<32>)0)
#define WRITE_REG(r, v) r.write(FIRST_INDEX, v)
#define READ_REG(r,  v) r.read(v, FIRST_INDEX)

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

// Flow-based feature types
typedef bit<48> iat_t;          // Inter-Arrival Time (nanoseconds)
typedef bit<48> duration_t;     // Flow duration (nanoseconds)
typedef bit<16> port_t;         // Port number
typedef bit<32> bytes_t;        // Byte count
typedef bit<''' + str(self.bits) + '''> pca_code_t;   // PCA component code (quantized)
typedef bit<8>  inference_result_t; // DT classification result

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> udpTotalLen;
    bit<16> checksum;
}
'''

    def generate_metadata(self):
        """Generate metadata struct with dynamic PCA component codes."""
        code = '''
struct metadata {
    // Flow identification (5-tuple) — raw, as parsed from headers
    ip4Addr_t src_ip;
    ip4Addr_t dst_ip;
    port_t src_port;
    port_t dst_port;
    bit<8>  protocol;

    // Canonical (direction-normalized) bidirectional flow key.
    // compute_flow_hash() fills these so A->B and B->A map to the same slot.
    ip4Addr_t canon_src_ip;
    ip4Addr_t canon_dst_ip;
    port_t    canon_src_port;
    port_t    canon_dst_port;
    bit<1>    is_reverse_dir;   // 1 = packet in reverse direction

    // Flow state tracking
    bit<32> flow_hash;
    bit<32> flow_hash_2;
    bit<1>  is_first_packet;
    bit<1>  hash_collision;
    bit<1>  flow_ended;     // 1 = flow is complete (timeout or FIN/RST) — gate classify+digest
    
    // Flow-based features
    duration_t duration;
    iat_t      max_iat;
    bit<32>    urg_count;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    max_win_size;
    bit<32>    flags_syn;    // SYN flag packet count
    bit<32>    flags_ack;    // ACK flag packet count
    bit<32>    flags_fin;    // FIN flag packet count
    bit<32>    flags_rst;    // RST flag packet count
    
    // PCA-transformed features (quantized)
'''
        # Add PCA component codes dynamically
        for i in range(1, self.n_components + 1):
            code += f'    pca_code_t pc{i}_code;\n'
        
        code += '''    
    // Classification result
    inference_result_t ml_result;
    
    // Timestamp
    bit<48> ingress_timestamp;
'''
        # ---- RF packed vote field ----
        if self.model_type == 'rf':
            n_est        = self.rf_params.get('n_estimators', 8)
            vote_bits    = self.rf_params.get('vote_bits',    2)
            total_v_bits = n_est * vote_bits
            code += f'\n    // RF packed vote field ({n_est} trees x {vote_bits} bits)\n'
            code += f'    bit<{total_v_bits}> rf_votes;\n'

        # ---- XGB per-class score accumulators ----
        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            code += f'\n    // XGB per-class score accumulators (16-bit, bias=128)\n'
            for c in range(n_cls):
                code += f'    bit<16> xgb_score_c{c};\n'

        code += '''}

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;
    udp_t        udp;
}

struct digest_t {
    // Flow identification (5-tuple)
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
    port_t srcPort;
    port_t dstPort;
    bit<8>  protocol;
    
    // Flow-based features
    duration_t duration;
    iat_t      max_iat;
    bit<32>    urg_count;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    max_win_size;
    bit<32>    flags_syn;    // SYN flag packet count
    bit<32>    flags_ack;    // ACK flag packet count
    bit<32>    flags_fin;    // FIN flag packet count
    bit<32>    flags_rst;    // RST flag packet count
    
    // PCA component codes
'''
        # Add PCA component codes to digest
        for i in range(1, self.n_components + 1):
            code += f'    pca_code_t pc{i}_code;\n'
        
        # Add XGB scores if model_type is 'xgb'
        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                code += f'    bit<16> xgb_score_c{c};\n'
        
        code += '''    
    // Classification result
    inference_result_t ml_result;
}
'''
        return code

    def generate_parser(self):
        """Generate parser logic."""
        return '''
/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default  : accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        meta.src_ip = hdr.ipv4.srcAddr;
        meta.dst_ip = hdr.ipv4.dstAddr;
        meta.protocol = hdr.ipv4.protocol;
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default : accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        meta.src_port = hdr.tcp.srcPort;
        meta.dst_port = hdr.tcp.dstPort;
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        meta.src_port = hdr.udp.srcPort;
        meta.dst_port = hdr.udp.dstPort;
        transition accept;
    }
}

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {   
    apply {  }
}
'''

    def generate_ingress_forwarding(self):
        """Generate ingress control with flow tracking and feature extraction."""
        code = '''
/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    // Registers for flow state tracking
    register<bit<48>>(MAX_REGISTER_ENTRIES) reg_time_first_pkt;   // Time of first packet
    register<bit<48>>(MAX_REGISTER_ENTRIES) reg_time_last_pkt;    // Time of last packet
    register<iat_t>(MAX_REGISTER_ENTRIES)   reg_max_iat;          // Max inter-arrival time
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_urg_count;        // URG packet count
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_fwd_pkt_count;    // Forward packet count
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_bwd_pkt_count;    // Backward packet count
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_fwd_bytes;        // Forward bytes
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_bwd_bytes;        // Backward bytes
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_max_win_size;     // Max TCP window size
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_syn;        // SYN flag packet count
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_ack;        // ACK flag packet count
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_fin;        // FIN flag packet count
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_rst;        // RST flag packet count
    
    // Bloom filter for efficient flow tracking
    register<bit<1>>(MAX_REGISTER_ENTRIES) bloom_filter;

    // Forwarding actions
    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = drop();
    }

    // Bidirectional canonical flow hash.
    // Normalises direction so A->B and B->A map to the same register slot.
    action compute_flow_hash() {
        if (meta.src_ip < meta.dst_ip) {
            meta.canon_src_ip   = meta.src_ip;
            meta.canon_dst_ip   = meta.dst_ip;
            meta.canon_src_port = meta.src_port;
            meta.canon_dst_port = meta.dst_port;
            meta.is_reverse_dir = 1w0;
        } else if (meta.src_ip > meta.dst_ip) {
            meta.canon_src_ip   = meta.dst_ip;
            meta.canon_dst_ip   = meta.src_ip;
            meta.canon_src_port = meta.dst_port;
            meta.canon_dst_port = meta.src_port;
            meta.is_reverse_dir = 1w1;
        } else {
            if (meta.src_port <= meta.dst_port) {
                meta.canon_src_ip   = meta.src_ip;
                meta.canon_dst_ip   = meta.dst_ip;
                meta.canon_src_port = meta.src_port;
                meta.canon_dst_port = meta.dst_port;
                meta.is_reverse_dir = 1w0;
            } else {
                meta.canon_src_ip   = meta.dst_ip;
                meta.canon_dst_ip   = meta.src_ip;
                meta.canon_src_port = meta.dst_port;
                meta.canon_dst_port = meta.src_port;
                meta.is_reverse_dir = 1w1;
            }
        }
        hash(meta.flow_hash, HashAlgorithm.crc16, (bit<16>)0,
            {meta.canon_src_ip, meta.canon_dst_ip,
             meta.canon_src_port, meta.canon_dst_port, meta.protocol},
            (bit<32>)MAX_REGISTER_ENTRIES);
        hash(meta.flow_hash_2, HashAlgorithm.crc32, (bit<16>)0,
            {meta.canon_src_ip, meta.canon_dst_ip,
             meta.canon_src_port, meta.canon_dst_port, meta.protocol},
            (bit<32>)MAX_REGISTER_ENTRIES);
        bloom_filter.write(meta.flow_hash, 1w1);
    }

    // Helper to update flow state
    // TCP flag counts (FlagsSyn/Ack/Fin/Rst) are accumulated here via register +=
    // matching exactly the per-packet += logic in 1_data_extraction.py.
    action update_flow_state() {
        bit<48> current_time_us = standard_metadata.ingress_global_timestamp;
        bit<48> current_time = current_time_us * 1000;  // convert to nanoseconds
        bit<48> time_first;
        bit<48> time_last;
        iat_t   max_iat;
        bit<32> fwd_pkt_count;
        bit<32> bwd_pkt_count;
        bytes_t fwd_bytes;
        bytes_t bwd_bytes;
        bit<16> max_win_size;
        bit<32> urg_count;
        bit<32> flags_syn;
        bit<32> flags_ack;
        bit<32> flags_fin;
        bit<32> flags_rst;

        // Read current state
        reg_time_first_pkt.read(time_first, meta.flow_hash);
        reg_time_last_pkt.read(time_last, meta.flow_hash);
        reg_max_iat.read(max_iat, meta.flow_hash);
        reg_fwd_pkt_count.read(fwd_pkt_count, meta.flow_hash);
        reg_bwd_pkt_count.read(bwd_pkt_count, meta.flow_hash);
        reg_fwd_bytes.read(fwd_bytes, meta.flow_hash);
        reg_bwd_bytes.read(bwd_bytes, meta.flow_hash);
        reg_max_win_size.read(max_win_size, meta.flow_hash);
        reg_urg_count.read(urg_count, meta.flow_hash);
        reg_flags_syn.read(flags_syn, meta.flow_hash);
        reg_flags_ack.read(flags_ack, meta.flow_hash);
        reg_flags_fin.read(flags_fin, meta.flow_hash);
        reg_flags_rst.read(flags_rst, meta.flow_hash);

        if (time_last != 0 && (current_time - time_last) > FLOW_TIMEOUT) {
            // Flow timeout: the previous flow is complete — classify it before resetting.
            // Snapshot the old accumulated features into meta so PCA+classify runs on them.
            meta.flow_ended    = 1w1;
            meta.duration      = current_time - time_first;  // full duration of old flow
            meta.max_iat       = max_iat;
            meta.urg_count     = urg_count;
            meta.fwd_pkt_count = fwd_pkt_count;
            meta.bwd_pkt_count = bwd_pkt_count;
            meta.fwd_bytes     = fwd_bytes;
            meta.bwd_bytes     = bwd_bytes;
            meta.max_win_size  = max_win_size;
            // Snapshot accumulated flag counts BEFORE clearing the registers
            meta.flags_syn = flags_syn;
            meta.flags_ack = flags_ack;
            meta.flags_fin = flags_fin;
            meta.flags_rst = flags_rst;

            // Now reset registers for the new flow starting with this packet
            meta.is_first_packet = 1w1;
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            reg_time_last_pkt.write(meta.flow_hash, current_time);
            reg_max_iat.write(meta.flow_hash, 0);
            reg_urg_count.write(meta.flow_hash, 0);
            reg_fwd_pkt_count.write(meta.flow_hash,
                meta.is_reverse_dir == 1w0 ? 32w1 : 32w0);
            reg_bwd_pkt_count.write(meta.flow_hash,
                meta.is_reverse_dir == 1w1 ? 32w1 : 32w0);
            reg_fwd_bytes.write(meta.flow_hash,
                meta.is_reverse_dir == 1w0 ? (bytes_t)standard_metadata.packet_length : 32w0);
            reg_bwd_bytes.write(meta.flow_hash,
                meta.is_reverse_dir == 1w1 ? (bytes_t)standard_metadata.packet_length : 32w0);
            reg_max_win_size.write(meta.flow_hash,
                (meta.protocol == TYPE_TCP) ? hdr.tcp.window : 16w0);
            reg_flags_syn.write(meta.flow_hash, 32w0);
            reg_flags_ack.write(meta.flow_hash, 32w0);
            reg_flags_fin.write(meta.flow_hash, 32w0);
            reg_flags_rst.write(meta.flow_hash, 32w0);
        } else if (time_first == 0) {
            // First packet of flow
            meta.flow_ended    = 1w0;
            meta.is_first_packet = 1w1;
            meta.duration      = 0;
            meta.max_iat       = 0;
            meta.urg_count     = 0;
            meta.fwd_pkt_count = meta.is_reverse_dir == 1w0 ? 32w1 : 32w0;
            meta.bwd_pkt_count = meta.is_reverse_dir == 1w1 ? 32w1 : 32w0;
            meta.fwd_bytes     = meta.is_reverse_dir == 1w0 ? (bytes_t)standard_metadata.packet_length : 32w0;
            meta.bwd_bytes     = meta.is_reverse_dir == 1w1 ? (bytes_t)standard_metadata.packet_length : 32w0;
            meta.max_win_size  = (meta.protocol == TYPE_TCP) ? hdr.tcp.window : 16w0;
            meta.flags_syn     = 32w0;
            meta.flags_ack     = 32w0;
            meta.flags_fin     = 32w0;
            meta.flags_rst     = 32w0;

            reg_time_first_pkt.write(meta.flow_hash, current_time);
            reg_fwd_pkt_count.write(meta.flow_hash, meta.fwd_pkt_count);
            reg_bwd_pkt_count.write(meta.flow_hash, meta.bwd_pkt_count);
            reg_fwd_bytes.write(meta.flow_hash, meta.fwd_bytes);
            reg_bwd_bytes.write(meta.flow_hash, meta.bwd_bytes);
            reg_max_win_size.write(meta.flow_hash, meta.max_win_size);
        } else {
            // Subsequent packet
            meta.flow_ended      = 1w0;
            meta.is_first_packet = 1w0;

            // Max IAT
            iat_t current_iat = current_time - time_last;
            if (current_iat > max_iat) {
                max_iat = current_iat;
            }
            reg_max_iat.write(meta.flow_hash, max_iat);
            meta.max_iat = max_iat;

            // Duration
            meta.duration = current_time - time_first;

            // Directional packet counts and bytes
            if (meta.is_reverse_dir == 1w0) {
                fwd_pkt_count = fwd_pkt_count + 1;
                fwd_bytes = fwd_bytes + (bytes_t)standard_metadata.packet_length;
                reg_fwd_pkt_count.write(meta.flow_hash, fwd_pkt_count);
                reg_fwd_bytes.write(meta.flow_hash, fwd_bytes);
            } else {
                bwd_pkt_count = bwd_pkt_count + 1;
                bwd_bytes = bwd_bytes + (bytes_t)standard_metadata.packet_length;
                reg_bwd_pkt_count.write(meta.flow_hash, bwd_pkt_count);
                reg_bwd_bytes.write(meta.flow_hash, bwd_bytes);
            }
            meta.fwd_pkt_count = fwd_pkt_count;
            meta.bwd_pkt_count = bwd_pkt_count;
            meta.fwd_bytes = fwd_bytes;
            meta.bwd_bytes = bwd_bytes;

            // Max window size
            if (meta.protocol == TYPE_TCP) {
                if (hdr.tcp.window > max_win_size) {
                    max_win_size = hdr.tcp.window;
                    reg_max_win_size.write(meta.flow_hash, max_win_size);
                }
            }
            meta.max_win_size = max_win_size;
        }

        // Always update last packet time
        reg_time_last_pkt.write(meta.flow_hash, current_time);

        // URG count (TCP only)
        if (meta.protocol == TYPE_TCP && hdr.tcp.ctrl[5:5] == 1w1) {
            urg_count = urg_count + 1;
            reg_urg_count.write(meta.flow_hash, urg_count);
        }
        meta.urg_count = urg_count;

        // TCP flag counts: accumulate per-packet counts (mirrors P4 register += logic)
        if (meta.protocol == TYPE_TCP) {
            flags_syn = flags_syn + (bit<32>)hdr.tcp.ctrl[1:1];
            flags_ack = flags_ack + (bit<32>)hdr.tcp.ctrl[4:4];
            flags_fin = flags_fin + (bit<32>)hdr.tcp.ctrl[0:0];
            flags_rst = flags_rst + (bit<32>)hdr.tcp.ctrl[2:2];
            reg_flags_syn.write(meta.flow_hash, flags_syn);
            reg_flags_ack.write(meta.flow_hash, flags_ack);
            reg_flags_fin.write(meta.flow_hash, flags_fin);
            reg_flags_rst.write(meta.flow_hash, flags_rst);
        }
        meta.flags_syn = flags_syn;
        meta.flags_ack = flags_ack;
        meta.flags_fin = flags_fin;
        meta.flags_rst = flags_rst;

        // FIN or RST ends the flow — snapshot current accumulated state and mark ended.
        // (Only when flow_ended not already set by timeout branch above)
        if (meta.flow_ended == 1w0 &&
                meta.protocol == TYPE_TCP &&
                (meta.flags_fin > 32w0 || meta.flags_rst > 32w0)) {
            meta.flow_ended    = 1w1;
            meta.duration      = current_time - time_first;
            meta.max_iat       = max_iat;
            meta.urg_count     = urg_count;
            reg_fwd_pkt_count.read(meta.fwd_pkt_count, meta.flow_hash);
            reg_bwd_pkt_count.read(meta.bwd_pkt_count, meta.flow_hash);
            reg_fwd_bytes.read(meta.fwd_bytes, meta.flow_hash);
            reg_bwd_bytes.read(meta.bwd_bytes, meta.flow_hash);
            reg_max_win_size.read(meta.max_win_size, meta.flow_hash);
            // All flag counts already accumulated into meta.flags_* above
            // Reset register slot so next flow on same 5-tuple starts fresh
            reg_time_first_pkt.write(meta.flow_hash, 0);
            reg_time_last_pkt.write(meta.flow_hash, 0);
            reg_max_iat.write(meta.flow_hash, 0);
            reg_urg_count.write(meta.flow_hash, 0);
            reg_fwd_pkt_count.write(meta.flow_hash, 0);
            reg_bwd_pkt_count.write(meta.flow_hash, 0);
            reg_fwd_bytes.write(meta.flow_hash, 0);
            reg_bwd_bytes.write(meta.flow_hash, 0);
            reg_max_win_size.write(meta.flow_hash, 0);
            reg_flags_syn.write(meta.flow_hash, 32w0);
            reg_flags_ack.write(meta.flow_hash, 32w0);
            reg_flags_fin.write(meta.flow_hash, 32w0);
            reg_flags_rst.write(meta.flow_hash, 32w0);
        }
    }

'''

        # Add PCA transformation tables
        for i in range(1, self.n_components + 1):
            code += f'''
    // PCA component {i} transformation
    action set_pc{i}_code(pca_code_t code) {{
        meta.pc{i}_code = code;
    }}

    table pca_component{i} {{
        key = {{
            meta.protocol        : range;
            meta.duration        : range;
            meta.max_iat         : range;
            meta.urg_count       : range;
            meta.fwd_pkt_count   : range;
            meta.bwd_pkt_count   : range;
            meta.fwd_bytes       : range;
            meta.bwd_bytes       : range;
            meta.max_win_size    : range;
            meta.flags_syn       : range;
            meta.flags_ack       : range;
            meta.flags_fin       : range;
            meta.flags_rst       : range;
        }}
        actions = {{
            set_pc{i}_code;
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

        # -------------------------------------------------------------------------
        # Classification tables — model-type specific
        # -------------------------------------------------------------------------

        # Shared set_result action (used by all model types)
        code += '''
    // Shared classification result action
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }
'''

        if self.model_type == 'dt':
            # ---- DT: single ml_code range-match table ----
            code += '''
    // Decision Tree classification using PCA component codes
    table ml_code {
        key = {
'''
            for i in range(1, self.n_components + 1):
                code += f'            meta.pc{i}_code : range;\n'
            code += '''        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }
'''

        elif self.model_type == 'rf':
            # ---- RF: one table per tree + vote aggregation table ----
            n_est     = self.rf_params.get('n_estimators', 8)
            vote_bits = self.rf_params.get('vote_bits',    2)

            # Determine which PCA code fields this RF model actually uses
            rf_feature_names = self.rf_params.get('feature_names',
                [f'PC{j}_code' for j in range(1, self.n_components + 1)])

            for i in range(n_est):
                lo_bit = i * vote_bits
                hi_bit = lo_bit + vote_bits - 1
                code += f'''
    // RF tree {i} — range match on PCA codes, output vote to rf_votes[{hi_bit}:{lo_bit}]
    action set_rf_tree_{i}_vote(bit<{vote_bits}> vote) {{
        meta.rf_votes[{hi_bit}:{lo_bit}] = vote;
    }}

    table rf_tree_{i} {{
        key = {{
'''
                for feat_name in rf_feature_names:
                    meta_field = feat_name.lower()   # e.g. "PC1_code" -> "pc1_code"
                    code += f'            meta.{meta_field} : range;\n'
                code += f'''        }}
        actions = {{
            set_rf_tree_{i}_vote;
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

            total_vote_bits = n_est * vote_bits
            code += f'''
    // RF vote aggregation — exact match on packed vote field ({total_vote_bits} bits)
    table rf_vote_classify {{
        key = {{
            meta.rf_votes : exact;
        }}
        actions = {{
            set_result;
            NoAction;
        }}
        size = {2**total_vote_bits};
    }}
'''

        elif self.model_type == 'xgb':
            # ---- XGB: one table per tree accumulates class scores ----
            total_trees = self.xgb_params.get('total_trees', 0)
            n_cls       = self.xgb_params.get('n_classes',   2)

            # Define all class score accumulator actions once (not repeated per tree)
            for class_idx in range(n_cls):
                code += f'''
    // XGB accumulator action for class {class_idx}
    action add_xgb_score_c{class_idx}(bit<8> delta) {{
        meta.xgb_score_c{class_idx} = meta.xgb_score_c{class_idx} + (bit<16>)delta;
    }}
'''

            # Generate one table per tree
            for tree_idx in range(total_trees):
                class_idx = tree_idx % n_cls
                # Determine which PCA code fields this model actually uses
                xgb_feature_names = self.xgb_params.get('feature_names',
                    [f'PC{j}_code' for j in range(1, self.n_components + 1)])

                code += f'''
    // XGB tree {tree_idx} (class {class_idx}) — adds quantised leaf delta to score accumulator
    table xgb_tree_{tree_idx} {{
        key = {{
'''
                for feat_name in xgb_feature_names:
                    meta_field = feat_name.lower()   # e.g. "PC1_code" -> "pc1_code"
                    code += f'            meta.{meta_field} : range;\n'
                code += f'''        }}
        actions = {{
            add_xgb_score_c{class_idx};
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

            # Final proxy-DT classify table — range match on accumulated scores
            code += '''
    // XGB final classification — range match on per-class accumulated scores
    table xgb_classify {
        key = {
'''
            for c in range(n_cls):
                code += f'            meta.xgb_score_c{c} : range;\n'
            code += '''        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }
'''

        # -------------------------------------------------------------------------
        # Apply block
        # -------------------------------------------------------------------------
        code += '''
    apply {
        if (hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP)) {
            // Step 1: Compute bidirectional flow hash (canonical direction)
            compute_flow_hash();

            // Step 2: Update flow state.
            //         Flag counts are accumulated inside update_flow_state().
            //         meta.flow_ended is set to 1 when the flow is complete
            //         (timeout exposing previous flow, or FIN/RST ending current flow).
            update_flow_state();

            // Step 3–6 only run when a complete flow is ready for classification.
            // Require at least 2 packets to match training data filtering.
            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 2) {
                // (Flag counts already snapshotted inside update_flow_state before register clear)

                // Step 3: Apply PCA transformations
'''
        for i in range(1, self.n_components + 1):
            code += f'            pca_component{i}.apply();\n'

        code += '\n            // Step 5: Apply classifier\n'

        if self.model_type == 'dt':
            code += '            ml_code.apply();\n'

        elif self.model_type == 'rf':
            n_est = self.rf_params.get('n_estimators', 8)
            code += '            // Initialize packed vote field\n'
            total_vote_bits = self.rf_params.get('n_estimators', 8) * self.rf_params.get('vote_bits', 2)
            code += f'            meta.rf_votes = {total_vote_bits}w0;\n'
            for i in range(n_est):
                code += f'            rf_tree_{i}.apply();\n'
            code += '            rf_vote_classify.apply();\n'

        elif self.model_type == 'xgb':
            total_trees = self.xgb_params.get('total_trees', 0)
            n_cls       = self.xgb_params.get('n_classes',   2)
            code += '            // Initialize per-class score accumulators\n'
            for c in range(n_cls):
                code += f'            meta.xgb_score_c{c} = 16w0;\n'
            for tree_idx in range(total_trees):
                code += f'            xgb_tree_{tree_idx}.apply();\n'
            code += '            xgb_classify.apply();\n'

        code += '''
            // Step 6: Send digest with flow features and classification result
            digest<digest_t>(1, {
                meta.canon_src_ip,
                meta.canon_dst_ip,
                meta.canon_src_port,
                meta.canon_dst_port,
                meta.protocol,
                meta.duration,
                meta.max_iat,
                meta.urg_count,
                meta.fwd_pkt_count,
                meta.bwd_pkt_count,
                meta.fwd_bytes,
                meta.bwd_bytes,
                meta.max_win_size,
                meta.flags_syn,
                meta.flags_ack,
                meta.flags_fin,
                meta.flags_rst,
'''
        for i in range(1, self.n_components + 1):
            code += f'                meta.pc{i}_code,\n'
        
        # Add XGB scores if model_type is 'xgb'
        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                code += f'                meta.xgb_score_c{c},\n'

        code += '''                meta.ml_result
            });
            } // end if (meta.flow_ended == 1w1 && >= 2 packets)

            // Step 7: Forward packet (always, regardless of flow state)
            ipv4_lpm.apply();
        }
    }
}
'''
        return code


    def generate_egress_and_tail(self):
        """Generate egress, checksum, deparser, and main switch."""
        return '''
/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply {  }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
     apply {
         update_checksum(
            hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
'''

    def generate(self):
        """Generate complete P4 code."""
        logger.info(f"Generating P4 code: model_type={self.model_type}, "
                    f"{self.n_components} PCA components, {self.bits}-bit codes")

        code = self.generate_header()
        code += self.generate_metadata()
        code += self.generate_parser()
        code += self.generate_ingress_forwarding()
        code += self.generate_egress_and_tail()
        return code

    def write_to_file(self):
        """Write generated P4 code to file."""
        code = self.generate()
        with open(self.output_file, 'w') as f:
            f.write(code)

        logger.info(f"Successfully generated {self.output_file}")
        logger.info(f"  Model type   : {self.model_type}")
        logger.info(f"  PCA components: {self.n_components}")
        logger.info(f"  Quantization  : {self.bits}-bit codes")
        if self.model_type == 'rf':
            logger.info(f"  RF trees      : {self.rf_params.get('n_estimators')}")
            logger.info(f"  Vote bits     : {self.rf_params.get('vote_bits')}")
        elif self.model_type == 'xgb':
            logger.info(f"  XGB trees     : {self.xgb_params.get('total_trees')}")
            logger.info(f"  XGB classes   : {self.xgb_params.get('n_classes')}")



def detect_n_components(params_file='tables/pca_encoding_params.json',
                        commands_file='tables/s1-commands.txt'):
    """
    Detect PCA component count and bit-width from saved parameter files.
    Priority: pca_encoding_params.json → s1-commands.txt heuristic → default.
    """
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
            n_components = params.get('n_components')
            bits         = params.get('bits', 16)
            if n_components:
                logger.info(f"Detected {n_components} PCA components ({bits}-bit) from {params_file}")
                return n_components, bits
        except Exception as e:
            logger.warning(f"Could not read {params_file}: {e}")

    if os.path.exists(commands_file):
        try:
            with open(commands_file, 'r') as f:
                tables = set()
                for line in f:
                    m = re.search(r'pca_component(\d+)', line)
                    if m:
                        tables.add(int(m.group(1)))
            if tables:
                n = max(tables)
                logger.info(f"Detected {n} PCA components from {commands_file}")
                return n, 16
        except Exception as e:
            logger.warning(f"Could not parse {commands_file}: {e}")

    logger.warning("Could not auto-detect PCA components — defaulting to 2 (16-bit)")
    return 2, 16


def load_rf_params(rf_params_file='tables/rf_params.json'):
    """Load RF deployment parameters saved by 3_rf_training_model.py."""
    if os.path.exists(rf_params_file):
        try:
            with open(rf_params_file) as f:
                params = json.load(f)
            logger.info(f"Loaded RF params: n_estimators={params.get('n_estimators')}, "
                        f"vote_bits={params.get('vote_bits')}, "
                        f"n_classes={params.get('n_classes')}")
            return params
        except Exception as e:
            logger.warning(f"Could not read {rf_params_file}: {e}")
    logger.warning("RF params file not found — using defaults (8 estimators, 2 vote_bits)")
    return {"n_estimators": 8, "vote_bits": 2, "n_classes": 4}


def load_xgb_params(xgb_params_file='tables/xgb_params.json'):
    """Load XGB deployment parameters saved by 3_xgb_training_model.py."""
    if os.path.exists(xgb_params_file):
        try:
            with open(xgb_params_file) as f:
                params = json.load(f)
            logger.info(f"Loaded XGB params: total_trees={params.get('total_trees')}, "
                        f"n_classes={params.get('n_classes')}")
            return params
        except Exception as e:
            logger.warning(f"Could not read {xgb_params_file}: {e}")
    logger.warning("XGB params file not found — using defaults")
    return {"total_trees": 16, "n_classes": 2, "n_estimators": 8}


def main():
    parser = argparse.ArgumentParser(
        description='Generate P4 code for PCA + ML classification (DT / RF / XGB)')
    parser.add_argument('--output', default='../basic.p4',
                        help='Output P4 file path (default: ../basic.p4)')
    parser.add_argument('--params-file', default='tables/pca_encoding_params.json',
                        help='PCA encoding parameters JSON')
    parser.add_argument('--commands-file', default='tables/s1-commands.txt',
                        help='S1 commands file (fallback for component detection)')
    parser.add_argument('--model-type', default='dt',
                        choices=['dt', 'rf', 'xgb'],
                        help='Classifier back-end: dt (default) | rf | xgb')
    parser.add_argument('--rf-params',  default='tables/rf_params.json',
                        help='RF params JSON (used when --model-type rf)')
    parser.add_argument('--xgb-params', default='tables/xgb_params.json',
                        help='XGB params JSON (used when --model-type xgb)')
    parser.add_argument('--register-entries', type=int, default=65536,
                        help='MAX_REGISTER_ENTRIES constant in generated P4 (default: 65536)')
    parser.add_argument('--flow-timeout-s', type=int, default=120,
                        help='FLOW_TIMEOUT in seconds for generated P4 (default: 120)')

    args = parser.parse_args()

    # Auto-detect PCA config
    n_components, bits = detect_n_components(args.params_file, args.commands_file)

    # Load model-specific params
    rf_params  = load_rf_params(args.rf_params)   if args.model_type == 'rf'  else {}
    xgb_params = load_xgb_params(args.xgb_params) if args.model_type == 'xgb' else {}

    generator = P4CodeGenerator(
        n_components=n_components,
        bits=bits,
        output_file=args.output,
        model_type=args.model_type,
        rf_params=rf_params,
        xgb_params=xgb_params,
        n_registers=args.register_entries,
        flow_timeout_s=args.flow_timeout_s,
    )
    generator.write_to_file()

    logger.info("\nGeneration complete!")
    logger.info(f"Model type : {args.model_type.upper()}")
    if args.model_type == 'dt':
        logger.info("Entry gen  : run 4_dt_generating_entries.py to populate tables/s1-commands.txt")
    elif args.model_type == 'rf':
        logger.info("Entry gen  : run 4_rf_generating_entries.py to populate tables/s1-commands.txt")
    elif args.model_type == 'xgb':
        logger.info("Entry gen  : run 4_xgb_generating_entries.py to populate tables/s1-commands.txt")
    logger.info(f"To compile : make")


if __name__ == '__main__':
    main()