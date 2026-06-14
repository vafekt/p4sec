#!/usr/bin/env python3
"""
Universal P4 Code Generator for ML Classification with Flow-Based Features.

Supports three dimensionality-reduction methods:
    PCA               → pca_component* transform tables + classifier on pc*_code
    LDA               → lda_component* transform tables + classifier on ld*_code
    Autoencoder       → ae_component*  transform tables + classifier on ae*_code

Supports two classifier back-ends:
  --model-type dt   DecisionTree    single ml_code table
  --model-type rf   RandomForest    N rf_tree_i tables + rf_vote_classify

Reads configuration from:
  tables/reduction_config.json     (written by any step 2 — method, features, max values)
    tables/encoding_params.json      (transform encoding params — fallback)
  tables/model_params.json         (universal model metadata)
"""

import json
import os
import argparse
import sys
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# ─── All P4 raw flow features (paper Table 2, cross-flow features omitted) ──
FLOW_FEATURES = [
    "SrcIP", "DstIP",
    "Protocol", "SrcPort", "DstPort",
    "Duration", "MaxIAT",
    "FwdPktCount", "BwdPktCount", "FwdBytes", "BwdBytes",
    "FwdMaxPktLen", "BwdMaxPktLen",
    "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst", "FlagsPsh",
    "MaxWinSize", "InitFwdWinBytes",
]

# Features used in PCA transform table keys (and as raw classifier keys).
# SrcIP/DstIP are excluded: they are flow identifiers, not ML features.
TRANSFORM_KEY_FEATURES = [f for f in FLOW_FEATURES if f not in ("SrcIP", "DstIP")]

# Map raw feature name → P4 metadata field name (without meta. prefix)
FEATURE_TO_META = {
    "SrcIP":              "canon_src_ip",
    "DstIP":              "canon_dst_ip",
    "Protocol":           "protocol",
    "SrcPort":            "canon_src_port",
    "DstPort":            "canon_dst_port",
    "Duration":           "duration",
    "MaxIAT":             "max_iat",
    "FwdPktCount":        "fwd_pkt_count",
    "BwdPktCount":        "bwd_pkt_count",
    "FwdBytes":           "fwd_bytes",
    "BwdBytes":           "bwd_bytes",
    "FwdMaxPktLen":       "fwd_max_pkt_len",
    "BwdMaxPktLen":       "bwd_max_pkt_len",
    "FlagsSyn":           "flags_syn",
    "FlagsAck":           "flags_ack",
    "FlagsFin":           "flags_fin",
    "FlagsRst":           "flags_rst",
    "FlagsPsh":           "flags_psh",
    "MaxWinSize":         "max_win_size",
    "InitFwdWinBytes":    "init_fwd_win",
}

# P4 bit widths for raw features
FEATURE_P4_TYPE = {
    "SrcIP":              "ip4Addr_t",     # bit<32>
    "DstIP":              "ip4Addr_t",     # bit<32>
    "Protocol":           "bit<8>",
    "SrcPort":            "port_t",        # bit<16>
    "DstPort":            "port_t",        # bit<16>
    "Duration":           "duration_t",    # bit<48>
    "MaxIAT":             "iat_t",         # bit<48>
    "FwdPktCount":        "bit<32>",
    "BwdPktCount":        "bit<32>",
    "FwdBytes":           "bytes_t",       # bit<32>
    "BwdBytes":           "bytes_t",       # bit<32>
    "FwdMaxPktLen":       "bit<16>",
    "BwdMaxPktLen":       "bit<16>",
    "FlagsSyn":           "bit<32>",
    "FlagsAck":           "bit<32>",
    "FlagsFin":           "bit<32>",
    "FlagsRst":           "bit<32>",
    "FlagsPsh":           "bit<32>",
    "MaxWinSize":         "bit<16>",
    "InitFwdWinBytes":    "bit<16>",
}

FEATURE_P4_WIDTH = {
    "SrcIP":              32,
    "DstIP":              32,
    "Protocol":           8,
    "SrcPort":            16,
    "DstPort":            16,
    "Duration":           48,
    "MaxIAT":             48,
    "FwdPktCount":        32,
    "BwdPktCount":        32,
    "FwdBytes":           32,
    "BwdBytes":           32,
    "FwdMaxPktLen":       16,
    "BwdMaxPktLen":       16,
    "FlagsSyn":           32,
    "FlagsAck":           32,
    "FlagsFin":           32,
    "FlagsRst":           32,
    "FlagsPsh":           32,
    "MaxWinSize":         16,
    "InitFwdWinBytes":    16,
}

# ─── Feature quantization for range-match compatibility ────────────────────
# Mirrors FEATURE_QUANTIZE from pipeline_utils.py.
# Format: { feature_name: (shift_amount, quantized_bits) }
FEATURE_QUANTIZE = {
    "Duration":           (20, 16),
    "MaxIAT":             (20, 16),
    "FwdPktCount":        (0,  16),
    "BwdPktCount":        (0,  16),
    "FwdBytes":           (4,  16),
    "BwdBytes":           (4,  16),
    "FlagsSyn":           (0,   8),
    "FlagsAck":           (0,  16),
    "FlagsFin":           (0,   8),
    "FlagsRst":           (0,   8),
    "FlagsPsh":           (0,  16),
}

# Map feature name → quantized P4 metadata field name (without meta. prefix)
FEATURE_TO_META_Q = {}
for _feat, _meta in FEATURE_TO_META.items():
    if _feat in FEATURE_QUANTIZE:
        FEATURE_TO_META_Q[_feat] = _meta + '_q'
    else:
        FEATURE_TO_META_Q[_feat] = _meta


class P4CodeGenerator:
    def __init__(self, n_components=2, bits=16, output_file='basic.p4',
                 model_type='dt', rf_params=None,
                 n_registers=65536, flow_timeout_s=20, active_timeout_s=60,
                 reduction_config=None):
        self.n_components    = n_components
        self.bits            = bits
        self.output_file     = output_file
        self.model_type      = model_type
        self.rf_params       = rf_params  or {}
        self.n_registers     = n_registers
        self.flow_timeout_ns = int(flow_timeout_s * 1_000_000_000)
        self.active_timeout_ns = int(active_timeout_s * 1_000_000_000)
        self.reduction_config = reduction_config or {}

        # Derived from reduction_config
        method = self.reduction_config.get('method', 'pca')
        self.needs_transform = self.reduction_config.get('needs_transform_tables', True)

        # Code prefix: "pc" for PCA, "ld" for LDA, "ae" for Autoencoder.
        # 'raw' skips the transform stage entirely (paper Section 4.2 baseline).
        # 'pca_linear' is PCA implemented as additive per-feature contribution
        # tables (single-field range keys) instead of a multivariate surrogate.
        _KNOWN_METHODS = {'pca', 'pca_linear', 'lda', 'lda_linear', 'autoencoder', 'raw'}
        if method not in _KNOWN_METHODS:
            import warnings
            warnings.warn(
                f"Unknown reduction method '{method}' in reduction_config.json — defaulting to PCA. "
                f"Known methods: {sorted(_KNOWN_METHODS)}", stacklevel=2)

        # PCA-linear: additive per-feature projection (see 2_pca_linear_entries.py)
        self.linear = method in ('pca_linear', 'lda_linear')
        self.linear_params = self.reduction_config.get('linear', {}) if self.linear else {}

        if method in ('lda', 'lda_linear'):
            self.code_prefix = 'ld'
        elif method == 'autoencoder':
            self.code_prefix = 'ae'
        elif method == 'raw':
            self.code_prefix = 'raw'
        else:
            self.code_prefix = 'pc'   # pca and pca_linear both emit pc*_code

        # Transform table/action prefixes — each method gets its own distinct table name
        if method == 'lda':
            self.table_prefix = 'lda'
            self.action_prefix = 'ld'
        elif method == 'autoencoder':
            self.table_prefix = 'ae'
            self.action_prefix = 'ae'
        elif method == 'raw':
            self.table_prefix = 'raw'
            self.action_prefix = 'raw'
        elif method in ('pca_linear', 'lda_linear'):
            self.table_prefix = 'featc'
            self.action_prefix = 'addc'
        else:
            self.table_prefix = 'pca'
            self.action_prefix = 'pc'

        # Classifier feature names (what the ml_code / rf_tree keys match on)
        self.classifier_features = self.reduction_config.get('feature_columns', None)
        if self.classifier_features is None:
            # Fallback: pc*_code
            self.classifier_features = [f'PC{i+1}_code' for i in range(n_components)]

    def _meta_field_for_feature(self, feat_name):
        """Map a feature name to a P4 meta.* field reference.

        Raw-feature classifier mode trains the DT/RF on the same quantised
        widths the data plane carries (FEATURE_QUANTIZE), so the table key
        must reference the *_q metadata field, not the full-width raw field.
        """
        if feat_name in FEATURE_TO_META:
            if feat_name in FEATURE_QUANTIZE:
                return f'meta.{FEATURE_TO_META_Q[feat_name]}'
            return f'meta.{FEATURE_TO_META[feat_name]}'
        # Transform code (PC1_code, LD2_code, AE3_code, etc.)
        return f'meta.{feat_name.lower()}'

    def _feature_bit_width(self, feat_name):
        if feat_name in FEATURE_P4_WIDTH:
            if feat_name in FEATURE_QUANTIZE:
                return FEATURE_QUANTIZE[feat_name][1]
            return FEATURE_P4_WIDTH[feat_name]
        return int(self.bits or 16)

    # ─────────────────────────────────────────────────────────────────────
    def generate_header(self):
        return '''/* -*- P4_16 -*- */
/*
 * P4 Flow-Based ML Classification
 * Auto-generated - supports PCA / LDA / Autoencoder + DT / RF
 */

#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4       = 0x800;
const bit<16> TYPE_ARP        = 0x0806;
const bit<8>  TYPE_TCP        = 6;
const bit<8>  TYPE_UDP        = 17;
const bit<8>  TYPE_ICMP       = 1;
const bit<8>  TYPE_ARP_PSEUDO = 253;  // pseudo proto used in flow key for ARP

const bit<32> NB_ENTRIES = ''' + str(self.n_registers) + ''';
const bit<32> MAX_REGISTER_ENTRIES = ''' + str(self.n_registers) + ''';

#define FLOW_TIMEOUT ''' + str(self.flow_timeout_ns) + '''  // ''' + str(self.flow_timeout_ns // 1_000_000_000) + '''s idle timeout in nanoseconds
#define ACTIVE_TIMEOUT ''' + str(self.active_timeout_ns) + '''  // ''' + str(self.active_timeout_ns // 1_000_000_000) + '''s active timeout in nanoseconds

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

typedef bit<48> iat_t;
typedef bit<48> duration_t;
typedef bit<16> port_t;
typedef bit<32> bytes_t;
''' + (f'typedef bit<{self.bits}> pca_code_t;   // Quantized code ({self.code_prefix.upper()})\n'
       if self.needs_transform else '') + '''typedef bit<8>  inference_result_t;

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

header icmp_t {
    bit<8>  icmp_type;   // ICMP type (used as pseudo src_port in flow key)
    bit<8>  icmp_code;   // ICMP code (used as pseudo dst_port in flow key)
    bit<16> checksum;
    bit<32> rest;        // identifier+seq_num for echo; varies by type
}

// ARP for IPv4-over-Ethernet (fixed 28-byte payload)
header arp_ipv4_t {
    bit<16>   htype;  // hardware type  (1 = Ethernet)
    bit<16>   ptype;  // protocol type  (0x0800 = IPv4)
    bit<8>    hlen;   // hardware addr length (6)
    bit<8>    plen;   // protocol addr length (4)
    bit<16>   oper;   // operation: 1=request, 2=reply  (pseudo src_port)
    macAddr_t sha;    // sender hardware address
    ip4Addr_t spa;    // sender protocol address  (→ meta.src_ip)
    macAddr_t tha;    // target hardware address
    ip4Addr_t tpa;    // target protocol address  (→ meta.dst_ip)
}
'''

    # ─────────────────────────────────────────────────────────────────────
    def generate_metadata(self):
        pfx = self.code_prefix
        code = '''
struct metadata {
    // Flow identification (5-tuple)
    ip4Addr_t src_ip;
    ip4Addr_t dst_ip;
    port_t src_port;
    port_t dst_port;
    bit<8>  protocol;

    // Canonical bidirectional flow key
    ip4Addr_t canon_src_ip;
    ip4Addr_t canon_dst_ip;
    macAddr_t canon_src_mac;
    macAddr_t canon_dst_mac;
    port_t    canon_src_port;
    port_t    canon_dst_port;
    bit<1>    is_reverse_dir;

    // Flow state tracking
    bit<32> flow_hash;
    bit<32> flow_hash_2;
    bit<1>  is_first_packet;
    bit<1>  hash_collision;
    bit<1>  flow_ended;

    // Flow-based features (Table 2 of the paper)
    duration_t duration;
    iat_t      max_iat;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    fwd_max_pkt_len;
    bit<16>    bwd_max_pkt_len;
    bit<32>    flags_syn;
    bit<32>    flags_ack;
    bit<32>    flags_fin;
    bit<32>    flags_rst;
    bit<32>    flags_psh;
    bit<16>    max_win_size;
    bit<16>    init_fwd_win;
    bytes_t    pkt_len;      // IP totalLen for IPv4; 28 for ARP (fixed); used for byte counting

    // Quantized features for range-match tables
'''
        for feat in FLOW_FEATURES:
            if feat in FEATURE_QUANTIZE:
                shift, qbits = FEATURE_QUANTIZE[feat]
                meta_q = FEATURE_TO_META_Q[feat]
                code += f'    bit<{qbits:>2}> {meta_q:20s};  // {FEATURE_TO_META[feat]} >> {shift}\n'

        # Transformed feature codes (PCA, LDA, or Autoencoder)
        if self.needs_transform:
            code += f'\n    // {pfx.upper()} transformed features (quantized)\n'
            for i in range(1, self.n_components + 1):
                code += f'    pca_code_t {pfx}{i}_code;\n'

        # PCA-linear: signed accumulators that sum per-feature contributions
        if self.linear:
            acc_w = int(self.linear_params.get('acc_width', 64))
            code += f'\n    // PCA-linear signed accumulators (summed contributions)\n'
            for i in range(1, self.n_components + 1):
                code += f'    int<{acc_w}> {pfx}{i}_acc;\n'

        code += '''
    // Classification result
    inference_result_t ml_result;

    // Timestamp
    bit<48> ingress_timestamp;
'''
        # RF packed votes
        if self.model_type == 'rf':
            n_est     = self.rf_params.get('n_estimators', 4)
            vote_bits = self.rf_params.get('vote_bits', 2)
            total_vb  = n_est * vote_bits
            code += f'\n    // RF packed vote field ({n_est} trees x {vote_bits} bits)\n'
            code += f'    bit<{total_vb}> rf_votes;\n'

        code += '''}

struct headers {
    ethernet_t   ethernet;
    arp_ipv4_t   arp;
    ipv4_t       ipv4;
    icmp_t       icmp;
    tcp_t        tcp;
    udp_t        udp;
}

struct digest_t {
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
    macAddr_t srcMAC;
    macAddr_t dstMAC;
    port_t srcPort;
    port_t dstPort;
    bit<8>  protocol;

    duration_t duration;
    iat_t      max_iat;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    fwd_max_pkt_len;
    bit<16>    bwd_max_pkt_len;
    bit<32>    flags_syn;
    bit<32>    flags_ack;
    bit<32>    flags_fin;
    bit<32>    flags_rst;
    bit<32>    flags_psh;
    bit<16>    max_win_size;
    bit<16>    init_fwd_win;

'''
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'    pca_code_t {pfx}{i}_code;\n'

        code += '''
    inference_result_t ml_result;
}
'''
        return code

    # ─────────────────────────────────────────────────────────────────────
    def generate_parser(self):
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
            TYPE_ARP : parse_arp;
            default  : accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        meta.src_ip   = hdr.ipv4.srcAddr;
        meta.dst_ip   = hdr.ipv4.dstAddr;
        meta.protocol = hdr.ipv4.protocol;
        meta.pkt_len  = (bytes_t)hdr.ipv4.totalLen;
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP : parse_tcp;
            TYPE_UDP : parse_udp;
            TYPE_ICMP: parse_icmp;
            default  : accept;
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

    state parse_icmp {
        packet.extract(hdr.icmp);
        meta.src_port = (port_t)hdr.icmp.icmp_type;
        meta.dst_port = (port_t)hdr.icmp.icmp_code;
        transition accept;
    }

    state parse_arp {
        packet.extract(hdr.arp);
        meta.src_ip   = hdr.arp.spa;
        meta.dst_ip   = hdr.arp.tpa;
        meta.protocol = TYPE_ARP_PSEUDO;
        meta.src_port = hdr.arp.oper;
        meta.dst_port = 16w0;
        meta.pkt_len  = 32w28;  // ARP IPv4 payload is fixed 28 bytes
        transition accept;
    }
}

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}
'''

    # ─────────────────────────────────────────────────────────────────────
    def generate_ingress_forwarding(self):
        pfx = self.code_prefix
        code = '''
/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    // Registers for flow state tracking
    register<bit<48>>(MAX_REGISTER_ENTRIES) reg_time_first_pkt;
    register<bit<48>>(MAX_REGISTER_ENTRIES) reg_time_last_pkt;
    register<iat_t>(MAX_REGISTER_ENTRIES)   reg_max_iat;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_fwd_pkt_count;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_bwd_pkt_count;
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_fwd_bytes;
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_bwd_bytes;
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_max_win_size;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_syn;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_ack;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_fin;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_rst;
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_fwd_max_pkt_len;
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_bwd_max_pkt_len;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_psh;
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_init_fwd_win;

    register<bit<1>>(MAX_REGISTER_ENTRIES) bloom_filter_1;  // indexed by CRC16 hash
    register<bit<1>>(MAX_REGISTER_ENTRIES) bloom_filter_2;  // indexed by CRC32 hash
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flow_hash_2;  // stores CRC32 at CRC16 slot (Bloom filter slot bookkeeping)
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_canon_src_port;  // canonical src port (written on new flow, reset on FIN/RST/timeout)
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_canon_dst_port;  // canonical dst port
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_canon_src_ip;   // canonical src IP
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_canon_dst_ip;   // canonical dst IP
    register<bit<48>>(MAX_REGISTER_ENTRIES) reg_canon_src_mac;  // canonical src MAC
    register<bit<48>>(MAX_REGISTER_ENTRIES) reg_canon_dst_mac;  // canonical dst MAC
    register<bit<8>>(MAX_REGISTER_ENTRIES)  reg_protocol;       // IP protocol

    // Amortized drain: scan index cycles through all register slots.
    // Each packet checks one extra slot for stale flows, so after
    // ~MAX_REGISTER_ENTRIES packets all slots have been scanned.
    register<bit<32>>(1) reg_scan_idx;

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

    action compute_flow_hash() {
        if (meta.src_ip < meta.dst_ip) {
            meta.canon_src_ip   = meta.src_ip;
            meta.canon_dst_ip   = meta.dst_ip;
            meta.canon_src_mac  = hdr.ethernet.srcAddr;
            meta.canon_dst_mac  = hdr.ethernet.dstAddr;
            meta.canon_src_port = meta.src_port;
            meta.canon_dst_port = meta.dst_port;
            meta.is_reverse_dir = 1w0;
        } else if (meta.src_ip > meta.dst_ip) {
            meta.canon_src_ip   = meta.dst_ip;
            meta.canon_dst_ip   = meta.src_ip;
            meta.canon_src_mac  = hdr.ethernet.dstAddr;
            meta.canon_dst_mac  = hdr.ethernet.srcAddr;
            meta.canon_src_port = meta.dst_port;
            meta.canon_dst_port = meta.src_port;
            meta.is_reverse_dir = 1w1;
        } else {
            if (meta.src_port <= meta.dst_port) {
                meta.canon_src_ip   = meta.src_ip;
                meta.canon_dst_ip   = meta.dst_ip;
                meta.canon_src_mac  = hdr.ethernet.srcAddr;
                meta.canon_dst_mac  = hdr.ethernet.dstAddr;
                meta.canon_src_port = meta.src_port;
                meta.canon_dst_port = meta.dst_port;
                meta.is_reverse_dir = 1w0;
            } else {
                meta.canon_src_ip   = meta.dst_ip;
                meta.canon_dst_ip   = meta.src_ip;
                meta.canon_src_mac  = hdr.ethernet.dstAddr;
                meta.canon_dst_mac  = hdr.ethernet.srcAddr;
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
        // Bloom filter collision detection:
        // bf1 slot occupied (1) but bf2 fingerprint absent (0) → different flow at this slot
        bit<1> bf_val_1;
        bit<1> bf_val_2;
        bloom_filter_1.read(bf_val_1, meta.flow_hash);
        bloom_filter_2.read(bf_val_2, meta.flow_hash_2);
        if (bf_val_1 == 1w1 && bf_val_2 == 1w0) {
            meta.hash_collision = 1w1;
        } else {
            meta.hash_collision = 1w0;
        }
    }

    action read_and_timeout_check() {
        bit<48> current_time_us = standard_metadata.ingress_global_timestamp;
        bit<48> current_time = current_time_us * 1000;
        // Save current packet's MACs before any register read can overwrite meta
        bit<48> pkt_src_mac = meta.canon_src_mac;
        bit<48> pkt_dst_mac = meta.canon_dst_mac;
        bit<48> time_first;
        bit<48> time_last;
        iat_t   max_iat;
        bit<32> fwd_pkt_count;
        bit<32> bwd_pkt_count;
        bytes_t fwd_bytes;
        bytes_t bwd_bytes;
        bit<16> max_win_size;
        bit<32> flags_syn;
        bit<32> flags_ack;
        bit<32> flags_fin;
        bit<32> flags_rst;
        bit<16> fwd_max_pkt_len;
        bit<16> bwd_max_pkt_len;
        bit<32> flags_psh;
        bit<16> init_fwd_win;

        reg_time_first_pkt.read(time_first, meta.flow_hash);
        reg_time_last_pkt.read(time_last, meta.flow_hash);
        reg_max_iat.read(max_iat, meta.flow_hash);
        reg_fwd_pkt_count.read(fwd_pkt_count, meta.flow_hash);
        reg_bwd_pkt_count.read(bwd_pkt_count, meta.flow_hash);
        reg_fwd_bytes.read(fwd_bytes, meta.flow_hash);
        reg_bwd_bytes.read(bwd_bytes, meta.flow_hash);
        reg_max_win_size.read(max_win_size, meta.flow_hash);
        reg_flags_syn.read(flags_syn, meta.flow_hash);
        reg_flags_ack.read(flags_ack, meta.flow_hash);
        reg_flags_fin.read(flags_fin, meta.flow_hash);
        reg_flags_rst.read(flags_rst, meta.flow_hash);
        reg_fwd_max_pkt_len.read(fwd_max_pkt_len, meta.flow_hash);
        reg_bwd_max_pkt_len.read(bwd_max_pkt_len, meta.flow_hash);
        reg_flags_psh.read(flags_psh, meta.flow_hash);
        reg_init_fwd_win.read(init_fwd_win, meta.flow_hash);
        // Read stored MACs into locals — do NOT overwrite meta.canon_src_mac
        // which was correctly set by compute_flow_hash() from the current packet.
        bit<48> stored_src_mac;
        bit<48> stored_dst_mac;
        reg_canon_src_mac.read(stored_src_mac, meta.flow_hash);
        reg_canon_dst_mac.read(stored_dst_mac, meta.flow_hash);

        // Idle timeout — previous flow on this slot has been idle
        if (time_first != 0 && time_last != 0 &&
                (current_time - time_last) > FLOW_TIMEOUT) {
            if ((fwd_pkt_count + bwd_pkt_count) >= 1) {
                meta.flow_ended         = 1w1;
                meta.duration           = time_last - time_first;
                meta.max_iat            = max_iat;
                meta.fwd_pkt_count      = fwd_pkt_count;
                meta.bwd_pkt_count      = bwd_pkt_count;
                meta.fwd_bytes          = fwd_bytes;
                meta.bwd_bytes          = bwd_bytes;
                meta.max_win_size       = max_win_size;
                meta.flags_syn          = flags_syn;
                meta.flags_ack          = flags_ack;
                meta.flags_fin          = flags_fin;
                meta.flags_rst          = flags_rst;
                meta.fwd_max_pkt_len    = fwd_max_pkt_len;
                meta.bwd_max_pkt_len    = bwd_max_pkt_len;
                meta.flags_psh          = flags_psh;
                meta.init_fwd_win       = init_fwd_win;
                meta.canon_src_mac      = stored_src_mac;
                meta.canon_dst_mac      = stored_dst_mac;
            }
            // Reset ALL registers for the new flow (regardless of pkt count)
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            reg_time_last_pkt.write(meta.flow_hash, 0);  // 0 so update_packet_stats skips IAT for 1st pkt
            reg_max_iat.write(meta.flow_hash, 0);
            reg_fwd_pkt_count.write(meta.flow_hash, 0);
            reg_bwd_pkt_count.write(meta.flow_hash, 0);
            reg_fwd_bytes.write(meta.flow_hash, 0);
            reg_bwd_bytes.write(meta.flow_hash, 0);
            reg_max_win_size.write(meta.flow_hash, 0);
            reg_flags_syn.write(meta.flow_hash, 32w0);
            reg_flags_ack.write(meta.flow_hash, 32w0);
            reg_flags_fin.write(meta.flow_hash, 32w0);
            reg_flags_rst.write(meta.flow_hash, 32w0);
            reg_fwd_max_pkt_len.write(meta.flow_hash, 16w0);
            reg_bwd_max_pkt_len.write(meta.flow_hash, 16w0);
            reg_flags_psh.write(meta.flow_hash, 32w0);
            reg_init_fwd_win.write(meta.flow_hash, 16w0);
            bloom_filter_1.write(meta.flow_hash,   1w1);
            bloom_filter_2.write(meta.flow_hash_2, 1w1);
            reg_flow_hash_2.write(meta.flow_hash, meta.flow_hash_2);
            reg_canon_src_port.write(meta.flow_hash, meta.canon_src_port);
            reg_canon_dst_port.write(meta.flow_hash, meta.canon_dst_port);
            reg_canon_src_ip.write(meta.flow_hash, meta.canon_src_ip);
            reg_canon_dst_ip.write(meta.flow_hash, meta.canon_dst_ip);
            reg_canon_src_mac.write(meta.flow_hash, pkt_src_mac);
            reg_canon_dst_mac.write(meta.flow_hash, pkt_dst_mac);
            reg_protocol.write(meta.flow_hash, meta.protocol);
        }
        // Active timeout — flow has been running longer than ACTIVE_TIMEOUT
        else if (time_first != 0 && time_last != 0 &&
                (current_time - time_first) > ACTIVE_TIMEOUT) {
            if ((fwd_pkt_count + bwd_pkt_count) >= 1) {
                meta.flow_ended         = 1w1;
                meta.duration           = time_last - time_first;
                meta.max_iat            = max_iat;
                meta.fwd_pkt_count      = fwd_pkt_count;
                meta.bwd_pkt_count      = bwd_pkt_count;
                meta.fwd_bytes          = fwd_bytes;
                meta.bwd_bytes          = bwd_bytes;
                meta.max_win_size       = max_win_size;
                meta.flags_syn          = flags_syn;
                meta.flags_ack          = flags_ack;
                meta.flags_fin          = flags_fin;
                meta.flags_rst          = flags_rst;
                meta.fwd_max_pkt_len    = fwd_max_pkt_len;
                meta.bwd_max_pkt_len    = bwd_max_pkt_len;
                meta.flags_psh          = flags_psh;
                meta.init_fwd_win       = init_fwd_win;
                meta.canon_src_mac      = stored_src_mac;
                meta.canon_dst_mac      = stored_dst_mac;
            }
            // Reset registers for new flow
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            reg_time_last_pkt.write(meta.flow_hash, 0);
            reg_max_iat.write(meta.flow_hash, 0);
            reg_fwd_pkt_count.write(meta.flow_hash, 0);
            reg_bwd_pkt_count.write(meta.flow_hash, 0);
            reg_fwd_bytes.write(meta.flow_hash, 0);
            reg_bwd_bytes.write(meta.flow_hash, 0);
            reg_max_win_size.write(meta.flow_hash, 0);
            reg_flags_syn.write(meta.flow_hash, 32w0);
            reg_flags_ack.write(meta.flow_hash, 32w0);
            reg_flags_fin.write(meta.flow_hash, 32w0);
            reg_flags_rst.write(meta.flow_hash, 32w0);
            reg_fwd_max_pkt_len.write(meta.flow_hash, 16w0);
            reg_bwd_max_pkt_len.write(meta.flow_hash, 16w0);
            reg_flags_psh.write(meta.flow_hash, 32w0);
            reg_init_fwd_win.write(meta.flow_hash, 16w0);
            bloom_filter_1.write(meta.flow_hash,   1w1);
            bloom_filter_2.write(meta.flow_hash_2, 1w1);
            reg_flow_hash_2.write(meta.flow_hash, meta.flow_hash_2);
            reg_canon_src_port.write(meta.flow_hash, meta.canon_src_port);
            reg_canon_dst_port.write(meta.flow_hash, meta.canon_dst_port);
            reg_canon_src_ip.write(meta.flow_hash, meta.canon_src_ip);
            reg_canon_dst_ip.write(meta.flow_hash, meta.canon_dst_ip);
            reg_canon_src_mac.write(meta.flow_hash, pkt_src_mac);
            reg_canon_dst_mac.write(meta.flow_hash, pkt_dst_mac);
            reg_protocol.write(meta.flow_hash, meta.protocol);
        }
    }

    action update_packet_stats() {
        bit<48> current_time_us = standard_metadata.ingress_global_timestamp;
        bit<48> current_time = current_time_us * 1000;
        bit<48> time_first;
        bit<48> time_last;
        iat_t   max_iat;
        bit<32> fwd_pkt_count;
        bit<32> bwd_pkt_count;
        bytes_t fwd_bytes;
        bytes_t bwd_bytes;
        bit<16> max_win_size;
        bit<32> flags_syn;
        bit<32> flags_ack;
        bit<32> flags_fin;
        bit<32> flags_rst;
        bit<16> fwd_max_pkt_len;
        bit<16> bwd_max_pkt_len;
        bit<32> flags_psh;
        bit<16> init_fwd_win;

        reg_time_first_pkt.read(time_first, meta.flow_hash);
        reg_time_last_pkt.read(time_last, meta.flow_hash);
        reg_max_iat.read(max_iat, meta.flow_hash);
        reg_fwd_pkt_count.read(fwd_pkt_count, meta.flow_hash);
        reg_bwd_pkt_count.read(bwd_pkt_count, meta.flow_hash);
        reg_fwd_bytes.read(fwd_bytes, meta.flow_hash);
        reg_bwd_bytes.read(bwd_bytes, meta.flow_hash);
        reg_max_win_size.read(max_win_size, meta.flow_hash);
        reg_flags_syn.read(flags_syn, meta.flow_hash);
        reg_flags_ack.read(flags_ack, meta.flow_hash);
        reg_flags_fin.read(flags_fin, meta.flow_hash);
        reg_flags_rst.read(flags_rst, meta.flow_hash);
        reg_fwd_max_pkt_len.read(fwd_max_pkt_len, meta.flow_hash);
        reg_bwd_max_pkt_len.read(bwd_max_pkt_len, meta.flow_hash);
        reg_flags_psh.read(flags_psh, meta.flow_hash);
        reg_init_fwd_win.read(init_fwd_win, meta.flow_hash);
        // NOTE: Do NOT read reg_canon_src_mac/dst_mac into meta here.
        // meta.canon_src_mac was correctly set by compute_flow_hash() and
        // overwriting it with an uninitialized register (0) corrupts MAC
        // for new flows and PCA table lookups.

        // First packet for a new flow (fresh empty slot)
        if (time_first == 0) {
            time_first = current_time;
            meta.is_first_packet = 1w1;
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            bloom_filter_1.write(meta.flow_hash,   1w1);
            bloom_filter_2.write(meta.flow_hash_2, 1w1);
            reg_flow_hash_2.write(meta.flow_hash, meta.flow_hash_2);
            reg_canon_src_port.write(meta.flow_hash, meta.canon_src_port);
            reg_canon_dst_port.write(meta.flow_hash, meta.canon_dst_port);
            reg_canon_src_ip.write(meta.flow_hash, meta.canon_src_ip);
            reg_canon_dst_ip.write(meta.flow_hash, meta.canon_dst_ip);
            reg_canon_src_mac.write(meta.flow_hash, meta.canon_src_mac);
            reg_canon_dst_mac.write(meta.flow_hash, meta.canon_dst_mac);
            reg_protocol.write(meta.flow_hash, meta.protocol);
        }

        // IAT update (MaxIAT only — paper Table 2)
        if (time_last != 0) {
            iat_t current_iat = current_time - time_last;
            if (current_iat > max_iat) {
                max_iat = current_iat;
                reg_max_iat.write(meta.flow_hash, max_iat);
            }
        }
        if (meta.flow_ended == 1w0) {
            meta.max_iat = max_iat;
        }

        // Direction-based counters + max packet length per direction
        if (meta.is_reverse_dir == 1w0) {
            fwd_pkt_count = fwd_pkt_count + 1;
            fwd_bytes = fwd_bytes + meta.pkt_len;
            reg_fwd_pkt_count.write(meta.flow_hash, fwd_pkt_count);
            reg_fwd_bytes.write(meta.flow_hash, fwd_bytes);
            if ((bit<16>)meta.pkt_len > fwd_max_pkt_len) {
                fwd_max_pkt_len = (bit<16>)meta.pkt_len;
                reg_fwd_max_pkt_len.write(meta.flow_hash, fwd_max_pkt_len);
            }
            // InitFwdWinBytes: capture on first forward TCP packet (mirrors Python)
            if (meta.protocol == TYPE_TCP && init_fwd_win == 16w0) {
                init_fwd_win = hdr.tcp.window;
                reg_init_fwd_win.write(meta.flow_hash, init_fwd_win);
            }
        } else {
            bwd_pkt_count = bwd_pkt_count + 1;
            bwd_bytes = bwd_bytes + meta.pkt_len;
            reg_bwd_pkt_count.write(meta.flow_hash, bwd_pkt_count);
            reg_bwd_bytes.write(meta.flow_hash, bwd_bytes);
            if ((bit<16>)meta.pkt_len > bwd_max_pkt_len) {
                bwd_max_pkt_len = (bit<16>)meta.pkt_len;
                reg_bwd_max_pkt_len.write(meta.flow_hash, bwd_max_pkt_len);
            }
        }
        if (meta.flow_ended == 1w0) {
            meta.fwd_pkt_count   = fwd_pkt_count;
            meta.bwd_pkt_count   = bwd_pkt_count;
            meta.fwd_bytes       = fwd_bytes;
            meta.bwd_bytes       = bwd_bytes;
            meta.fwd_max_pkt_len = fwd_max_pkt_len;
            meta.bwd_max_pkt_len = bwd_max_pkt_len;
        }

        // Window size (max)
        if (meta.protocol == TYPE_TCP) {
            if (hdr.tcp.window > max_win_size) {
                max_win_size = hdr.tcp.window;
                reg_max_win_size.write(meta.flow_hash, max_win_size);
            }
        }
        if (meta.flow_ended == 1w0) {
            meta.max_win_size = max_win_size;
            meta.init_fwd_win = init_fwd_win;
        }

        reg_time_last_pkt.write(meta.flow_hash, current_time);

        // TCP flag counts (SYN, ACK, FIN, RST, PSH)
        if (meta.protocol == TYPE_TCP) {
            flags_syn = flags_syn + (bit<32>)hdr.tcp.ctrl[1:1];
            flags_ack = flags_ack + (bit<32>)hdr.tcp.ctrl[4:4];
            flags_fin = flags_fin + (bit<32>)hdr.tcp.ctrl[0:0];
            flags_rst = flags_rst + (bit<32>)hdr.tcp.ctrl[2:2];
            flags_psh = flags_psh + (bit<32>)hdr.tcp.ctrl[3:3];
            reg_flags_syn.write(meta.flow_hash, flags_syn);
            reg_flags_ack.write(meta.flow_hash, flags_ack);
            reg_flags_fin.write(meta.flow_hash, flags_fin);
            reg_flags_rst.write(meta.flow_hash, flags_rst);
            reg_flags_psh.write(meta.flow_hash, flags_psh);
        }
        if (meta.flow_ended == 1w0) {
            meta.flags_syn = flags_syn;
            meta.flags_ack = flags_ack;
            meta.flags_fin = flags_fin;
            meta.flags_rst = flags_rst;
            meta.flags_psh = flags_psh;
        }

        // FIN/RST ends the flow
        if (meta.flow_ended == 1w0 &&
                meta.protocol == TYPE_TCP &&
                (meta.flags_fin > 32w0 || meta.flags_rst > 32w0)) {
            meta.flow_ended         = 1w1;
            meta.duration           = current_time - time_first;
            meta.max_iat            = max_iat;
            reg_fwd_pkt_count.read(meta.fwd_pkt_count, meta.flow_hash);
            reg_bwd_pkt_count.read(meta.bwd_pkt_count, meta.flow_hash);
            reg_fwd_bytes.read(meta.fwd_bytes, meta.flow_hash);
            reg_bwd_bytes.read(meta.bwd_bytes, meta.flow_hash);
            reg_max_win_size.read(meta.max_win_size, meta.flow_hash);
            reg_fwd_max_pkt_len.read(meta.fwd_max_pkt_len, meta.flow_hash);
            reg_bwd_max_pkt_len.read(meta.bwd_max_pkt_len, meta.flow_hash);
            reg_flags_psh.read(meta.flags_psh, meta.flow_hash);
            reg_init_fwd_win.read(meta.init_fwd_win, meta.flow_hash);
            // Reset registers
            reg_time_first_pkt.write(meta.flow_hash, 0);
            reg_time_last_pkt.write(meta.flow_hash, 0);
            reg_max_iat.write(meta.flow_hash, 0);
            reg_fwd_pkt_count.write(meta.flow_hash, 0);
            reg_bwd_pkt_count.write(meta.flow_hash, 0);
            reg_fwd_bytes.write(meta.flow_hash, 0);
            reg_bwd_bytes.write(meta.flow_hash, 0);
            reg_max_win_size.write(meta.flow_hash, 0);
            reg_flags_syn.write(meta.flow_hash, 32w0);
            reg_flags_ack.write(meta.flow_hash, 32w0);
            reg_flags_fin.write(meta.flow_hash, 32w0);
            reg_flags_rst.write(meta.flow_hash, 32w0);
            reg_fwd_max_pkt_len.write(meta.flow_hash, 16w0);
            reg_bwd_max_pkt_len.write(meta.flow_hash, 16w0);
            reg_flags_psh.write(meta.flow_hash, 32w0);
            reg_init_fwd_win.write(meta.flow_hash, 16w0);
            bloom_filter_1.write(meta.flow_hash,   1w0);  // release slot
            bloom_filter_2.write(meta.flow_hash_2, 1w0);
            reg_flow_hash_2.write(meta.flow_hash, 32w0);  // clear CRC32 bookmark
            reg_canon_src_port.write(meta.flow_hash, 16w0);  // clear port bookmark
            reg_canon_dst_port.write(meta.flow_hash, 16w0);
            reg_canon_src_ip.write(meta.flow_hash, 32w0);   // clear IP bookmark
            reg_canon_dst_ip.write(meta.flow_hash, 32w0);
            reg_canon_src_mac.write(meta.flow_hash, 48w0);  // clear MAC bookmark
            reg_canon_dst_mac.write(meta.flow_hash, 48w0);
            reg_protocol.write(meta.flow_hash, 8w0);        // clear protocol bookmark
        }
    }

'''

        # ── Transform tables ─────────────────────────────────────────────
        if self.needs_transform and self.linear:
            # PCA-linear: one single-field range table PER FEATURE.  Each table
            # adds that feature's K precomputed fixed-point contributions to the
            # K accumulators.  No feature shares a match key with another.
            acc_w = int(self.linear_params.get('acc_width', 64))
            dw    = int(self.linear_params.get('delta_width', 64))
            ncomp = self.n_components
            for feat in TRANSFORM_KEY_FEATURES:
                meta_f = FEATURE_TO_META_Q[feat]
                params = ', '.join(f'int<{dw}> d{j}' for j in range(1, ncomp + 1))
                adds = ''.join(
                    f'        meta.{pfx}{j}_acc = meta.{pfx}{j}_acc + (int<{acc_w}>)d{j};\n'
                    for j in range(1, ncomp + 1))
                code += f'''
    // Per-feature contribution: {feat}
    action addc_{meta_f}({params}) {{
{adds}    }}

    table featc_{meta_f} {{
        key = {{
            meta.{meta_f:20s}: range;
        }}
        actions = {{
            addc_{meta_f};
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''
        elif self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'''
    // {pfx.upper()} component {i} transformation
    action set_{self.action_prefix}{i}_code(pca_code_t code) {{
        meta.{pfx}{i}_code = code;
    }}

    table {self.table_prefix}_component{i} {{
        key = {{
'''
                for feat in TRANSFORM_KEY_FEATURES:
                    meta_f = FEATURE_TO_META_Q[feat]
                    code += f'            meta.{meta_f:20s}: range;\n'
                code += f'''        }}
        actions = {{
            set_{self.action_prefix}{i}_code;
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

        # ── Classification tables ────────────────────────────────────────
        if self.model_type in ('dt', 'rf'):
            code += '''
    // Shared classification result action
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }
'''

        # Helper: generate range-match key block from classifier features
        def _classifier_key_block():
            lines = ''
            for feat_name in self.classifier_features:
                meta_f = self._meta_field_for_feature(feat_name)
                lines += f'            {meta_f:30s}: range;\n'
            return lines

        if self.model_type == 'dt':
            code += '''
    // Decision Tree classification
    table ml_code {
        key = {
'''
            code += _classifier_key_block()
            code += '''        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }
'''

        elif self.model_type == 'rf':
            n_est     = self.rf_params.get('n_estimators', 4)
            vote_bits = self.rf_params.get('vote_bits', 2)
            n_classes = self.rf_params.get('n_classes', 2)
            rf_feats  = self.rf_params.get('feature_names', self.classifier_features)

            for i in range(n_est):
                lo_bit = i * vote_bits
                hi_bit = lo_bit + vote_bits - 1
                code += f'''
    action set_rf_tree_{i}_vote(bit<{vote_bits}> vote) {{
        meta.rf_votes[{hi_bit}:{lo_bit}] = vote;
    }}

    table rf_tree_{i} {{
        key = {{
'''
                for feat_name in rf_feats:
                    meta_f = self._meta_field_for_feature(feat_name)
                    code += f'            {meta_f:30s}: range;\n'
                code += f'''        }}
        actions = {{
            set_rf_tree_{i}_vote;
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

            total_vb = n_est * vote_bits
            vote_table_size = n_classes ** n_est  # actual entries = n_classes^n_est, not 2^total_vb
            code += f'''
    table rf_vote_classify {{
        key = {{
            meta.rf_votes : exact;
        }}
        actions = {{
            set_result;
            NoAction;
        }}
        size = {vote_table_size};
    }}
'''

        # ── Build classify+digest snippet (used for BOTH timeout and FIN/RST paths) ──
        classify_snippet = ''

        # Pre-quantize features for range-match tables.
        # Needed for BOTH raw-feature classification (ml_code reads meta.*_q
        # directly) AND for the PCA/LDA/Autoencoder transform tables. Earlier
        # versions of this codegen emitted the shift block only inside
        # `if self.needs_transform:`, so raw-mode `basic.p4` left every _q
        # field unassigned (default 0) and the live classifier matched the
        # wrong leaf. Always emit the shifts.
        classify_snippet += '\n                // Quantize features for range-match tables\n'
        for feat in TRANSFORM_KEY_FEATURES:
            if feat in FEATURE_QUANTIZE:
                shift, qbits = FEATURE_QUANTIZE[feat]
                raw_meta = FEATURE_TO_META[feat]
                q_meta = FEATURE_TO_META_Q[feat]
                if shift > 0:
                    classify_snippet += f'                meta.{q_meta} = (bit<{qbits}>)(meta.{raw_meta} >> {shift});\n'
                else:
                    classify_snippet += f'                meta.{q_meta} = (bit<{qbits}>)meta.{raw_meta};\n'

        # Transform tables (PCA/LDA/Autoencoder only)
        if self.needs_transform:
            if self.linear:
                acc_w   = int(self.linear_params.get('acc_width', 64))
                fp      = int(self.linear_params.get('fp_shift', 0))
                maxcode = int(self.linear_params.get('maxcode', (1 << self.bits) - 1))
                init    = self.linear_params.get('init', [0] * self.n_components)
                classify_snippet += '\n                // PCA-linear: seed accumulators with INIT_j\n'
                for j in range(1, self.n_components + 1):
                    classify_snippet += f'                meta.{pfx}{j}_acc = (int<{acc_w}>)({int(init[j-1])});\n'
                classify_snippet += '\n                // Add each feature\'s contribution\n'
                for feat in TRANSFORM_KEY_FEATURES:
                    meta_f = FEATURE_TO_META_Q[feat]
                    classify_snippet += f'                featc_{meta_f}.apply();\n'
                classify_snippet += '\n                // Shift back to code domain and clamp to [0, MAXCODE]\n'
                for j in range(1, self.n_components + 1):
                    classify_snippet += (
                        f'                int<{acc_w}> {pfx}{j}_t = meta.{pfx}{j}_acc >> {fp};\n'
                        f'                if ({pfx}{j}_t < (int<{acc_w}>)0) {{ {pfx}{j}_t = (int<{acc_w}>)0; }}\n'
                        f'                if ({pfx}{j}_t > (int<{acc_w}>){maxcode}) {{ {pfx}{j}_t = (int<{acc_w}>){maxcode}; }}\n'
                        # int<W> -> bit<W> (same width) -> pca_code_t (truncate); value is clamped to [0,MAXCODE]
                        f'                meta.{pfx}{j}_code = (pca_code_t)(bit<{acc_w}>){pfx}{j}_t;\n')
            else:
                classify_snippet += f'\n                // Apply {pfx.upper()} transformations\n'
                for i in range(1, self.n_components + 1):
                    classify_snippet += f'                {self.table_prefix}_component{i}.apply();\n'

        # Classifier
        classify_snippet += '\n                // Apply classifier\n'
        if self.model_type == 'dt':
            classify_snippet += '                ml_code.apply();\n'
        elif self.model_type == 'rf':
            n_est = self.rf_params.get('n_estimators', 4)
            total_vb = n_est * self.rf_params.get('vote_bits', 2)
            classify_snippet += f'                meta.rf_votes = {total_vb}w0;\n'
            for i in range(n_est):
                classify_snippet += f'                rf_tree_{i}.apply();\n'
            classify_snippet += '                rf_vote_classify.apply();\n'

        # Digest fields
        classify_snippet += '''
                // Send digest
                digest<digest_t>(1, {
                    meta.canon_src_ip,
                    meta.canon_dst_ip,
                    meta.canon_src_mac,
                    meta.canon_dst_mac,
                    meta.canon_src_port,
                    meta.canon_dst_port,
                    meta.protocol,
                    meta.duration,
                    meta.max_iat,
                    meta.fwd_pkt_count,
                    meta.bwd_pkt_count,
                    meta.fwd_bytes,
                    meta.bwd_bytes,
                    meta.fwd_max_pkt_len,
                    meta.bwd_max_pkt_len,
                    meta.flags_syn,
                    meta.flags_ack,
                    meta.flags_fin,
                    meta.flags_rst,
                    meta.flags_psh,
                    meta.max_win_size,
                    meta.init_fwd_win,
'''
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                classify_snippet += f'                    meta.{pfx}{i}_code,\n'
        classify_snippet += '                    meta.ml_result\n                });\n'

        # ── Apply block ──────────────────────────────────────────────────
        code += '''
    // Amortized drain: each packet scans one extra register slot for stale
    // flows whose timeout expired but no matching packet arrived to trigger
    // the normal read_and_timeout_check.  After MAX_REGISTER_ENTRIES packets
    // every slot has been scanned, matching the Python extractor's EOF flush.
    action scan_and_drain() {
        bit<48> current_time_us = standard_metadata.ingress_global_timestamp;
        bit<48> current_time = current_time_us * 1000;

        bit<32> scan_idx;
        reg_scan_idx.read(scan_idx, 0);
        bit<32> slot = scan_idx % MAX_REGISTER_ENTRIES;
        reg_scan_idx.write(0, scan_idx + 1);

        // Skip if this slot is the current packet's slot (already handled)
        if (slot == meta.flow_hash) {
            return;
        }

        bit<48> s_time_first;
        bit<48> s_time_last;
        reg_time_first_pkt.read(s_time_first, slot);
        reg_time_last_pkt.read(s_time_last, slot);

        if (s_time_first == 0 || s_time_last == 0) {
            return;
        }
        if ((current_time - s_time_last) <= FLOW_TIMEOUT &&
            (current_time - s_time_first) <= ACTIVE_TIMEOUT) {
            return;
        }

        // Stale/long-lived flow detected — read remaining registers
        bit<32> s_fwd_pkt; bit<32> s_bwd_pkt;
        reg_fwd_pkt_count.read(s_fwd_pkt, slot);
        reg_bwd_pkt_count.read(s_bwd_pkt, slot);
        if ((s_fwd_pkt + s_bwd_pkt) < 1) {
            // Empty flow — just clear the slot
            reg_time_first_pkt.write(slot, 0);
            reg_time_last_pkt.write(slot, 0);
            reg_fwd_pkt_count.write(slot, 0);
            reg_bwd_pkt_count.write(slot, 0);
            bloom_filter_1.write(slot, 1w0);
            return;
        }

        // Snapshot features into meta (overwrite current packet's meta)
        meta.flow_ended    = 1w1;
        meta.duration      = s_time_last - s_time_first;
        reg_max_iat.read(meta.max_iat, slot);
        meta.fwd_pkt_count = s_fwd_pkt;
        meta.bwd_pkt_count = s_bwd_pkt;
        reg_fwd_bytes.read(meta.fwd_bytes, slot);
        reg_bwd_bytes.read(meta.bwd_bytes, slot);
        reg_max_win_size.read(meta.max_win_size, slot);
        reg_flags_syn.read(meta.flags_syn, slot);
        reg_flags_ack.read(meta.flags_ack, slot);
        reg_flags_fin.read(meta.flags_fin, slot);
        reg_flags_rst.read(meta.flags_rst, slot);
        reg_fwd_max_pkt_len.read(meta.fwd_max_pkt_len, slot);
        reg_bwd_max_pkt_len.read(meta.bwd_max_pkt_len, slot);
        reg_flags_psh.read(meta.flags_psh, slot);
        reg_init_fwd_win.read(meta.init_fwd_win, slot);
        reg_canon_src_ip.read(meta.canon_src_ip, slot);
        reg_canon_dst_ip.read(meta.canon_dst_ip, slot);
        reg_canon_src_mac.read(meta.canon_src_mac, slot);
        reg_canon_dst_mac.read(meta.canon_dst_mac, slot);
        reg_canon_src_port.read(meta.canon_src_port, slot);
        reg_canon_dst_port.read(meta.canon_dst_port, slot);
        reg_protocol.read(meta.protocol, slot);

        // Clear the drained slot
        reg_time_first_pkt.write(slot, 0);
        reg_time_last_pkt.write(slot, 0);
        reg_max_iat.write(slot, 0);
        reg_fwd_pkt_count.write(slot, 0);
        reg_bwd_pkt_count.write(slot, 0);
        reg_fwd_bytes.write(slot, 0);
        reg_bwd_bytes.write(slot, 0);
        reg_max_win_size.write(slot, 0);
        reg_flags_syn.write(slot, 32w0);
        reg_flags_ack.write(slot, 32w0);
        reg_flags_fin.write(slot, 32w0);
        reg_flags_rst.write(slot, 32w0);
        reg_fwd_max_pkt_len.write(slot, 16w0);
        reg_bwd_max_pkt_len.write(slot, 16w0);
        reg_flags_psh.write(slot, 32w0);
        reg_init_fwd_win.write(slot, 16w0);
        bloom_filter_1.write(slot, 1w0);
        bit<32> s_h2;
        reg_flow_hash_2.read(s_h2, slot);
        bloom_filter_2.write(s_h2, 1w0);
        reg_flow_hash_2.write(slot, 32w0);
        reg_canon_src_port.write(slot, 16w0);
        reg_canon_dst_port.write(slot, 16w0);
        reg_canon_src_ip.write(slot, 32w0);
        reg_canon_dst_ip.write(slot, 32w0);
        reg_canon_src_mac.write(slot, 48w0);
        reg_canon_dst_mac.write(slot, 48w0);
        reg_protocol.write(slot, 8w0);
    }

    apply {
        if ((hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP ||
                                    meta.protocol == TYPE_ICMP)) ||
            hdr.arp.isValid()) {
            compute_flow_hash();
            // Step 1: read registers, check timeout, snapshot old flow features
            // into meta.* if timed out, then reset registers for the new flow.
            if (meta.hash_collision == 1w0) {
                read_and_timeout_check();
            }

            // Step 2: update stats for the current packet. meta.* writes inside
            // update_packet_stats() are guarded by (flow_ended == 1w0) so the
            // timeout snapshot in meta.* is preserved when the timer fired.
            if (meta.hash_collision == 1w0) {
                update_packet_stats();
            }

            // Step 3: scan_and_drain ONLY on drain-trigger packets
            // (src_ip = 10.255.255.254 = 0x0AFFFFFE).  During normal
            // traffic, flows end via FIN/RST or timeout-on-arrival only,
            // making flow boundaries deterministic across runs.
            // Each packet exports at most 1 stale flow (flow_ended guard).
            if (meta.src_ip == 32w0x0AFFFFFE) {
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
                if (meta.flow_ended == 1w0) { scan_and_drain(); }
            }

            // Classify if flow ended (timeout from read_and_timeout_check,
            // FIN/RST from update_packet_stats, or drain from scan_and_drain).
            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 1) {
'''
        code += classify_snippet
        code += '''
            } // end if flow_ended

            if (hdr.ipv4.isValid()) {
                ipv4_lpm.apply();
            }
        }
    }
}
'''
        return code

    # ─────────────────────────────────────────────────────────────────────
    def generate_egress_and_tail(self):
        return '''
control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply {  }
}

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
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

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.arp);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.icmp);
        packet.emit(hdr.tcp);
        packet.emit(hdr.udp);
    }
}

V1Switch(
    MyParser(),
    MyVerifyChecksum(),
    MyIngress(),
    MyEgress(),
    MyComputeChecksum(),
    MyDeparser()
) main;
'''

    # ─────────────────────────────────────────────────────────────────────
    def generate(self):
        method_label = self.reduction_config.get('method', 'pca').upper()
        logger.info(f"Generating P4: method={method_label}, model={self.model_type}, "
                    f"transform={self.needs_transform}, "
                    f"components={self.n_components}, bits={self.bits}")

        code = self.generate_header()
        code += self.generate_metadata()
        code += self.generate_parser()
        code += self.generate_ingress_forwarding()
        code += self.generate_egress_and_tail()
        return code

    def write_to_file(self):
        code = self.generate()
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        logger.info(f"Generated {self.output_file}")


# ─── Tofino P4 Code Generator ────────────────────────────────────────────
def load_reduction_config(config_file='tables/reduction_config.json'):
    """Load the universal reduction config written by any step 2."""
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            logger.info(f"Loaded reduction_config: method={cfg.get('method')}, "
                        f"features={cfg.get('feature_columns')}, "
                        f"transform={cfg.get('needs_transform_tables')}")
            return cfg
        except Exception as e:
            logger.warning(f"Could not read {config_file}: {e}")
    return None


def detect_n_components(params_file='tables/encoding_params.json',
                        commands_file='tables/s1-commands.txt',
                        table_prefix='pca'):
    """Detect component count and bit-width from transform encoding params."""
    if os.path.exists(params_file):
        try:
            with open(params_file) as f:
                params = json.load(f)
            n = params.get('n_components')
            bits = params.get('bits', 16)
            if n and bits:
                logger.info(f"Detected {n} components ({bits}-bit) from {params_file}")
                return n, bits
        except Exception as e:
            logger.warning(f"Could not read {params_file}: {e}")

    if os.path.exists(commands_file):
        try:
            with open(commands_file) as f:
                tables = set()
                for line in f:
                    m = re.search(rf'{table_prefix}_component(\d+)', line)
                    if m:
                        tables.add(int(m.group(1)))
            if tables:
                n = max(tables)
                logger.info(f"Detected {n} components from {commands_file}")
                return n, 16
        except Exception as e:
            logger.warning(f"Could not parse {commands_file}: {e}")

    logger.warning("Could not auto-detect components — defaulting to 2 (16-bit)")
    return 2, 16


def load_model_params(path='tables/model_params.json'):
    """Load universal model params. Returns {} if file missing."""
    if os.path.exists(path):
        try:
            with open(path) as f:
                p = json.load(f)
            mt = p.get('model_type', '?')
            logger.info(f"Model params ({mt}): {path}")
            return p
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    return {}


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = P4secArgumentParser(
        description='Generate BMv2 P4 code for ML classification',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Notes:\n"
            "  - Reduction method is read from tables/reduction_config.json.\n"
        )
    )
    parser.add_argument('--output', default='../basic.p4',
                        help='BMv2 P4 output (default: ../basic.p4)')
    parser.add_argument('--params-file', default='tables/encoding_params.json')
    parser.add_argument('--commands-file', default='tables/s1-commands.txt')
    parser.add_argument('--reduction-config', default='tables/reduction_config.json')
    parser.add_argument('-m', '--model-type', default='dt',
                        choices=['dt', 'rf'],
                        help='Classifier: dt | rf')
    parser.add_argument('--model-params', default='tables/model_params.json',
                        help='Universal model params JSON (default: tables/model_params.json)')
    parser.add_argument('--register-entries', type=int, default=65536)
    parser.add_argument('--flow-timeout-s', type=int, default=20)
    args = parser.parse_args()

    # Load universal reduction config (may be None for old PCA pipeline)
    red_cfg = load_reduction_config(args.reduction_config)

    # Determine n_components and bits
    _method = (red_cfg or {}).get('method', 'pca')
    _prefix_map = {'lda': 'lda', 'autoencoder': 'ae'}
    _table_prefix = _prefix_map.get(_method, 'pca')
    if red_cfg and red_cfg.get('needs_transform_tables', True):
        n_components, bits = detect_n_components(args.params_file, args.commands_file, table_prefix=_table_prefix)
    elif red_cfg and not red_cfg.get('needs_transform_tables', True):
        n_components = red_cfg.get('n_components', 0)
        bits = 16
    else:
        n_components, bits = detect_n_components(args.params_file, args.commands_file, table_prefix=_table_prefix)

    # Load universal model params — dispatch to the right internal dict by model_type
    _all_params = load_model_params(args.model_params) if args.model_type == 'rf' else {}
    rf_params  = _all_params if args.model_type == 'rf'  else {}

    generator = P4CodeGenerator(
        n_components=n_components,
        bits=bits,
        output_file=args.output,
        model_type=args.model_type,
        rf_params=rf_params,
        n_registers=args.register_entries,
        flow_timeout_s=args.flow_timeout_s,
        reduction_config=red_cfg or {},
    )
    generator.write_to_file()

    method_str = (red_cfg or {}).get('method', 'pca').upper()
    logger.info(f"\nGeneration complete!")
    logger.info(f"  Reduction : {method_str}")
    logger.info(f"  Model     : {args.model_type.upper()}")
    logger.info(f"  Transform : {'yes' if generator.needs_transform else 'no (direct features)'}")


if __name__ == '__main__':
    main()
