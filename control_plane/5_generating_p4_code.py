#!/usr/bin/env python3
"""
Universal P4 Code Generator for ML Classification with Flow-Based Features.

Supports five dimensionality-reduction methods:
    PCA               → pca_component* transform tables + classifier on pc*_code
    LDA               → lda_component* transform tables + classifier on ld*_code
    Autoencoder       → ae_component*  transform tables + classifier on ae*_code
    UMAP              → umap_component* transform tables + classifier on um*_code
    Feature Selection → NO transform tables, classifier matches raw flow features

Supports six classifier back-ends:
  --model-type dt   DecisionTree    single ml_code table
  --model-type rf   RandomForest    N rf_tree_i tables + rf_vote_classify
  --model-type xgb  XGBoost         N*K xgb_tree_i tables + xgb_classify
  --model-type gb   GradientBoost   (same P4 architecture as XGB)
  --model-type knn  KNN             (deploys as DT proxy in P4)
  --model-type svm  SVM             (deploys as DT proxy in P4)
  --model-type cnn  1D CNN          neural lookup tables (P4 deployable)

Reads configuration from:
  tables/reduction_config.json     (written by any step 2 — method, features, max values)
    tables/encoding_params.json      (transform encoding params — fallback)
  tables/rf_params.json            (RF metadata)
  tables/xgb_params.json           (XGB/GB metadata)
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

# ─── All 13 P4 raw flow features ────────────────────────────────────────────
FLOW_FEATURES = [
    "Protocol", "SrcPort", "DstPort",
    "Duration", "MaxIAT", "UrgCount",
    "FwdPktCount", "BwdPktCount", "FwdBytes", "BwdBytes",
    "MaxWinSize", "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst",
    "MinIAT", "FwdMaxPktLen", "BwdMaxPktLen", "FlagsPsh", "InitFwdWinBytes",
]

# Map raw feature name → P4 metadata field name (without meta. prefix)
FEATURE_TO_META = {
    "Protocol":        "protocol",
    "SrcPort":         "canon_src_port",
    "DstPort":         "canon_dst_port",
    "Duration":        "duration",
    "MaxIAT":          "max_iat",
    "UrgCount":        "urg_count",
    "FwdPktCount":     "fwd_pkt_count",
    "BwdPktCount":     "bwd_pkt_count",
    "FwdBytes":        "fwd_bytes",
    "BwdBytes":        "bwd_bytes",
    "MaxWinSize":      "max_win_size",
    "FlagsSyn":        "flags_syn",
    "FlagsAck":        "flags_ack",
    "FlagsFin":        "flags_fin",
    "FlagsRst":        "flags_rst",
    "MinIAT":          "min_iat",
    "FwdMaxPktLen":    "fwd_max_pkt_len",
    "BwdMaxPktLen":    "bwd_max_pkt_len",
    "FlagsPsh":        "flags_psh",
    "InitFwdWinBytes": "init_fwd_win",
}

# P4 bit widths for raw features
FEATURE_P4_TYPE = {
    "Protocol":        "bit<8>",
    "SrcPort":         "port_t",        # bit<16>
    "DstPort":         "port_t",        # bit<16>
    "Duration":        "duration_t",    # bit<48>
    "MaxIAT":          "iat_t",         # bit<48>
    "UrgCount":        "bit<32>",
    "FwdPktCount":     "bit<32>",
    "BwdPktCount":     "bit<32>",
    "FwdBytes":        "bytes_t",       # bit<32>
    "BwdBytes":        "bytes_t",       # bit<32>
    "MaxWinSize":      "bit<16>",
    "FlagsSyn":        "bit<32>",
    "FlagsAck":        "bit<32>",
    "FlagsFin":        "bit<32>",
    "FlagsRst":        "bit<32>",
    "MinIAT":          "iat_t",         # bit<48>
    "FwdMaxPktLen":    "bit<16>",
    "BwdMaxPktLen":    "bit<16>",
    "FlagsPsh":        "bit<32>",
    "InitFwdWinBytes": "bit<16>",
}

FEATURE_P4_WIDTH = {
    "Protocol":        8,
    "SrcPort":         16,
    "DstPort":         16,
    "Duration":        48,
    "MaxIAT":          48,
    "UrgCount":        32,
    "FwdPktCount":     32,
    "BwdPktCount":     32,
    "FwdBytes":        32,
    "BwdBytes":        32,
    "MaxWinSize":      16,
    "FlagsSyn":        32,
    "FlagsAck":        32,
    "FlagsFin":        32,
    "FlagsRst":        32,
    "MinIAT":          48,
    "FwdMaxPktLen":    16,
    "BwdMaxPktLen":    16,
    "FlagsPsh":        32,
    "InitFwdWinBytes": 16,
}


class P4CodeGenerator:
    def __init__(self, n_components=2, bits=16, output_file='basic.p4',
                 model_type='dt', rf_params=None, xgb_params=None, cnn_params=None,
                 n_registers=65536, flow_timeout_s=20,
                 reduction_config=None):
        self.n_components    = n_components
        self.bits            = bits
        self.output_file     = output_file
        self.model_type      = model_type
        self.rf_params       = rf_params  or {}
        self.xgb_params      = xgb_params or {}
        self.cnn_params      = cnn_params or {}
        self.n_registers     = n_registers
        self.flow_timeout_ns = int(flow_timeout_s * 1_000_000_000)
        self.reduction_config = reduction_config or {}

        # Derived from reduction_config
        method = self.reduction_config.get('method', 'pca')
        self.needs_transform = self.reduction_config.get('needs_transform_tables', True)

        # Code prefix: "pc" for PCA, "ld" for LDA, "ae" for Autoencoder, "um" for UMAP.
        if method == 'lda':
            self.code_prefix = 'ld'
        elif method == 'autoencoder':
            self.code_prefix = 'ae'
        elif method == 'umap':
            self.code_prefix = 'um'
        else:
            self.code_prefix = 'pc'

        # Transform table/action prefixes — each method gets its own distinct table name
        if method == 'lda':
            self.table_prefix = 'lda'
            self.action_prefix = 'ld'
        elif method == 'autoencoder':
            self.table_prefix = 'ae'
            self.action_prefix = 'ae'
        elif method == 'umap':
            self.table_prefix = 'umap'
            self.action_prefix = 'um'
        else:
            self.table_prefix = 'pca'
            self.action_prefix = 'pc'

        # Classifier feature names (what the ml_code / rf_tree / xgb_tree keys match on)
        self.classifier_features = self.reduction_config.get('feature_columns', None)
        if self.classifier_features is None:
            # Fallback: pc*_code
            self.classifier_features = [f'PC{i+1}_code' for i in range(n_components)]

        if self.model_type == 'cnn' and self.cnn_params.get('feature_names'):
            self.classifier_features = self.cnn_params['feature_names']

    def _meta_field_for_feature(self, feat_name):
        """Map a feature name to a P4 meta.* field reference."""
        # Raw feature (Duration, FwdBytes, etc.)
        if feat_name in FEATURE_TO_META:
            return f'meta.{FEATURE_TO_META[feat_name]}'
        # Transform code (PC1_code, LD2_code, AE3_code, etc.)
        return f'meta.{feat_name.lower()}'

    def _feature_bit_width(self, feat_name):
        if feat_name in FEATURE_P4_WIDTH:
            return FEATURE_P4_WIDTH[feat_name]
        return int(self.bits or 16)

    # ─────────────────────────────────────────────────────────────────────
    def generate_header(self):
        return '''/* -*- P4_16 -*- */
/*
 * P4 Flow-Based ML Classification
 * Auto-generated - supports PCA / LDA / Autoencoder / UMAP / Feature Selection + DT / RF / XGB / GB / CNN
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

#define BLOOM_FILTER_BIT_WIDTH 32
#define FLOW_TIMEOUT ''' + str(self.flow_timeout_ns) + '''  // ''' + str(self.flow_timeout_ns // 1_000_000_000) + '''s in nanoseconds

#define FIRST_INDEX ((bit<32>)0)
#define WRITE_REG(r, v) r.write(FIRST_INDEX, v)
#define READ_REG(r,  v) r.read(v, FIRST_INDEX)

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
    port_t    canon_src_port;
    port_t    canon_dst_port;
    bit<1>    is_reverse_dir;

    // Flow state tracking
    bit<32> flow_hash;
    bit<32> flow_hash_2;
    bit<1>  is_first_packet;
    bit<1>  hash_collision;
    bit<1>  flow_ended;
    
    // Flow-based features
    duration_t duration;
    iat_t      max_iat;
    bit<32>    urg_count;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    max_win_size;
    bit<32>    flags_syn;
    bit<32>    flags_ack;
    bit<32>    flags_fin;
    bit<32>    flags_rst;
    bytes_t    pkt_len;      // IP totalLen for IPv4; 28 for ARP (fixed); used for byte counting
    // New features
    iat_t      min_iat;
    bit<16>    fwd_max_pkt_len;
    bit<16>    bwd_max_pkt_len;
    bit<32>    flags_psh;
    bit<16>    init_fwd_win;
'''
        # Transformed feature codes (PCA, LDA, or Autoencoder)
        if self.needs_transform:
            code += f'\n    // {pfx.upper()} transformed features (quantized)\n'
            for i in range(1, self.n_components + 1):
                code += f'    pca_code_t {pfx}{i}_code;\n'

        code += '''
    // Classification result
    inference_result_t ml_result;

    // Timestamp
    bit<48> ingress_timestamp;
'''
        # RF packed votes
        if self.model_type == 'rf':
            n_est     = self.rf_params.get('n_estimators', 8)
            vote_bits = self.rf_params.get('vote_bits', 2)
            total_vb  = n_est * vote_bits
            code += f'\n    // RF packed vote field ({n_est} trees x {vote_bits} bits)\n'
            code += f'    bit<{total_vb}> rf_votes;\n'

        # XGB per-class accumulators
        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            code += f'\n    // XGB per-class score accumulators\n'
            for c in range(n_cls):
                code += f'    bit<16> xgb_score_c{c};\n'

        # CNN inputs, hidden activations, and class scores
        if self.model_type == 'cnn':
            input_bits = int(self.cnn_params.get('input_bits', 8))
            hidden_bits = int(self.cnn_params.get('hidden_bits', 8))
            hidden1_units = int(self.cnn_params.get('hidden1_units', 0))
            hidden2_units = int(self.cnn_params.get('hidden2_units', 0))
            pool = int(self.cnn_params.get('pool', 2))
            n_cls = len(self.cnn_params.get('classes', []))
            code += f'\n    // CNN quantized inputs\n'
            for i in range(len(self.classifier_features)):
                code += f'    bit<{input_bits}> cnn_in{i};\n'
            code += f'\n    // CNN hidden accumulators + activations\n'
            for h in range(hidden1_units):
                code += f'    bit<32>  cnn1_h{h}_sum;\n'
                code += f'    bit<{hidden_bits}> cnn1_h{h};\n'
            pooled = hidden1_units // max(1, pool)
            code += f'\n    // CNN pooled activations\n'
            for p in range(pooled):
                code += f'    bit<{hidden_bits}> cnn_p{p};\n'
            code += f'\n    // CNN hidden2 accumulators + activations\n'
            for h in range(hidden2_units):
                code += f'    bit<32>  cnn2_h{h}_sum;\n'
                code += f'    bit<{hidden_bits}> cnn2_h{h};\n'
            code += f'\n    // CNN class scores\n'
            for c in range(n_cls):
                code += f'    bit<32> cnn_score_c{c};\n'

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
    port_t srcPort;
    port_t dstPort;
    bit<8>  protocol;

    duration_t duration;
    iat_t      max_iat;
    bit<32>    urg_count;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    max_win_size;
    bit<32>    flags_syn;
    bit<32>    flags_ack;
    bit<32>    flags_fin;
    bit<32>    flags_rst;
    iat_t      min_iat;
    bit<16>    fwd_max_pkt_len;
    bit<16>    bwd_max_pkt_len;
    bit<32>    flags_psh;
    bit<16>    init_fwd_win;
'''
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'    pca_code_t {pfx}{i}_code;\n'

        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                code += f'    bit<16> xgb_score_c{c};\n'

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
        if self.model_type == 'cnn':
            pool = int(self.cnn_params.get('pool', 2))
            if pool not in (1, 2):
                raise ValueError("CNN P4 generation currently supports only pool=1 or 2.")
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
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_urg_count;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_fwd_pkt_count;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_bwd_pkt_count;
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_fwd_bytes;
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_bwd_bytes;
    register<bit<16>>(MAX_REGISTER_ENTRIES) reg_max_win_size;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_syn;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_ack;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_fin;
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_rst;
    register<iat_t>(MAX_REGISTER_ENTRIES)   reg_min_iat;
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
    register<bit<8>>(MAX_REGISTER_ENTRIES)  reg_protocol;       // IP protocol

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
        bit<48> time_first;
        bit<48> time_last;
        iat_t   max_iat;
        iat_t   min_iat;
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
        bit<16> fwd_max_pkt_len;
        bit<16> bwd_max_pkt_len;
        bit<32> flags_psh;
        bit<16> init_fwd_win;

        reg_time_first_pkt.read(time_first, meta.flow_hash);
        reg_time_last_pkt.read(time_last, meta.flow_hash);
        reg_max_iat.read(max_iat, meta.flow_hash);
        reg_min_iat.read(min_iat, meta.flow_hash);
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
        reg_fwd_max_pkt_len.read(fwd_max_pkt_len, meta.flow_hash);
        reg_bwd_max_pkt_len.read(bwd_max_pkt_len, meta.flow_hash);
        reg_flags_psh.read(flags_psh, meta.flow_hash);
        reg_init_fwd_win.read(init_fwd_win, meta.flow_hash);

        // Timeout check — previous flow on this slot has been idle
        if (time_first != 0 && time_last != 0 &&
                (current_time - time_last) > FLOW_TIMEOUT) {
            if ((fwd_pkt_count + bwd_pkt_count) >= 2) {
                meta.flow_ended       = 1w1;
                meta.duration         = time_last - time_first;
                meta.max_iat          = max_iat;
                meta.min_iat          = min_iat;
                meta.urg_count        = urg_count;
                meta.fwd_pkt_count    = fwd_pkt_count;
                meta.bwd_pkt_count    = bwd_pkt_count;
                meta.fwd_bytes        = fwd_bytes;
                meta.bwd_bytes        = bwd_bytes;
                meta.max_win_size     = max_win_size;
                meta.flags_syn        = flags_syn;
                meta.flags_ack        = flags_ack;
                meta.flags_fin        = flags_fin;
                meta.flags_rst        = flags_rst;
                meta.fwd_max_pkt_len  = fwd_max_pkt_len;
                meta.bwd_max_pkt_len  = bwd_max_pkt_len;
                meta.flags_psh        = flags_psh;
                meta.init_fwd_win     = init_fwd_win;
            }
            // Reset ALL registers for the new flow (regardless of pkt count)
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            reg_time_last_pkt.write(meta.flow_hash, current_time);
            reg_max_iat.write(meta.flow_hash, 0);
            reg_min_iat.write(meta.flow_hash, 0);
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
            reg_protocol.write(meta.flow_hash, meta.protocol);
        }
    }

    action update_packet_stats() {
        bit<48> current_time_us = standard_metadata.ingress_global_timestamp;
        bit<48> current_time = current_time_us * 1000;
        bit<48> time_first;
        bit<48> time_last;
        iat_t   max_iat;
        iat_t   min_iat;
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
        bit<16> fwd_max_pkt_len;
        bit<16> bwd_max_pkt_len;
        bit<32> flags_psh;
        bit<16> init_fwd_win;

        reg_time_first_pkt.read(time_first, meta.flow_hash);
        reg_time_last_pkt.read(time_last, meta.flow_hash);
        reg_max_iat.read(max_iat, meta.flow_hash);
        reg_min_iat.read(min_iat, meta.flow_hash);
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
        reg_fwd_max_pkt_len.read(fwd_max_pkt_len, meta.flow_hash);
        reg_bwd_max_pkt_len.read(bwd_max_pkt_len, meta.flow_hash);
        reg_flags_psh.read(flags_psh, meta.flow_hash);
        reg_init_fwd_win.read(init_fwd_win, meta.flow_hash);

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
            reg_protocol.write(meta.flow_hash, meta.protocol);
        }

        // IAT update (MaxIAT and MinIAT)
        if (time_last != 0) {
            iat_t current_iat = current_time - time_last;
            if (current_iat > max_iat) {
                max_iat = current_iat;
                reg_max_iat.write(meta.flow_hash, max_iat);
            }
            if (min_iat == 0 || current_iat < min_iat) {
                min_iat = current_iat;
                reg_min_iat.write(meta.flow_hash, min_iat);
            }
        }
        if (meta.flow_ended == 1w0) {
            meta.max_iat = max_iat;
            meta.min_iat = min_iat;
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

        // TCP flag counts (URG, SYN, ACK, FIN, RST, PSH)
        if (meta.protocol == TYPE_TCP) {
            if (hdr.tcp.ctrl[5:5] == 1w1) {
                urg_count = urg_count + 1;
                reg_urg_count.write(meta.flow_hash, urg_count);
            }
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
            meta.urg_count  = urg_count;
            meta.flags_syn  = flags_syn;
            meta.flags_ack  = flags_ack;
            meta.flags_fin  = flags_fin;
            meta.flags_rst  = flags_rst;
            meta.flags_psh  = flags_psh;
        }

        // FIN/RST ends the flow
        if (meta.flow_ended == 1w0 &&
                meta.protocol == TYPE_TCP &&
                (meta.flags_fin > 32w0 || meta.flags_rst > 32w0)) {
            meta.flow_ended   = 1w1;
            meta.duration     = current_time - time_first;
            meta.max_iat      = max_iat;
            meta.min_iat      = min_iat;
            meta.urg_count    = urg_count;
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
            reg_min_iat.write(meta.flow_hash, 0);
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
            reg_protocol.write(meta.flow_hash, 8w0);        // clear protocol bookmark
        }
    }

'''

        # ── Transform tables (PCA / LDA / Autoencoder / UMAP) ────────────
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'''
    // {pfx.upper()} component {i} transformation
    action set_{self.action_prefix}{i}_code(pca_code_t code) {{
        meta.{pfx}{i}_code = code;
    }}

    table {self.table_prefix}_component{i} {{
        key = {{
'''
                for feat in FLOW_FEATURES:
                    meta_f = FEATURE_TO_META[feat]
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
            n_est     = self.rf_params.get('n_estimators', 8)
            vote_bits = self.rf_params.get('vote_bits', 2)
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
            code += f'''
    table rf_vote_classify {{
        key = {{
            meta.rf_votes : exact;
        }}
        actions = {{
            set_result;
            NoAction;
        }}
        size = {2**total_vb};
    }}
'''

        elif self.model_type == 'xgb':
            total_trees = self.xgb_params.get('total_trees', 0)
            n_cls       = self.xgb_params.get('n_classes', 2)
            xgb_feats   = self.xgb_params.get('feature_names', self.classifier_features)

            for c in range(n_cls):
                code += f'''
    action add_xgb_score_c{c}(bit<8> delta) {{
        meta.xgb_score_c{c} = meta.xgb_score_c{c} + (bit<16>)delta;
    }}
'''
            for tidx in range(total_trees):
                cidx = tidx % n_cls
                code += f'''
    table xgb_tree_{tidx} {{
        key = {{
'''
                for feat_name in xgb_feats:
                    meta_f = self._meta_field_for_feature(feat_name)
                    code += f'            {meta_f:30s}: range;\n'
                code += f'''        }}
        actions = {{
            add_xgb_score_c{cidx};
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

            code += '''
    table xgb_classify {
        key = {
'''
            for c in range(n_cls):
                code += f'            meta.xgb_score_c{c:30s}: range;\n' if False else \
                        f'            meta.xgb_score_c{c} : range;\n'
            code += '''        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }
'''

        elif self.model_type == 'cnn':
            input_bits = int(self.cnn_params.get('input_bits', 8))
            hidden_bits = int(self.cnn_params.get('hidden_bits', 8))
            hidden1_units = int(self.cnn_params.get('hidden1_units', 0))
            hidden2_units = int(self.cnn_params.get('hidden2_units', 0))
            pool = int(self.cnn_params.get('pool', 2))
            n_cls = len(self.cnn_params.get('classes', []))
            use_quanti = bool(self.cnn_params.get('use_quanti', False))

            for h in range(hidden1_units):
                code += f'''
    action cnn1_add_h{h}(bit<32> delta) {{
        meta.cnn1_h{h}_sum = meta.cnn1_h{h}_sum + delta;
    }}
'''
            if use_quanti:
                for h in range(hidden1_units):
                    code += f'''
    action set_cnn1_h{h}(bit<{hidden_bits}> val) {{
        meta.cnn1_h{h} = val;
    }}
'''
            for h in range(hidden2_units):
                code += f'''
    action cnn2_add_h{h}(bit<32> delta) {{
        meta.cnn2_h{h}_sum = meta.cnn2_h{h}_sum + delta;
    }}
'''
            if use_quanti:
                for h in range(hidden2_units):
                    code += f'''
    action set_cnn2_h{h}(bit<{hidden_bits}> val) {{
        meta.cnn2_h{h} = val;
    }}
'''
            for c in range(n_cls):
                code += f'''
    action cnn_out_add_c{c}(bit<32> delta) {{
        meta.cnn_score_c{c} = meta.cnn_score_c{c} + delta;
    }}
'''
            for h in range(hidden1_units):
                for fi in range(len(self.classifier_features)):
                    code += f'''
    table cnn1_h{h}_f{fi} {{
        key = {{
            meta.cnn_in{fi} : exact;
        }}
        actions = {{
            cnn1_add_h{h};
            NoAction;
        }}
        size = {2**input_bits};
    }}
'''
            if use_quanti:
                for h in range(hidden1_units):
                    code += f'''
    table cnn1_quant_h{h} {{
        key = {{
            meta.cnn1_h{h}_sum : range;
        }}
        actions = {{
            set_cnn1_h{h};
            NoAction;
        }}
        size = 65535;
    }}
'''
            pooled = hidden1_units // max(1, pool)
            for h in range(hidden2_units):
                for pi in range(pooled):
                    code += f'''
    table cnn2_h{h}_p{pi} {{
        key = {{
            meta.cnn_p{pi} : exact;
        }}
        actions = {{
            cnn2_add_h{h};
            NoAction;
        }}
        size = {2**hidden_bits};
    }}
'''
            if use_quanti:
                for h in range(hidden2_units):
                    code += f'''
    table cnn2_quant_h{h} {{
        key = {{
            meta.cnn2_h{h}_sum : range;
        }}
        actions = {{
            set_cnn2_h{h};
            NoAction;
        }}
        size = 65535;
    }}
'''
            for c in range(n_cls):
                for h in range(hidden2_units):
                    code += f'''
    table cnn_out_c{c}_h{h} {{
        key = {{
            meta.cnn2_h{h} : exact;
        }}
        actions = {{
            cnn_out_add_c{c};
            NoAction;
        }}
        size = {2**hidden_bits};
    }}
'''

        # ── Build classify+digest snippet (used for BOTH timeout and FIN/RST paths) ──
        classify_snippet = ''

        # Transform tables (PCA/LDA/Autoencoder/UMAP only)
        if self.needs_transform:
            classify_snippet += f'\n                // Apply {pfx.upper()} transformations\n'
            for i in range(1, self.n_components + 1):
                classify_snippet += f'                {self.table_prefix}_component{i}.apply();\n'

        if self.model_type == 'cnn':
            input_bits = int(self.cnn_params.get('input_bits', 8))
            classify_snippet += '\n                // Quantize CNN inputs\n'
            for i, feat_name in enumerate(self.classifier_features):
                meta_f = self._meta_field_for_feature(feat_name)
                width = self._feature_bit_width(feat_name)
                shift = max(0, width - input_bits)
                if shift > 0:
                    classify_snippet += f'                meta.cnn_in{i} = (bit<{input_bits}>)({meta_f} >> {shift});\n'
                else:
                    classify_snippet += f'                meta.cnn_in{i} = (bit<{input_bits}>){meta_f};\n'

        # Classifier
        classify_snippet += '\n                // Apply classifier\n'
        if self.model_type == 'dt':
            classify_snippet += '                ml_code.apply();\n'
        elif self.model_type == 'rf':
            n_est = self.rf_params.get('n_estimators', 8)
            total_vb = n_est * self.rf_params.get('vote_bits', 2)
            classify_snippet += f'                meta.rf_votes = {total_vb}w0;\n'
            for i in range(n_est):
                classify_snippet += f'                rf_tree_{i}.apply();\n'
            classify_snippet += '                rf_vote_classify.apply();\n'
        elif self.model_type == 'xgb':
            total_trees = self.xgb_params.get('total_trees', 0)
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                classify_snippet += f'                meta.xgb_score_c{c} = 16w0;\n'
            for tidx in range(total_trees):
                classify_snippet += f'                xgb_tree_{tidx}.apply();\n'
            classify_snippet += '                xgb_classify.apply();\n'
        elif self.model_type == 'cnn':
            hidden1_units = int(self.cnn_params.get('hidden1_units', 0))
            hidden2_units = int(self.cnn_params.get('hidden2_units', 0))
            hidden_bits = int(self.cnn_params.get('hidden_bits', 8))
            pool = int(self.cnn_params.get('pool', 2))
            h1_shift = int(self.cnn_params.get('h1_shift', 0))
            h2_shift = int(self.cnn_params.get('h2_shift', 0))
            use_quanti = bool(self.cnn_params.get('use_quanti', False))
            h_max = (1 << hidden_bits) - 1
            n_cls = len(self.cnn_params.get('classes', []))
            b1_int = self.cnn_params.get('b1_int', [0] * hidden1_units)
            b2_int = self.cnn_params.get('b2_int', [0] * hidden2_units)
            b3_int = self.cnn_params.get('b3_int', [0] * n_cls)

            classify_snippet += '                // CNN hidden accumulators init\n'
            for h in range(hidden1_units):
                bias = int(b1_int[h]) if h < len(b1_int) else 0
                bias_u = (bias + (1 << 32)) % (1 << 32)
                classify_snippet += f'                meta.cnn1_h{h}_sum = 32w{bias_u};\n'

            classify_snippet += '                // CNN class scores init\n'
            for c in range(n_cls):
                bias = int(b3_int[c]) if c < len(b3_int) else 0
                bias_u = (bias + (1 << 32)) % (1 << 32)
                classify_snippet += f'                meta.cnn_score_c{c} = 32w{bias_u};\n'

            classify_snippet += '                // CNN hidden1 layer lookups\n'
            for h in range(hidden1_units):
                for fi in range(len(self.classifier_features)):
                    classify_snippet += f'                cnn1_h{h}_f{fi}.apply();\n'

            if use_quanti:
                classify_snippet += '                // CNN quantize hidden1 (table)\n'
                for h in range(hidden1_units):
                    classify_snippet += (f'                if (meta.cnn1_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn1_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else {{ cnn1_quant_h{h}.apply(); }}\n')
            else:
                classify_snippet += '                // CNN ReLU + quantize hidden1\n'
                for h in range(hidden1_units):
                    classify_snippet += (f'                if (meta.cnn1_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn1_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else if ((meta.cnn1_h{h}_sum >> {h1_shift}) > {h_max}) '
                                         f'{{ meta.cnn1_h{h} = {hidden_bits}w{h_max}; }}\n')
                    classify_snippet += (f'                else {{ meta.cnn1_h{h} = '
                                         f'(bit<{hidden_bits}>)(meta.cnn1_h{h}_sum >> {h1_shift}); }}\n')

            pooled = hidden1_units // max(1, pool)
            classify_snippet += '                // CNN maxpool\n'
            for p in range(pooled):
                idx0 = p * pool
                if pool == 1:
                    classify_snippet += f'                meta.cnn_p{p} = meta.cnn1_h{idx0};\n'
                else:
                    idx1 = idx0 + 1
                    classify_snippet += (f'                if (meta.cnn1_h{idx0} >= meta.cnn1_h{idx1}) '
                                         f'{{ meta.cnn_p{p} = meta.cnn1_h{idx0}; }} '
                                         f'else {{ meta.cnn_p{p} = meta.cnn1_h{idx1}; }}\n')

            classify_snippet += '                // CNN hidden2 accumulators init\n'
            for h in range(hidden2_units):
                bias = int(b2_int[h]) if h < len(b2_int) else 0
                bias_u = (bias + (1 << 32)) % (1 << 32)
                classify_snippet += f'                meta.cnn2_h{h}_sum = 32w{bias_u};\n'

            classify_snippet += '                // CNN hidden2 layer lookups\n'
            for h in range(hidden2_units):
                for p in range(pooled):
                    classify_snippet += f'                cnn2_h{h}_p{p}.apply();\n'

            if use_quanti:
                classify_snippet += '                // CNN quantize hidden2 (table)\n'
                for h in range(hidden2_units):
                    classify_snippet += (f'                if (meta.cnn2_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn2_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else {{ cnn2_quant_h{h}.apply(); }}\n')
            else:
                classify_snippet += '                // CNN ReLU + quantize hidden2\n'
                for h in range(hidden2_units):
                    classify_snippet += (f'                if (meta.cnn2_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn2_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else if ((meta.cnn2_h{h}_sum >> {h2_shift}) > {h_max}) '
                                         f'{{ meta.cnn2_h{h} = {hidden_bits}w{h_max}; }}\n')
                    classify_snippet += (f'                else {{ meta.cnn2_h{h} = '
                                         f'(bit<{hidden_bits}>)(meta.cnn2_h{h}_sum >> {h2_shift}); }}\n')

            classify_snippet += '                // CNN output layer lookups\n'
            for c in range(n_cls):
                for h in range(hidden2_units):
                    classify_snippet += f'                cnn_out_c{c}_h{h}.apply();\n'

            classify_snippet += '                // CNN argmax (signed compare)\n'
            if n_cls > 0:
                classify_snippet += '                meta.ml_result = 0;\n'
                classify_snippet += '                bit<32> best = meta.cnn_score_c0;\n'
                for c in range(1, n_cls):
                    classify_snippet += (f'                if ((best[31:31] == 1w1 && meta.cnn_score_c{c}[31:31] == 1w0) || '
                                         f'(best[31:31] == meta.cnn_score_c{c}[31:31] && meta.cnn_score_c{c} > best)) '
                                         f'{{ best = meta.cnn_score_c{c}; meta.ml_result = {c}; }}\n')

        # Digest fields
        classify_snippet += '''
                // Send digest
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
                    meta.min_iat,
                    meta.fwd_max_pkt_len,
                    meta.bwd_max_pkt_len,
                    meta.flags_psh,
                    meta.init_fwd_win,
'''
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                classify_snippet += f'                    meta.{pfx}{i}_code,\n'
        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                classify_snippet += f'                    meta.xgb_score_c{c},\n'
        classify_snippet += '                    meta.ml_result\n                });\n'

        # ── Apply block ──────────────────────────────────────────────────
        code += '''
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

            // Classify if flow ended (timeout from read_and_timeout_check, or
            // FIN/RST detected inside update_packet_stats).
            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 2) {
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

class TofinoP4CodeGenerator(P4CodeGenerator):
    """Generates TNA (Tofino Native Architecture) P4_16 code."""

    # ─────────────────────────────────────────────────────────────────────
    def generate_header(self):
        return '''/* -*- P4_16 -*- */
/*
 * P4 Flow-Based ML Classification — Tofino TNA target
 * Auto-generated - supports PCA / LDA / Autoencoder / UMAP / Feature Selection + DT / RF / XGB / GB / CNN
 */

#include <core.p4>
#include <tna.p4>

const bit<16> TYPE_IPV4       = 0x800;
const bit<16> TYPE_ARP        = 0x0806;
const bit<8>  TYPE_TCP        = 6;
const bit<8>  TYPE_UDP        = 17;
const bit<8>  TYPE_ICMP       = 1;
const bit<8>  TYPE_ARP_PSEUDO = 253;  // pseudo proto used in flow key for ARP

const bit<32> NB_ENTRIES = ''' + str(self.n_registers) + ''';
const bit<32> MAX_REGISTER_ENTRIES = ''' + str(self.n_registers) + ''';

#define BLOOM_FILTER_BIT_WIDTH 32
#define FLOW_TIMEOUT ''' + str(self.flow_timeout_ns) + '''  // ''' + str(self.flow_timeout_ns // 1_000_000_000) + '''s in nanoseconds

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

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
    port_t    canon_src_port;
    port_t    canon_dst_port;
    bit<1>    is_reverse_dir;

    // Flow state tracking
    bit<32> flow_hash;
    bit<32> flow_hash_2;
    bit<1>  is_first_packet;
    bit<1>  hash_collision;
    bit<1>  flow_ended;

    // Timestamps (read from registers into metadata for Tofino)
    bit<48> time_first;
    bit<48> time_last;

    // Tofino deparser-based digest trigger
    bit<1>  send_digest;

    // Flow-based features
    duration_t duration;
    iat_t      max_iat;
    bit<32>    urg_count;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    max_win_size;
    bit<32>    flags_syn;
    bit<32>    flags_ack;
    bit<32>    flags_fin;
    bit<32>    flags_rst;
    bytes_t    pkt_len;      // IP totalLen for IPv4; 28 for ARP (fixed); used for byte counting
    // New features
    iat_t      min_iat;
    bit<16>    fwd_max_pkt_len;
    bit<16>    bwd_max_pkt_len;
    bit<32>    flags_psh;
    bit<16>    init_fwd_win;
'''
        # Transformed feature codes (PCA, LDA, or Autoencoder)
        if self.needs_transform:
            code += f'\n    // {pfx.upper()} transformed features (quantized)\n'
            for i in range(1, self.n_components + 1):
                code += f'    pca_code_t {pfx}{i}_code;\n'

        code += '''
    // Classification result
    inference_result_t ml_result;
'''
        # RF packed votes
        if self.model_type == 'rf':
            n_est     = self.rf_params.get('n_estimators', 8)
            vote_bits = self.rf_params.get('vote_bits', 2)
            total_vb  = n_est * vote_bits
            code += f'\n    // RF packed vote field ({n_est} trees x {vote_bits} bits)\n'
            code += f'    bit<{total_vb}> rf_votes;\n'

        # XGB per-class accumulators
        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            code += f'\n    // XGB per-class score accumulators\n'
            for c in range(n_cls):
                code += f'    bit<16> xgb_score_c{c};\n'

        # CNN inputs, hidden activations, and class scores
        if self.model_type == 'cnn':
            input_bits = int(self.cnn_params.get('input_bits', 8))
            hidden_bits = int(self.cnn_params.get('hidden_bits', 8))
            hidden1_units = int(self.cnn_params.get('hidden1_units', 0))
            hidden2_units = int(self.cnn_params.get('hidden2_units', 0))
            pool = int(self.cnn_params.get('pool', 2))
            n_cls = len(self.cnn_params.get('classes', []))
            code += f'\n    // CNN quantized inputs\n'
            for i in range(len(self.classifier_features)):
                code += f'    bit<{input_bits}> cnn_in{i};\n'
            code += f'\n    // CNN hidden accumulators + activations\n'
            for h in range(hidden1_units):
                code += f'    bit<32>  cnn1_h{h}_sum;\n'
                code += f'    bit<{hidden_bits}> cnn1_h{h};\n'
            pooled = hidden1_units // max(1, pool)
            code += f'\n    // CNN pooled activations\n'
            for p in range(pooled):
                code += f'    bit<{hidden_bits}> cnn_p{p};\n'
            code += f'\n    // CNN hidden2 accumulators + activations\n'
            for h in range(hidden2_units):
                code += f'    bit<32>  cnn2_h{h}_sum;\n'
                code += f'    bit<{hidden_bits}> cnn2_h{h};\n'
            code += f'\n    // CNN class scores\n'
            for c in range(n_cls):
                code += f'    bit<32> cnn_score_c{c};\n'

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
    port_t srcPort;
    port_t dstPort;
    bit<8>  protocol;

    duration_t duration;
    iat_t      max_iat;
    bit<32>    urg_count;
    bit<32>    fwd_pkt_count;
    bit<32>    bwd_pkt_count;
    bytes_t    fwd_bytes;
    bytes_t    bwd_bytes;
    bit<16>    max_win_size;
    bit<32>    flags_syn;
    bit<32>    flags_ack;
    bit<32>    flags_fin;
    bit<32>    flags_rst;
    iat_t      min_iat;
    bit<16>    fwd_max_pkt_len;
    bit<16>    bwd_max_pkt_len;
    bit<32>    flags_psh;
    bit<16>    init_fwd_win;
'''
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'    pca_code_t {pfx}{i}_code;\n'

        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                code += f'    bit<16> xgb_score_c{c};\n'

        code += '''
    inference_result_t ml_result;
}
'''
        return code

    # ─────────────────────────────────────────────────────────────────────
    def generate_parser(self):
        return '''
/*************************************************************************
*********************** P A R S E R  ************************************
*************************************************************************/

parser SwitchIngressParser(
        packet_in pkt,
        out headers hdr,
        out metadata meta,
        out ingress_intrinsic_metadata_t ig_intr_md) {

    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            TYPE_ARP : parse_arp;
            default  : accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
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
        pkt.extract(hdr.tcp);
        meta.src_port = hdr.tcp.srcPort;
        meta.dst_port = hdr.tcp.dstPort;
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        meta.src_port = hdr.udp.srcPort;
        meta.dst_port = hdr.udp.dstPort;
        transition accept;
    }

    state parse_icmp {
        pkt.extract(hdr.icmp);
        meta.src_port = (port_t)hdr.icmp.icmp_type;
        meta.dst_port = (port_t)hdr.icmp.icmp_code;
        transition accept;
    }

    state parse_arp {
        pkt.extract(hdr.arp);
        meta.src_ip   = hdr.arp.spa;
        meta.dst_ip   = hdr.arp.tpa;
        meta.protocol = TYPE_ARP_PSEUDO;
        meta.src_port = hdr.arp.oper;
        meta.dst_port = 16w0;
        meta.pkt_len  = 32w28;  // ARP IPv4 payload is fixed 28 bytes
        transition accept;
    }
}

parser SwitchEgressParser(
        packet_in pkt,
        out headers hdr,
        out metadata meta,
        out egress_intrinsic_metadata_t eg_intr_md) {

    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}
'''

    # ─────────────────────────────────────────────────────────────────────
    def generate_ingress_forwarding(self):
        pfx = self.code_prefix
        if self.model_type == 'cnn':
            pool = int(self.cnn_params.get('pool', 2))
            if pool not in (1, 2):
                raise ValueError("CNN P4 generation currently supports only pool=1 or 2.")

        code = '''
/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control SwitchIngress(
        inout headers hdr,
        inout metadata meta,
        in    ingress_intrinsic_metadata_t              ig_intr_md,
        in    ingress_intrinsic_metadata_from_parser_t  ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t       ig_tm_md) {

    // ── Registers for flow state tracking ────────────────────────────
    Register<bit<48>, bit<32>>(MAX_REGISTER_ENTRIES) reg_time_first_pkt;
    Register<bit<48>, bit<32>>(MAX_REGISTER_ENTRIES) reg_time_last_pkt;
    Register<iat_t,   bit<32>>(MAX_REGISTER_ENTRIES) reg_max_iat;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_urg_count;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_fwd_pkt_count;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_bwd_pkt_count;
    Register<bytes_t, bit<32>>(MAX_REGISTER_ENTRIES) reg_fwd_bytes;
    Register<bytes_t, bit<32>>(MAX_REGISTER_ENTRIES) reg_bwd_bytes;
    Register<bit<16>, bit<32>>(MAX_REGISTER_ENTRIES) reg_max_win_size;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_syn;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_ack;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_fin;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_rst;
    Register<iat_t,   bit<32>>(MAX_REGISTER_ENTRIES) reg_min_iat;
    Register<bit<16>, bit<32>>(MAX_REGISTER_ENTRIES) reg_fwd_max_pkt_len;
    Register<bit<16>, bit<32>>(MAX_REGISTER_ENTRIES) reg_bwd_max_pkt_len;
    Register<bit<32>, bit<32>>(MAX_REGISTER_ENTRIES) reg_flags_psh;
    Register<bit<16>, bit<32>>(MAX_REGISTER_ENTRIES) reg_init_fwd_win;
    Register<bit<1>,  bit<32>>(MAX_REGISTER_ENTRIES) bloom_filter;

    // ── RegisterActions: read time_first ─────────────────────────────
    RegisterAction<bit<48>, bit<32>, bit<48>>(reg_time_first_pkt) ra_read_time_first = {
        void apply(inout bit<48> val, out bit<48> rv) { rv = val; }
    };
    RegisterAction<bit<48>, bit<32>, void>(reg_time_first_pkt) ra_init_time_first = {
        void apply(inout bit<48> val) { val = ig_intr_md.ingress_mac_tstamp; }
    };
    RegisterAction<bit<48>, bit<32>, void>(reg_time_first_pkt) ra_clear_time_first = {
        void apply(inout bit<48> val) { val = 48w0; }
    };

    // ── RegisterActions: time_last ────────────────────────────────────
    RegisterAction<bit<48>, bit<32>, void>(reg_time_last_pkt) ra_update_time_last = {
        void apply(inout bit<48> val) { val = ig_intr_md.ingress_mac_tstamp; }
    };
    RegisterAction<bit<48>, bit<32>, bit<48>>(reg_time_last_pkt) ra_read_time_last = {
        void apply(inout bit<48> val, out bit<48> rv) { rv = val; }
    };
    RegisterAction<bit<48>, bit<32>, void>(reg_time_last_pkt) ra_clear_time_last = {
        void apply(inout bit<48> val) { val = 48w0; }
    };

    // ── RegisterActions: max_iat ──────────────────────────────────────
    RegisterAction<iat_t, bit<32>, iat_t>(reg_max_iat) ra_update_max_iat = {
        void apply(inout iat_t val, out iat_t rv) {
            // TODO: Tofino SALU - compare with (ig_intr_md.ingress_mac_tstamp - meta.time_last)
            rv = val;
        }
    };
    RegisterAction<iat_t, bit<32>, void>(reg_max_iat) ra_clear_max_iat = {
        void apply(inout iat_t val) { val = 48w0; }
    };

    // ── RegisterActions: fwd_pkt_count ────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_fwd_pkt_count) ra_incr_fwd_pkt = {
        void apply(inout bit<32> val, out bit<32> rv) { val = val + 1; rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_fwd_pkt_count) ra_read_fwd_pkt = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_fwd_pkt_count) ra_clear_fwd_pkt = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: bwd_pkt_count ────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_bwd_pkt_count) ra_incr_bwd_pkt = {
        void apply(inout bit<32> val, out bit<32> rv) { val = val + 1; rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_bwd_pkt_count) ra_read_bwd_pkt = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_bwd_pkt_count) ra_clear_bwd_pkt = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: fwd_bytes ────────────────────────────────────
    RegisterAction<bytes_t, bit<32>, bytes_t>(reg_fwd_bytes) ra_add_fwd_bytes = {
        void apply(inout bytes_t val, out bytes_t rv) {
            val = val + meta.pkt_len; rv = val;
        }
    };
    RegisterAction<bytes_t, bit<32>, bytes_t>(reg_fwd_bytes) ra_read_fwd_bytes = {
        void apply(inout bytes_t val, out bytes_t rv) { rv = val; }
    };
    RegisterAction<bytes_t, bit<32>, void>(reg_fwd_bytes) ra_clear_fwd_bytes = {
        void apply(inout bytes_t val) { val = 32w0; }
    };

    // ── RegisterActions: bwd_bytes ────────────────────────────────────
    RegisterAction<bytes_t, bit<32>, bytes_t>(reg_bwd_bytes) ra_add_bwd_bytes = {
        void apply(inout bytes_t val, out bytes_t rv) {
            val = val + meta.pkt_len; rv = val;
        }
    };
    RegisterAction<bytes_t, bit<32>, bytes_t>(reg_bwd_bytes) ra_read_bwd_bytes = {
        void apply(inout bytes_t val, out bytes_t rv) { rv = val; }
    };
    RegisterAction<bytes_t, bit<32>, void>(reg_bwd_bytes) ra_clear_bwd_bytes = {
        void apply(inout bytes_t val) { val = 32w0; }
    };

    // ── RegisterActions: max_win_size ────────────────────────────────
    RegisterAction<bit<16>, bit<32>, bit<16>>(reg_max_win_size) ra_update_max_win_size = {
        void apply(inout bit<16> val, out bit<16> rv) {
            if (hdr.tcp.window > val) { val = hdr.tcp.window; }
            rv = val;
        }
    };
    RegisterAction<bit<16>, bit<32>, void>(reg_max_win_size) ra_clear_max_win_size = {
        void apply(inout bit<16> val) { val = 16w0; }
    };

    // ── RegisterActions: urg_count ────────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_urg_count) ra_incr_urg_count = {
        void apply(inout bit<32> val, out bit<32> rv) {
            if (hdr.tcp.ctrl[5:5] == 1w1) { val = val + 1; }
            rv = val;
        }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_urg_count) ra_read_urg_count = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_urg_count) ra_clear_urg_count = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: flags_syn ────────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_syn) ra_incr_flags_syn = {
        void apply(inout bit<32> val, out bit<32> rv) {
            val = val + (bit<32>)hdr.tcp.ctrl[1:1]; rv = val;
        }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_syn) ra_read_flags_syn = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_flags_syn) ra_clear_flags_syn = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: flags_ack ────────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_ack) ra_incr_flags_ack = {
        void apply(inout bit<32> val, out bit<32> rv) {
            val = val + (bit<32>)hdr.tcp.ctrl[4:4]; rv = val;
        }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_ack) ra_read_flags_ack = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_flags_ack) ra_clear_flags_ack = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: flags_fin ────────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_fin) ra_incr_flags_fin = {
        void apply(inout bit<32> val, out bit<32> rv) {
            val = val + (bit<32>)hdr.tcp.ctrl[0:0]; rv = val;
        }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_fin) ra_read_flags_fin = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_flags_fin) ra_clear_flags_fin = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: flags_rst ────────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_rst) ra_incr_flags_rst = {
        void apply(inout bit<32> val, out bit<32> rv) {
            val = val + (bit<32>)hdr.tcp.ctrl[2:2]; rv = val;
        }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_rst) ra_read_flags_rst = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_flags_rst) ra_clear_flags_rst = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: min_iat ──────────────────────────────────────
    RegisterAction<iat_t, bit<32>, iat_t>(reg_min_iat) ra_update_min_iat = {
        void apply(inout iat_t val, out iat_t rv) {
            // TODO: Tofino SALU — update when iat < val (or val == 0)
            rv = val;
        }
    };
    RegisterAction<iat_t, bit<32>, void>(reg_min_iat) ra_clear_min_iat = {
        void apply(inout iat_t val) { val = 48w0; }
    };

    // ── RegisterActions: fwd_max_pkt_len ─────────────────────────────
    RegisterAction<bit<16>, bit<32>, bit<16>>(reg_fwd_max_pkt_len) ra_update_fwd_max_pkt = {
        void apply(inout bit<16> val, out bit<16> rv) {
            if ((bit<16>)meta.pkt_len > val) { val = (bit<16>)meta.pkt_len; }
            rv = val;
        }
    };
    RegisterAction<bit<16>, bit<32>, void>(reg_fwd_max_pkt_len) ra_clear_fwd_max_pkt = {
        void apply(inout bit<16> val) { val = 16w0; }
    };

    // ── RegisterActions: bwd_max_pkt_len ─────────────────────────────
    RegisterAction<bit<16>, bit<32>, bit<16>>(reg_bwd_max_pkt_len) ra_update_bwd_max_pkt = {
        void apply(inout bit<16> val, out bit<16> rv) {
            if ((bit<16>)meta.pkt_len > val) { val = (bit<16>)meta.pkt_len; }
            rv = val;
        }
    };
    RegisterAction<bit<16>, bit<32>, void>(reg_bwd_max_pkt_len) ra_clear_bwd_max_pkt = {
        void apply(inout bit<16> val) { val = 16w0; }
    };

    // ── RegisterActions: flags_psh ────────────────────────────────────
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_psh) ra_incr_flags_psh = {
        void apply(inout bit<32> val, out bit<32> rv) {
            val = val + (bit<32>)hdr.tcp.ctrl[3:3]; rv = val;
        }
    };
    RegisterAction<bit<32>, bit<32>, bit<32>>(reg_flags_psh) ra_read_flags_psh = {
        void apply(inout bit<32> val, out bit<32> rv) { rv = val; }
    };
    RegisterAction<bit<32>, bit<32>, void>(reg_flags_psh) ra_clear_flags_psh = {
        void apply(inout bit<32> val) { val = 32w0; }
    };

    // ── RegisterActions: init_fwd_win ─────────────────────────────────
    RegisterAction<bit<16>, bit<32>, bit<16>>(reg_init_fwd_win) ra_read_init_fwd_win = {
        void apply(inout bit<16> val, out bit<16> rv) { rv = val; }
    };
    // Called only in the forward direction — writes once when val == 0
    RegisterAction<bit<16>, bit<32>, bit<16>>(reg_init_fwd_win) ra_set_init_fwd_win = {
        void apply(inout bit<16> val, out bit<16> rv) {
            if (val == 16w0) { val = hdr.tcp.window; }
            rv = val;
        }
    };
    RegisterAction<bit<16>, bit<32>, void>(reg_init_fwd_win) ra_clear_init_fwd_win = {
        void apply(inout bit<16> val) { val = 16w0; }
    };

    // ── RegisterAction: bloom_filter ─────────────────────────────────
    RegisterAction<bit<1>, bit<32>, void>(bloom_filter) ra_bloom_set = {
        void apply(inout bit<1> val) { val = 1w1; }
    };

    // ── Hash externs ─────────────────────────────────────────────────
    Hash<bit<32>>(HashAlgorithm_t.CRC16) hash_crc16;
    Hash<bit<32>>(HashAlgorithm_t.CRC32) hash_crc32;

    // ── Basic forwarding actions ──────────────────────────────────────
    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

    action ipv4_forward(macAddr_t dstAddr, PortId_t port) {
        ig_tm_md.ucast_egress_port = port;
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

    // ── Flow hash computation ─────────────────────────────────────────
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
        meta.flow_hash = hash_crc16.get({
            meta.canon_src_ip, meta.canon_dst_ip,
            meta.canon_src_port, meta.canon_dst_port, meta.protocol});
        meta.flow_hash_2 = hash_crc32.get({
            meta.canon_src_ip, meta.canon_dst_ip,
            meta.canon_src_port, meta.canon_dst_port, meta.protocol});
        ra_bloom_set.execute(meta.flow_hash);
    }

    // ── Phase 1: read current flow state ─────────────────────────────
    action read_flow_state_a() {
        meta.time_first       = ra_read_time_first.execute(meta.flow_hash);
        meta.time_last        = ra_read_time_last.execute(meta.flow_hash);
        meta.max_iat          = ra_update_max_iat.execute(meta.flow_hash);
        meta.min_iat          = ra_update_min_iat.execute(meta.flow_hash);
        meta.fwd_pkt_count    = ra_read_fwd_pkt.execute(meta.flow_hash);
        meta.bwd_pkt_count    = ra_read_bwd_pkt.execute(meta.flow_hash);
        meta.fwd_bytes        = ra_read_fwd_bytes.execute(meta.flow_hash);
        meta.bwd_bytes        = ra_read_bwd_bytes.execute(meta.flow_hash);
        meta.max_win_size     = ra_update_max_win_size.execute(meta.flow_hash);
        meta.urg_count        = ra_read_urg_count.execute(meta.flow_hash);
        meta.flags_syn        = ra_read_flags_syn.execute(meta.flow_hash);
        meta.flags_ack        = ra_read_flags_ack.execute(meta.flow_hash);
        meta.flags_fin        = ra_read_flags_fin.execute(meta.flow_hash);
        meta.flags_rst        = ra_read_flags_rst.execute(meta.flow_hash);
        meta.fwd_max_pkt_len  = ra_update_fwd_max_pkt.execute(meta.flow_hash);
        meta.bwd_max_pkt_len  = ra_update_bwd_max_pkt.execute(meta.flow_hash);
        meta.flags_psh        = ra_read_flags_psh.execute(meta.flow_hash);
        meta.init_fwd_win     = ra_read_init_fwd_win.execute(meta.flow_hash);
    }
    @hidden table read_flow_state_t {
        actions = { read_flow_state_a; }
        const default_action = read_flow_state_a();
    }

    // ── Phase 2a: init new flow (time_first = now) ────────────────────
    action init_flow_a() {
        ra_init_time_first.execute(meta.flow_hash);
    }
    @hidden table init_flow_t {
        actions = { init_flow_a; }
        const default_action = init_flow_a();
    }

    // ── Phase 2b: reset flow after timeout (clear all, set time_first=now) ──
    action reset_flow_regs_a() {
        ra_init_time_first.execute(meta.flow_hash);
        ra_clear_time_last.execute(meta.flow_hash);
        ra_clear_max_iat.execute(meta.flow_hash);
        ra_clear_min_iat.execute(meta.flow_hash);
        ra_clear_fwd_pkt.execute(meta.flow_hash);
        ra_clear_bwd_pkt.execute(meta.flow_hash);
        ra_clear_fwd_bytes.execute(meta.flow_hash);
        ra_clear_bwd_bytes.execute(meta.flow_hash);
        ra_clear_max_win_size.execute(meta.flow_hash);
        ra_clear_urg_count.execute(meta.flow_hash);
        ra_clear_flags_syn.execute(meta.flow_hash);
        ra_clear_flags_ack.execute(meta.flow_hash);
        ra_clear_flags_fin.execute(meta.flow_hash);
        ra_clear_flags_rst.execute(meta.flow_hash);
        ra_clear_fwd_max_pkt.execute(meta.flow_hash);
        ra_clear_bwd_max_pkt.execute(meta.flow_hash);
        ra_clear_flags_psh.execute(meta.flow_hash);
        ra_clear_init_fwd_win.execute(meta.flow_hash);
    }
    @hidden table reset_flow_regs_t {
        actions = { reset_flow_regs_a; }
        const default_action = reset_flow_regs_a();
    }

    // ── Phase 2c: clear flow (FIN/RST) ───────────────────────────────
    action clear_flow_regs_a() {
        ra_clear_time_first.execute(meta.flow_hash);
        ra_clear_time_last.execute(meta.flow_hash);
        ra_clear_max_iat.execute(meta.flow_hash);
        ra_clear_min_iat.execute(meta.flow_hash);
        ra_clear_fwd_pkt.execute(meta.flow_hash);
        ra_clear_bwd_pkt.execute(meta.flow_hash);
        ra_clear_fwd_bytes.execute(meta.flow_hash);
        ra_clear_bwd_bytes.execute(meta.flow_hash);
        ra_clear_max_win_size.execute(meta.flow_hash);
        ra_clear_urg_count.execute(meta.flow_hash);
        ra_clear_flags_syn.execute(meta.flow_hash);
        ra_clear_flags_ack.execute(meta.flow_hash);
        ra_clear_flags_fin.execute(meta.flow_hash);
        ra_clear_flags_rst.execute(meta.flow_hash);
        ra_clear_fwd_max_pkt.execute(meta.flow_hash);
        ra_clear_bwd_max_pkt.execute(meta.flow_hash);
        ra_clear_flags_psh.execute(meta.flow_hash);
        ra_clear_init_fwd_win.execute(meta.flow_hash);
    }
    @hidden table clear_flow_regs_t {
        actions = { clear_flow_regs_a; }
        const default_action = clear_flow_regs_a();
    }

    // ── Phase 3: update time_last ─────────────────────────────────────
    action update_time_last_a() {
        ra_update_time_last.execute(meta.flow_hash);
    }
    @hidden table update_time_last_t {
        actions = { update_time_last_a; }
        const default_action = update_time_last_a();
    }

    // ── Phase 4a: update forward direction counters ───────────────────
    action update_fwd_counters_a() {
        meta.fwd_pkt_count   = ra_incr_fwd_pkt.execute(meta.flow_hash);
        meta.fwd_bytes       = ra_add_fwd_bytes.execute(meta.flow_hash);
        meta.fwd_max_pkt_len = ra_update_fwd_max_pkt.execute(meta.flow_hash);
        // InitFwdWinBytes: capture window on first forward TCP packet only
        if (meta.protocol == TYPE_TCP) {
            meta.init_fwd_win = ra_set_init_fwd_win.execute(meta.flow_hash);
        }
    }
    @hidden table update_fwd_counters_t {
        actions = { update_fwd_counters_a; }
        const default_action = update_fwd_counters_a();
    }

    // ── Phase 4b: update backward direction counters ──────────────────
    action update_bwd_counters_a() {
        meta.bwd_pkt_count   = ra_incr_bwd_pkt.execute(meta.flow_hash);
        meta.bwd_bytes       = ra_add_bwd_bytes.execute(meta.flow_hash);
        meta.bwd_max_pkt_len = ra_update_bwd_max_pkt.execute(meta.flow_hash);
    }
    @hidden table update_bwd_counters_t {
        actions = { update_bwd_counters_a; }
        const default_action = update_bwd_counters_a();
    }

    // ── Phase 5: update TCP-specific features ────────────────────────
    action update_tcp_features_a() {
        meta.max_win_size = ra_update_max_win_size.execute(meta.flow_hash);
        meta.urg_count    = ra_incr_urg_count.execute(meta.flow_hash);
        meta.flags_syn    = ra_incr_flags_syn.execute(meta.flow_hash);
        meta.flags_ack    = ra_incr_flags_ack.execute(meta.flow_hash);
        meta.flags_fin    = ra_incr_flags_fin.execute(meta.flow_hash);
        meta.flags_rst    = ra_incr_flags_rst.execute(meta.flow_hash);
        meta.flags_psh    = ra_incr_flags_psh.execute(meta.flow_hash);
    }
    @hidden table update_tcp_features_t {
        actions = { update_tcp_features_a; }
        const default_action = update_tcp_features_a();
    }

'''

        # ── Transform tables (PCA / LDA / Autoencoder / UMAP) ────────────
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'''
    // {pfx.upper()} component {i} transformation
    action set_{self.action_prefix}{i}_code(pca_code_t code) {{
        meta.{pfx}{i}_code = code;
    }}

    table {self.table_prefix}_component{i} {{
        key = {{
'''
                for feat in FLOW_FEATURES:
                    meta_f = FEATURE_TO_META[feat]
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
            n_est     = self.rf_params.get('n_estimators', 8)
            vote_bits = self.rf_params.get('vote_bits', 2)
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
            code += f'''
    table rf_vote_classify {{
        key = {{
            meta.rf_votes : exact;
        }}
        actions = {{
            set_result;
            NoAction;
        }}
        size = {2**total_vb};
    }}
'''

        elif self.model_type == 'xgb':
            total_trees = self.xgb_params.get('total_trees', 0)
            n_cls       = self.xgb_params.get('n_classes', 2)
            xgb_feats   = self.xgb_params.get('feature_names', self.classifier_features)

            for c in range(n_cls):
                code += f'''
    action add_xgb_score_c{c}(bit<8> delta) {{
        meta.xgb_score_c{c} = meta.xgb_score_c{c} + (bit<16>)delta;
    }}
'''
            for tidx in range(total_trees):
                cidx = tidx % n_cls
                code += f'''
    table xgb_tree_{tidx} {{
        key = {{
'''
                for feat_name in xgb_feats:
                    meta_f = self._meta_field_for_feature(feat_name)
                    code += f'            {meta_f:30s}: range;\n'
                code += f'''        }}
        actions = {{
            add_xgb_score_c{cidx};
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''

            code += '''
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

        elif self.model_type == 'cnn':
            input_bits = int(self.cnn_params.get('input_bits', 8))
            hidden_bits = int(self.cnn_params.get('hidden_bits', 8))
            hidden1_units = int(self.cnn_params.get('hidden1_units', 0))
            hidden2_units = int(self.cnn_params.get('hidden2_units', 0))
            pool = int(self.cnn_params.get('pool', 2))
            n_cls = len(self.cnn_params.get('classes', []))
            use_quanti = bool(self.cnn_params.get('use_quanti', False))

            for h in range(hidden1_units):
                code += f'''
    action cnn1_add_h{h}(bit<32> delta) {{
        meta.cnn1_h{h}_sum = meta.cnn1_h{h}_sum + delta;
    }}
'''
            if use_quanti:
                for h in range(hidden1_units):
                    code += f'''
    action set_cnn1_h{h}(bit<{hidden_bits}> val) {{
        meta.cnn1_h{h} = val;
    }}
'''
            for h in range(hidden2_units):
                code += f'''
    action cnn2_add_h{h}(bit<32> delta) {{
        meta.cnn2_h{h}_sum = meta.cnn2_h{h}_sum + delta;
    }}
'''
            if use_quanti:
                for h in range(hidden2_units):
                    code += f'''
    action set_cnn2_h{h}(bit<{hidden_bits}> val) {{
        meta.cnn2_h{h} = val;
    }}
'''
            for c in range(n_cls):
                code += f'''
    action cnn_out_add_c{c}(bit<32> delta) {{
        meta.cnn_score_c{c} = meta.cnn_score_c{c} + delta;
    }}
'''
            for h in range(hidden1_units):
                for fi in range(len(self.classifier_features)):
                    code += f'''
    table cnn1_h{h}_f{fi} {{
        key = {{
            meta.cnn_in{fi} : exact;
        }}
        actions = {{
            cnn1_add_h{h};
            NoAction;
        }}
        size = {2**input_bits};
    }}
'''
            if use_quanti:
                for h in range(hidden1_units):
                    code += f'''
    table cnn1_quant_h{h} {{
        key = {{
            meta.cnn1_h{h}_sum : range;
        }}
        actions = {{
            set_cnn1_h{h};
            NoAction;
        }}
        size = 65535;
    }}
'''
            pooled = hidden1_units // max(1, pool)
            for h in range(hidden2_units):
                for pi in range(pooled):
                    code += f'''
    table cnn2_h{h}_p{pi} {{
        key = {{
            meta.cnn_p{pi} : exact;
        }}
        actions = {{
            cnn2_add_h{h};
            NoAction;
        }}
        size = {2**hidden_bits};
    }}
'''
            if use_quanti:
                for h in range(hidden2_units):
                    code += f'''
    table cnn2_quant_h{h} {{
        key = {{
            meta.cnn2_h{h}_sum : range;
        }}
        actions = {{
            set_cnn2_h{h};
            NoAction;
        }}
        size = 65535;
    }}
'''
            for c in range(n_cls):
                for h in range(hidden2_units):
                    code += f'''
    table cnn_out_c{c}_h{h} {{
        key = {{
            meta.cnn2_h{h} : exact;
        }}
        actions = {{
            cnn_out_add_c{c};
            NoAction;
        }}
        size = {2**hidden_bits};
    }}
'''

        # ── Build classify+digest snippet (used for both timeout and FIN/RST) ──
        classify_snippet = ''

        # Transform tables (PCA/LDA/Autoencoder/UMAP only)
        if self.needs_transform:
            classify_snippet += f'\n                // Apply {pfx.upper()} transformations\n'
            for i in range(1, self.n_components + 1):
                classify_snippet += f'                {self.table_prefix}_component{i}.apply();\n'

        if self.model_type == 'cnn':
            input_bits = int(self.cnn_params.get('input_bits', 8))
            classify_snippet += '\n                // Quantize CNN inputs\n'
            for i, feat_name in enumerate(self.classifier_features):
                meta_f = self._meta_field_for_feature(feat_name)
                width = self._feature_bit_width(feat_name)
                shift = max(0, width - input_bits)
                if shift > 0:
                    classify_snippet += f'                meta.cnn_in{i} = (bit<{input_bits}>)({meta_f} >> {shift});\n'
                else:
                    classify_snippet += f'                meta.cnn_in{i} = (bit<{input_bits}>){meta_f};\n'

        # Classifier
        classify_snippet += '\n                // Apply classifier\n'
        if self.model_type == 'dt':
            classify_snippet += '                ml_code.apply();\n'
        elif self.model_type == 'rf':
            n_est = self.rf_params.get('n_estimators', 8)
            total_vb = n_est * self.rf_params.get('vote_bits', 2)
            classify_snippet += f'                meta.rf_votes = {total_vb}w0;\n'
            for i in range(n_est):
                classify_snippet += f'                rf_tree_{i}.apply();\n'
            classify_snippet += '                rf_vote_classify.apply();\n'
        elif self.model_type == 'xgb':
            total_trees = self.xgb_params.get('total_trees', 0)
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                classify_snippet += f'                meta.xgb_score_c{c} = 16w0;\n'
            for tidx in range(total_trees):
                classify_snippet += f'                xgb_tree_{tidx}.apply();\n'
            classify_snippet += '                xgb_classify.apply();\n'
        elif self.model_type == 'cnn':
            hidden1_units = int(self.cnn_params.get('hidden1_units', 0))
            hidden2_units = int(self.cnn_params.get('hidden2_units', 0))
            hidden_bits = int(self.cnn_params.get('hidden_bits', 8))
            pool = int(self.cnn_params.get('pool', 2))
            h1_shift = int(self.cnn_params.get('h1_shift', 0))
            h2_shift = int(self.cnn_params.get('h2_shift', 0))
            use_quanti = bool(self.cnn_params.get('use_quanti', False))
            h_max = (1 << hidden_bits) - 1
            n_cls = len(self.cnn_params.get('classes', []))
            b1_int = self.cnn_params.get('b1_int', [0] * hidden1_units)
            b2_int = self.cnn_params.get('b2_int', [0] * hidden2_units)
            b3_int = self.cnn_params.get('b3_int', [0] * n_cls)

            classify_snippet += '                // CNN hidden accumulators init\n'
            for h in range(hidden1_units):
                bias = int(b1_int[h]) if h < len(b1_int) else 0
                bias_u = (bias + (1 << 32)) % (1 << 32)
                classify_snippet += f'                meta.cnn1_h{h}_sum = 32w{bias_u};\n'

            classify_snippet += '                // CNN class scores init\n'
            for c in range(n_cls):
                bias = int(b3_int[c]) if c < len(b3_int) else 0
                bias_u = (bias + (1 << 32)) % (1 << 32)
                classify_snippet += f'                meta.cnn_score_c{c} = 32w{bias_u};\n'

            classify_snippet += '                // CNN hidden1 layer lookups\n'
            for h in range(hidden1_units):
                for fi in range(len(self.classifier_features)):
                    classify_snippet += f'                cnn1_h{h}_f{fi}.apply();\n'

            if use_quanti:
                classify_snippet += '                // CNN quantize hidden1 (table)\n'
                for h in range(hidden1_units):
                    classify_snippet += (f'                if (meta.cnn1_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn1_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else {{ cnn1_quant_h{h}.apply(); }}\n')
            else:
                classify_snippet += '                // CNN ReLU + quantize hidden1\n'
                for h in range(hidden1_units):
                    classify_snippet += (f'                if (meta.cnn1_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn1_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else if ((meta.cnn1_h{h}_sum >> {h1_shift}) > {h_max}) '
                                         f'{{ meta.cnn1_h{h} = {hidden_bits}w{h_max}; }}\n')
                    classify_snippet += (f'                else {{ meta.cnn1_h{h} = '
                                         f'(bit<{hidden_bits}>)(meta.cnn1_h{h}_sum >> {h1_shift}); }}\n')

            pooled = hidden1_units // max(1, pool)
            classify_snippet += '                // CNN maxpool\n'
            for p in range(pooled):
                idx0 = p * pool
                if pool == 1:
                    classify_snippet += f'                meta.cnn_p{p} = meta.cnn1_h{idx0};\n'
                else:
                    idx1 = idx0 + 1
                    classify_snippet += (f'                if (meta.cnn1_h{idx0} >= meta.cnn1_h{idx1}) '
                                         f'{{ meta.cnn_p{p} = meta.cnn1_h{idx0}; }} '
                                         f'else {{ meta.cnn_p{p} = meta.cnn1_h{idx1}; }}\n')

            classify_snippet += '                // CNN hidden2 accumulators init\n'
            for h in range(hidden2_units):
                bias = int(b2_int[h]) if h < len(b2_int) else 0
                bias_u = (bias + (1 << 32)) % (1 << 32)
                classify_snippet += f'                meta.cnn2_h{h}_sum = 32w{bias_u};\n'

            classify_snippet += '                // CNN hidden2 layer lookups\n'
            for h in range(hidden2_units):
                for p in range(pooled):
                    classify_snippet += f'                cnn2_h{h}_p{p}.apply();\n'

            if use_quanti:
                classify_snippet += '                // CNN quantize hidden2 (table)\n'
                for h in range(hidden2_units):
                    classify_snippet += (f'                if (meta.cnn2_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn2_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else {{ cnn2_quant_h{h}.apply(); }}\n')
            else:
                classify_snippet += '                // CNN ReLU + quantize hidden2\n'
                for h in range(hidden2_units):
                    classify_snippet += (f'                if (meta.cnn2_h{h}_sum[31:31] == 1w1) '
                                         f'{{ meta.cnn2_h{h} = {hidden_bits}w0; }}\n')
                    classify_snippet += (f'                else if ((meta.cnn2_h{h}_sum >> {h2_shift}) > {h_max}) '
                                         f'{{ meta.cnn2_h{h} = {hidden_bits}w{h_max}; }}\n')
                    classify_snippet += (f'                else {{ meta.cnn2_h{h} = '
                                         f'(bit<{hidden_bits}>)(meta.cnn2_h{h}_sum >> {h2_shift}); }}\n')

            classify_snippet += '                // CNN output layer lookups\n'
            for c in range(n_cls):
                for h in range(hidden2_units):
                    classify_snippet += f'                cnn_out_c{c}_h{h}.apply();\n'

            classify_snippet += '                // CNN argmax (signed compare)\n'
            if n_cls > 0:
                classify_snippet += '                meta.ml_result = 0;\n'
                classify_snippet += '                bit<32> best = meta.cnn_score_c0;\n'
                for c in range(1, n_cls):
                    classify_snippet += (f'                if ((best[31:31] == 1w1 && meta.cnn_score_c{c}[31:31] == 1w0) || '
                                         f'(best[31:31] == meta.cnn_score_c{c}[31:31] && meta.cnn_score_c{c} > best)) '
                                         f'{{ best = meta.cnn_score_c{c}; meta.ml_result = {c}; }}\n')

        # Set digest flag (TNA uses deparser-based digest)
        classify_snippet += '                meta.send_digest = 1w1;\n'

        # ── Apply block (TNA) ────────────────────────────────────────────
        code += '''
    apply {
        if ((hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP ||
                                    meta.protocol == TYPE_ICMP)) ||
            hdr.arp.isValid()) {
            compute_flow_hash();
            // Step 1: read all register state into meta.* (old flow's values)
            read_flow_state_t.apply();

            if (meta.time_first != 0 && meta.time_last != 0 &&
                    (ig_intr_md.ingress_mac_tstamp - meta.time_last) > (bit<48>)FLOW_TIMEOUT) {
                meta.flow_ended = 1w1;
                meta.duration   = meta.time_last - meta.time_first;
                reset_flow_regs_t.apply();
                meta.is_first_packet = 1w1;
            } else if (meta.time_first == 0) {
                init_flow_t.apply();
                meta.is_first_packet = 1w1;
            }

            // Step 2: classify the timed-out flow NOW — meta.* still holds the
            // old flow's features before update_counters overwrites them.
            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 2) {
'''
        code += classify_snippet
        code += '''
            } // end timeout classification
            meta.flow_ended = 1w0;  // reset so FIN/RST can trigger for the new flow

            // Step 3: update registers for the current packet
            update_time_last_t.apply();

            if (meta.is_reverse_dir == 1w0) {
                update_fwd_counters_t.apply();
            } else {
                update_bwd_counters_t.apply();
            }

            if (meta.protocol == TYPE_TCP) {
                update_tcp_features_t.apply();
            }

            // Step 4: check for FIN/RST termination
            if (meta.flow_ended == 1w0 && meta.protocol == TYPE_TCP &&
                    (meta.flags_fin > 32w0 || meta.flags_rst > 32w0)) {
                meta.flow_ended = 1w1;
                meta.duration   = ig_intr_md.ingress_mac_tstamp - meta.time_first;
                clear_flow_regs_t.apply();
            }

            // Step 5: classify FIN/RST terminated flow
            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 2) {
'''
        code += classify_snippet
        code += '''
            } // end FIN/RST classification

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
        pfx = self.code_prefix
        code = '''
/*************************************************************************
***************  D E P A R S E R  &  E G R E S S  ***********************
*************************************************************************/

control SwitchIngressDeparser(
        packet_out pkt,
        inout headers hdr,
        in metadata meta,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {

    Digest<digest_t>() flow_digest;

    apply {
        if (meta.send_digest == 1w1) {
            flow_digest.pack({
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
        if self.needs_transform:
            for i in range(1, self.n_components + 1):
                code += f'                meta.{pfx}{i}_code,\n'

        if self.model_type == 'xgb':
            n_cls = self.xgb_params.get('n_classes', 2)
            for c in range(n_cls):
                code += f'                meta.xgb_score_c{c},\n'

        code += '''                meta.min_iat,
                meta.fwd_max_pkt_len,
                meta.bwd_max_pkt_len,
                meta.flags_psh,
                meta.init_fwd_win,
                meta.ml_result
            });
        }
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.arp);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.icmp);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
    }
}

control SwitchEgress(
        inout headers hdr,
        inout metadata meta,
        in    egress_intrinsic_metadata_t                 eg_intr_md,
        in    egress_intrinsic_metadata_from_parser_t     eg_prsr_md,
        inout egress_intrinsic_metadata_for_deparser_t    eg_dprsr_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_oport_md) {
    apply { }
}

control SwitchEgressDeparser(
        packet_out pkt,
        inout headers hdr,
        in metadata meta,
        in egress_intrinsic_metadata_for_deparser_t eg_dprsr_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.arp);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.icmp);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
    }
}

Pipeline(
    SwitchIngressParser(),
    SwitchIngress(),
    SwitchIngressDeparser(),
    SwitchEgressParser(),
    SwitchEgress(),
    SwitchEgressDeparser()
) pipe;

Switch(pipe) main;
'''
        return code


# ─── Parameter loading utilities ─────────────────────────────────────────

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


def load_rf_params(path='tables/rf_params.json'):
    if os.path.exists(path):
        try:
            with open(path) as f:
                p = json.load(f)
            logger.info(f"RF params: n_estimators={p.get('n_estimators')}, vote_bits={p.get('vote_bits')}")
            return p
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    return {"n_estimators": 8, "vote_bits": 2, "n_classes": 4}


def load_xgb_params(path='tables/xgb_params.json'):
    if os.path.exists(path):
        try:
            with open(path) as f:
                p = json.load(f)
            logger.info(f"XGB params: total_trees={p.get('total_trees')}, n_classes={p.get('n_classes')}")
            return p
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    return {"total_trees": 16, "n_classes": 2, "n_estimators": 8}


def load_cnn_params(path='tables/cnn_params.json'):
    if os.path.exists(path):
        try:
            with open(path) as f:
                p = json.load(f)
            logger.info(f"CNN params: hidden_units={p.get('hidden_units')}, "
                        f"input_bits={p.get('input_bits')}, hidden_bits={p.get('hidden_bits')}")
            return p
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
    return {}


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = P4secArgumentParser(
        description='Generate P4 code for ML classification (universal)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Notes:\n"
            "  - Reduction method is read from tables/reduction_config.json.\n"
            "  - GB uses the XGB P4 architecture (auto-mapped).\n"
            "  - Also emits a Tofino P4 file via --tofino-output.\n"
        )
    )
    parser.add_argument('--output', default='../basic.p4',
                        help='BMv2 P4 output (default: ../basic.p4)')
    parser.add_argument('--tofino-output', default='../p4sec_tofino.p4',
                        help='Tofino P4 output (default: ../p4sec_tofino.p4)')
    parser.add_argument('--params-file', default='tables/encoding_params.json')
    parser.add_argument('--commands-file', default='tables/s1-commands.txt')
    parser.add_argument('--reduction-config', default='tables/reduction_config.json')
    parser.add_argument('-m', '--model-type', default='dt',
                        choices=['dt', 'rf', 'xgb', 'gb', 'knn', 'svm', 'cnn'],
                        help='Classifier: dt | rf | xgb | gb (uses XGB arch) | knn | svm | cnn')
    parser.add_argument('--rf-params', default='tables/rf_params.json')
    parser.add_argument('--xgb-params', default='tables/xgb_params.json')
    parser.add_argument('--register-entries', type=int, default=65536)
    parser.add_argument('--flow-timeout-s', type=int, default=20)
    args = parser.parse_args()

    # Map GB → XGB (identical P4 architecture)
    user_model_type = args.model_type
    if args.model_type == 'gb':
        args.model_type = 'xgb'
    if args.model_type in ('knn', 'svm'):
        # Deploy as DT proxy in P4
        args.model_type = 'dt'
        logger.info("GB uses XGB P4 architecture — generating XGB tables.")

    # Load universal reduction config (may be None for old PCA pipeline)
    red_cfg = load_reduction_config(args.reduction_config)

    # Determine n_components and bits
    _method = (red_cfg or {}).get('method', 'pca')
    _prefix_map = {'lda': 'lda', 'autoencoder': 'ae', 'umap': 'umap'}
    _table_prefix = _prefix_map.get(_method, 'pca')
    if red_cfg and red_cfg.get('needs_transform_tables', True):
        n_components, bits = detect_n_components(args.params_file, args.commands_file, table_prefix=_table_prefix)
    elif red_cfg and not red_cfg.get('needs_transform_tables', True):
        n_components = red_cfg.get('n_components', 0)
        bits = 16
    else:
        n_components, bits = detect_n_components(args.params_file, args.commands_file, table_prefix=_table_prefix)

    # Load model-specific params
    rf_params  = load_rf_params(args.rf_params)   if args.model_type == 'rf'  else {}
    xgb_params = load_xgb_params(args.xgb_params) if args.model_type == 'xgb' else {}
    cnn_params = load_cnn_params('tables/cnn_params.json') if args.model_type == 'cnn' else {}

    generator = P4CodeGenerator(
        n_components=n_components,
        bits=bits,
        output_file=args.output,
        model_type=args.model_type,
        rf_params=rf_params,
        xgb_params=xgb_params,
        cnn_params=cnn_params,
        n_registers=args.register_entries,
        flow_timeout_s=args.flow_timeout_s,
        reduction_config=red_cfg or {},
    )
    generator.write_to_file()

    # Also emit a Tofino-target P4 file (TNA architecture)
    if args.tofino_output:
        tofino_gen = TofinoP4CodeGenerator(
            n_components=n_components,
            bits=bits,
            output_file=args.tofino_output,
            model_type=args.model_type,
            rf_params=rf_params,
            xgb_params=xgb_params,
            cnn_params=cnn_params,
            n_registers=args.register_entries,
            flow_timeout_s=args.flow_timeout_s,
            reduction_config=red_cfg or {},
        )
        tofino_gen.write_to_file()

    method_str = (red_cfg or {}).get('method', 'pca').upper()
    logger.info(f"\nGeneration complete!")
    logger.info(f"  Reduction : {method_str}")
    logger.info(f"  Model     : {user_model_type.upper()}"
                f"{' (P4 arch: ' + args.model_type.upper() + ')' if user_model_type != args.model_type else ''}")
    logger.info(f"  Transform : {'yes' if generator.needs_transform else 'no (direct features)'}")


if __name__ == '__main__':
    main()
