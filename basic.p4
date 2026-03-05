/* -*- P4_16 -*- */
/*
 * P4 Flow-Based ML Classification
 * Extracts flow-based features and applies PCA + Decision Tree classification
 */

#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x800;
const bit<8>  TYPE_TCP  = 6;
const bit<8>  TYPE_UDP  = 17;

const bit<32> NB_ENTRIES = 8192;
const bit<32> MAX_REGISTER_ENTRIES = 8192;

// Bloom filter for flow detection
#define BLOOM_FILTER_BIT_WIDTH 32
#define FLOW_TIMEOUT 120000000000  // 120 seconds in nanoseconds

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
typedef bit<1>  flags_t;        // TCP flags
typedef bit<16> pca_code_t;   // PCA component code (quantized)
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
    flags_t flags_syn;
    flags_t flags_ack;
    flags_t flags_fin;
    flags_t flags_rst;
    
    // PCA-transformed features (quantized)
    pca_code_t pc1_code;
    pca_code_t pc2_code;
    pca_code_t pc3_code;
    pca_code_t pc4_code;
    pca_code_t pc5_code;
    
    // Classification result
    inference_result_t ml_result;
    
    // Timestamp
    bit<48> ingress_timestamp;

    // RF packed vote field (8 trees x 2 bits)
    bit<16> rf_votes;
}

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
    flags_t flags_syn;
    flags_t flags_ack;
    flags_t flags_fin;
    flags_t flags_rst;
    
    // PCA component codes
    pca_code_t pc1_code;
    pca_code_t pc2_code;
    pca_code_t pc3_code;
    pca_code_t pc4_code;
    pca_code_t pc5_code;
    
    // Classification result
    inference_result_t ml_result;
}

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
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_syn;        // SYN flag seen
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_ack;        // ACK flag seen
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_fin;        // FIN flag seen
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_rst;        // RST flag seen
    
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

    // Helper to extract TCP flags
    action extract_tcp_flags() {
        if (meta.protocol == TYPE_TCP) {
            meta.flags_syn = hdr.tcp.ctrl[1:1];   // SYN bit
            meta.flags_ack = hdr.tcp.ctrl[4:4];   // ACK bit
            meta.flags_fin = hdr.tcp.ctrl[0:0];   // FIN bit
            meta.flags_rst = hdr.tcp.ctrl[2:2];   // RST bit
        }
    }

    // Helper to update flow state
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
            // Snapshot accumulated flags BEFORE clearing the registers
            reg_flags_syn.read(meta.flags_syn, meta.flow_hash);
            reg_flags_ack.read(meta.flags_ack, meta.flow_hash);
            reg_flags_fin.read(meta.flags_fin, meta.flow_hash);
            reg_flags_rst.read(meta.flags_rst, meta.flow_hash);

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
            reg_flags_syn.write(meta.flow_hash, 1w0);
            reg_flags_ack.write(meta.flow_hash, 1w0);
            reg_flags_fin.write(meta.flow_hash, 1w0);
            reg_flags_rst.write(meta.flow_hash, 1w0);
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

        // TCP flags (bitwise OR to aggregate)
        if (meta.flags_syn == 1) {
            reg_flags_syn.write(meta.flow_hash, 1w1);
        }
        if (meta.flags_ack == 1) {
            reg_flags_ack.write(meta.flow_hash, 1w1);
        }
        if (meta.flags_fin == 1) {
            reg_flags_fin.write(meta.flow_hash, 1w1);
        }
        if (meta.flags_rst == 1) {
            reg_flags_rst.write(meta.flow_hash, 1w1);
        }

        // FIN or RST ends the flow — snapshot current accumulated state and mark ended.
        // (Only when flow_ended not already set by timeout branch above)
        if (meta.flow_ended == 1w0 &&
                meta.protocol == TYPE_TCP &&
                (meta.flags_fin == 1w1 || meta.flags_rst == 1w1)) {
            meta.flow_ended    = 1w1;
            meta.duration      = current_time - time_first;
            meta.max_iat       = max_iat;
            meta.urg_count     = urg_count;
            reg_fwd_pkt_count.read(meta.fwd_pkt_count, meta.flow_hash);
            reg_bwd_pkt_count.read(meta.bwd_pkt_count, meta.flow_hash);
            reg_fwd_bytes.read(meta.fwd_bytes, meta.flow_hash);
            reg_bwd_bytes.read(meta.bwd_bytes, meta.flow_hash);
            reg_max_win_size.read(meta.max_win_size, meta.flow_hash);
            // Read accumulated syn/ack flags BEFORE clearing (fin/rst already in meta)
            reg_flags_syn.read(meta.flags_syn, meta.flow_hash);
            reg_flags_ack.read(meta.flags_ack, meta.flow_hash);
            // meta.flags_fin and meta.flags_rst already set by extract_tcp_flags()
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
            reg_flags_syn.write(meta.flow_hash, 1w0);
            reg_flags_ack.write(meta.flow_hash, 1w0);
            reg_flags_fin.write(meta.flow_hash, 1w0);
            reg_flags_rst.write(meta.flow_hash, 1w0);
        }
    }

    // Helper to read aggregated flow features
    action read_flow_features() {
        reg_flags_syn.read(meta.flags_syn, meta.flow_hash);
        reg_flags_ack.read(meta.flags_ack, meta.flow_hash);
        reg_flags_fin.read(meta.flags_fin, meta.flow_hash);
        reg_flags_rst.read(meta.flags_rst, meta.flow_hash);
    }

    // PCA component 1 transformation
    action set_pc1_code(pca_code_t code) {
        meta.pc1_code = code;
    }

    table pca_component1 {
        key = {
            meta.duration        : range;
            meta.max_iat         : range;
            meta.urg_count       : range;
            meta.fwd_pkt_count   : range;
            meta.bwd_pkt_count   : range;
            meta.fwd_bytes       : range;
            meta.bwd_bytes       : range;
            meta.max_win_size    : range;
        }
        actions = {
            set_pc1_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 2 transformation
    action set_pc2_code(pca_code_t code) {
        meta.pc2_code = code;
    }

    table pca_component2 {
        key = {
            meta.duration        : range;
            meta.max_iat         : range;
            meta.urg_count       : range;
            meta.fwd_pkt_count   : range;
            meta.bwd_pkt_count   : range;
            meta.fwd_bytes       : range;
            meta.bwd_bytes       : range;
            meta.max_win_size    : range;
        }
        actions = {
            set_pc2_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 3 transformation
    action set_pc3_code(pca_code_t code) {
        meta.pc3_code = code;
    }

    table pca_component3 {
        key = {
            meta.duration        : range;
            meta.max_iat         : range;
            meta.urg_count       : range;
            meta.fwd_pkt_count   : range;
            meta.bwd_pkt_count   : range;
            meta.fwd_bytes       : range;
            meta.bwd_bytes       : range;
            meta.max_win_size    : range;
        }
        actions = {
            set_pc3_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 4 transformation
    action set_pc4_code(pca_code_t code) {
        meta.pc4_code = code;
    }

    table pca_component4 {
        key = {
            meta.duration        : range;
            meta.max_iat         : range;
            meta.urg_count       : range;
            meta.fwd_pkt_count   : range;
            meta.bwd_pkt_count   : range;
            meta.fwd_bytes       : range;
            meta.bwd_bytes       : range;
            meta.max_win_size    : range;
        }
        actions = {
            set_pc4_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 5 transformation
    action set_pc5_code(pca_code_t code) {
        meta.pc5_code = code;
    }

    table pca_component5 {
        key = {
            meta.duration        : range;
            meta.max_iat         : range;
            meta.urg_count       : range;
            meta.fwd_pkt_count   : range;
            meta.bwd_pkt_count   : range;
            meta.fwd_bytes       : range;
            meta.bwd_bytes       : range;
            meta.max_win_size    : range;
        }
        actions = {
            set_pc5_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // Shared classification result action
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }

    // RF tree 0 — range match on PCA codes, output vote to rf_votes[1:0]
    action set_rf_tree_0_vote(bit<2> vote) {
        meta.rf_votes[1:0] = vote;
    }

    table rf_tree_0 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_0_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 1 — range match on PCA codes, output vote to rf_votes[3:2]
    action set_rf_tree_1_vote(bit<2> vote) {
        meta.rf_votes[3:2] = vote;
    }

    table rf_tree_1 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_1_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 2 — range match on PCA codes, output vote to rf_votes[5:4]
    action set_rf_tree_2_vote(bit<2> vote) {
        meta.rf_votes[5:4] = vote;
    }

    table rf_tree_2 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_2_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 3 — range match on PCA codes, output vote to rf_votes[7:6]
    action set_rf_tree_3_vote(bit<2> vote) {
        meta.rf_votes[7:6] = vote;
    }

    table rf_tree_3 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_3_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 4 — range match on PCA codes, output vote to rf_votes[9:8]
    action set_rf_tree_4_vote(bit<2> vote) {
        meta.rf_votes[9:8] = vote;
    }

    table rf_tree_4 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_4_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 5 — range match on PCA codes, output vote to rf_votes[11:10]
    action set_rf_tree_5_vote(bit<2> vote) {
        meta.rf_votes[11:10] = vote;
    }

    table rf_tree_5 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_5_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 6 — range match on PCA codes, output vote to rf_votes[13:12]
    action set_rf_tree_6_vote(bit<2> vote) {
        meta.rf_votes[13:12] = vote;
    }

    table rf_tree_6 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_6_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF tree 7 — range match on PCA codes, output vote to rf_votes[15:14]
    action set_rf_tree_7_vote(bit<2> vote) {
        meta.rf_votes[15:14] = vote;
    }

    table rf_tree_7 {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
        }
        actions = {
            set_rf_tree_7_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // RF vote aggregation — exact match on packed vote field (16 bits)
    table rf_vote_classify {
        key = {
            meta.rf_votes : exact;
        }
        actions = {
            set_result;
            NoAction;
        }
        size = 65536;
    }

    apply {
        if (hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP)) {
            // Step 1: Compute bidirectional flow hash (canonical direction)
            compute_flow_hash();
            
            // Step 2: Extract TCP flags (if TCP)
            extract_tcp_flags();
            
            // Step 3: Update flow state.
            //         meta.flow_ended is set to 1 when the flow is complete
            //         (timeout exposing previous flow, or FIN/RST ending current flow).
            update_flow_state();
            
            // Step 4–7 only run when a complete flow is ready for classification.
            if (meta.flow_ended == 1w1) {
                // (Flags already snapshotted inside update_flow_state before register clear)
                
                // Step 4: Apply PCA transformations
            pca_component1.apply();
            pca_component2.apply();
            pca_component3.apply();
            pca_component4.apply();
            pca_component5.apply();

            // Step 6: Apply classifier
            // Initialize packed vote field
            meta.rf_votes = 16w0;
            rf_tree_0.apply();
            rf_tree_1.apply();
            rf_tree_2.apply();
            rf_tree_3.apply();
            rf_tree_4.apply();
            rf_tree_5.apply();
            rf_tree_6.apply();
            rf_tree_7.apply();
            rf_vote_classify.apply();

            // Step 7: Send digest with flow features and classification result
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
                meta.pc1_code,
                meta.pc2_code,
                meta.pc3_code,
                meta.pc4_code,
                meta.pc5_code,
                meta.ml_result
            });
            } // end if (meta.flow_ended == 1w1)
            
            // Step 8: Forward packet (always, regardless of flow state)
            ipv4_lpm.apply();
        }
    }
}

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
