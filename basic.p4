/* -*- P4_16 -*- */
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

const bit<32> NB_ENTRIES = 65536;
const bit<32> MAX_REGISTER_ENTRIES = 65536;

#define BLOOM_FILTER_BIT_WIDTH 32
#define FLOW_TIMEOUT 20000000000  // 20s in nanoseconds

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
typedef bit<16> pca_code_t;   // Quantized code (PC)
typedef bit<8>  inference_result_t;

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

    // PC transformed features (quantized)
    pca_code_t pc1_code;
    pca_code_t pc2_code;
    pca_code_t pc3_code;

    // Classification result
    inference_result_t ml_result;

    // Timestamp
    bit<48> ingress_timestamp;
}

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
    pca_code_t pc1_code;
    pca_code_t pc2_code;
    pca_code_t pc3_code;

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

    action update_flow_state() {
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
            // Reset for new flow — mirrors RST/FIN reset block: ALL registers must be
            // written here, not just local vars.  Any register with a conditional write
            // below (urg_count, init_fwd_win, directional counters, etc.) would otherwise
            // carry stale values from the previous flow into the new one.
            time_first       = current_time;
            time_last        = current_time;
            max_iat          = 0;
            min_iat          = 0;
            fwd_pkt_count    = 0;
            bwd_pkt_count    = 0;
            fwd_bytes        = 0;
            bwd_bytes        = 0;
            max_win_size     = 0;
            urg_count        = 0;
            flags_syn        = 0;
            flags_ack        = 0;
            flags_fin        = 0;
            flags_rst        = 0;
            fwd_max_pkt_len  = 0;
            bwd_max_pkt_len  = 0;
            flags_psh        = 0;
            init_fwd_win     = 0;
            reg_time_first_pkt.write(meta.flow_hash, current_time);
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
            bloom_filter_1.write(meta.flow_hash,   1w1);  // new flow claims slot
            bloom_filter_2.write(meta.flow_hash_2, 1w1);
        }

        // First packet for a new flow (fresh empty slot)
        if (time_first == 0) {
            time_first = current_time;
            meta.is_first_packet = 1w1;
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            bloom_filter_1.write(meta.flow_hash,   1w1);  // claim empty slot
            bloom_filter_2.write(meta.flow_hash_2, 1w1);
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
        meta.max_iat = max_iat;
        meta.min_iat = min_iat;

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
        meta.fwd_pkt_count   = fwd_pkt_count;
        meta.bwd_pkt_count   = bwd_pkt_count;
        meta.fwd_bytes       = fwd_bytes;
        meta.bwd_bytes       = bwd_bytes;
        meta.fwd_max_pkt_len = fwd_max_pkt_len;
        meta.bwd_max_pkt_len = bwd_max_pkt_len;

        // Window size (max)
        if (meta.protocol == TYPE_TCP) {
            if (hdr.tcp.window > max_win_size) {
                max_win_size = hdr.tcp.window;
                reg_max_win_size.write(meta.flow_hash, max_win_size);
            }
        }
        meta.max_win_size = max_win_size;
        meta.init_fwd_win = init_fwd_win;

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
        meta.urg_count  = urg_count;
        meta.flags_syn  = flags_syn;
        meta.flags_ack  = flags_ack;
        meta.flags_fin  = flags_fin;
        meta.flags_rst  = flags_rst;
        meta.flags_psh  = flags_psh;

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
        }
    }


    // PC component 1 transformation
    action set_pc1_code(pca_code_t code) {
        meta.pc1_code = code;
    }

    table pca_component1 {
        key = {
            meta.protocol            : range;
            meta.canon_src_port      : range;
            meta.canon_dst_port      : range;
            meta.duration            : range;
            meta.max_iat             : range;
            meta.urg_count           : range;
            meta.fwd_pkt_count       : range;
            meta.bwd_pkt_count       : range;
            meta.fwd_bytes           : range;
            meta.bwd_bytes           : range;
            meta.max_win_size        : range;
            meta.flags_syn           : range;
            meta.flags_ack           : range;
            meta.flags_fin           : range;
            meta.flags_rst           : range;
            meta.min_iat             : range;
            meta.fwd_max_pkt_len     : range;
            meta.bwd_max_pkt_len     : range;
            meta.flags_psh           : range;
            meta.init_fwd_win        : range;
        }
        actions = {
            set_pc1_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PC component 2 transformation
    action set_pc2_code(pca_code_t code) {
        meta.pc2_code = code;
    }

    table pca_component2 {
        key = {
            meta.protocol            : range;
            meta.canon_src_port      : range;
            meta.canon_dst_port      : range;
            meta.duration            : range;
            meta.max_iat             : range;
            meta.urg_count           : range;
            meta.fwd_pkt_count       : range;
            meta.bwd_pkt_count       : range;
            meta.fwd_bytes           : range;
            meta.bwd_bytes           : range;
            meta.max_win_size        : range;
            meta.flags_syn           : range;
            meta.flags_ack           : range;
            meta.flags_fin           : range;
            meta.flags_rst           : range;
            meta.min_iat             : range;
            meta.fwd_max_pkt_len     : range;
            meta.bwd_max_pkt_len     : range;
            meta.flags_psh           : range;
            meta.init_fwd_win        : range;
        }
        actions = {
            set_pc2_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PC component 3 transformation
    action set_pc3_code(pca_code_t code) {
        meta.pc3_code = code;
    }

    table pca_component3 {
        key = {
            meta.protocol            : range;
            meta.canon_src_port      : range;
            meta.canon_dst_port      : range;
            meta.duration            : range;
            meta.max_iat             : range;
            meta.urg_count           : range;
            meta.fwd_pkt_count       : range;
            meta.bwd_pkt_count       : range;
            meta.fwd_bytes           : range;
            meta.bwd_bytes           : range;
            meta.max_win_size        : range;
            meta.flags_syn           : range;
            meta.flags_ack           : range;
            meta.flags_fin           : range;
            meta.flags_rst           : range;
            meta.min_iat             : range;
            meta.fwd_max_pkt_len     : range;
            meta.bwd_max_pkt_len     : range;
            meta.flags_psh           : range;
            meta.init_fwd_win        : range;
        }
        actions = {
            set_pc3_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // Shared classification result action
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }

    // Decision Tree classification
    table ml_code {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    apply {
        if ((hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP ||
                                    meta.protocol == TYPE_ICMP)) ||
            hdr.arp.isValid()) {
            compute_flow_hash();
            // Only update state when no collision — colliding packets are forwarded
            // without corrupting another flow's registers.
            if (meta.hash_collision == 1w0) {
                update_flow_state();
            }

            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 2) {

                // Apply PC transformations
                pca_component1.apply();
                pca_component2.apply();
                pca_component3.apply();

                // Apply classifier
                ml_code.apply();

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
                    meta.pc1_code,
                    meta.pc2_code,
                    meta.pc3_code,
                    meta.ml_result
                });
            } // end if flow_ended

            if (hdr.ipv4.isValid()) {
                ipv4_lpm.apply();
            }
        }
    }
}

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
