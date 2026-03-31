/* -*- P4_16 -*- */
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

const bit<32> NB_ENTRIES = 65536;
const bit<32> MAX_REGISTER_ENTRIES = 65536;

#define BLOOM_FILTER_BIT_WIDTH 32
#define FLOW_TIMEOUT 20000000000  // 20s in nanoseconds

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

typedef bit<48> iat_t;
typedef bit<48> duration_t;
typedef bit<16> port_t;
typedef bit<32> bytes_t;
typedef bit<32> pca_code_t;   // Quantized code (PC)
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

    // PC transformed features (quantized)
    pca_code_t pc1_code;
    pca_code_t pc2_code;
    pca_code_t pc3_code;

    // Classification result
    inference_result_t ml_result;

    // RF packed vote field (8 trees x 3 bits)
    bit<24> rf_votes;
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

    action set_rf_tree_0_vote(bit<3> vote) {
        meta.rf_votes[2:0] = vote;
    }

    table rf_tree_0 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_0_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_1_vote(bit<3> vote) {
        meta.rf_votes[5:3] = vote;
    }

    table rf_tree_1 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_1_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_2_vote(bit<3> vote) {
        meta.rf_votes[8:6] = vote;
    }

    table rf_tree_2 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_2_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_3_vote(bit<3> vote) {
        meta.rf_votes[11:9] = vote;
    }

    table rf_tree_3 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_3_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_4_vote(bit<3> vote) {
        meta.rf_votes[14:12] = vote;
    }

    table rf_tree_4 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_4_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_5_vote(bit<3> vote) {
        meta.rf_votes[17:15] = vote;
    }

    table rf_tree_5 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_5_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_6_vote(bit<3> vote) {
        meta.rf_votes[20:18] = vote;
    }

    table rf_tree_6 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_6_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    action set_rf_tree_7_vote(bit<3> vote) {
        meta.rf_votes[23:21] = vote;
    }

    table rf_tree_7 {
        key = {
            meta.pc1_code                 : range;
            meta.pc2_code                 : range;
            meta.pc3_code                 : range;
        }
        actions = {
            set_rf_tree_7_vote;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    table rf_vote_classify {
        key = {
            meta.rf_votes : exact;
        }
        actions = {
            set_result;
            NoAction;
        }
        size = 16777216;
    }

    apply {
        if ((hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP ||
                                    meta.protocol == TYPE_ICMP)) ||
            hdr.arp.isValid()) {
            compute_flow_hash();
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

            update_time_last_t.apply();

            if (meta.is_reverse_dir == 1w0) {
                update_fwd_counters_t.apply();
            } else {
                update_bwd_counters_t.apply();
            }

            if (meta.protocol == TYPE_TCP) {
                update_tcp_features_t.apply();
            }

            if (meta.flow_ended == 1w0 && meta.protocol == TYPE_TCP &&
                    (meta.flags_fin > 32w0 || meta.flags_rst > 32w0)) {
                meta.flow_ended = 1w1;
                meta.duration   = ig_intr_md.ingress_mac_tstamp - meta.time_first;
                clear_flow_regs_t.apply();
            }

            if (meta.flow_ended == 1w1 &&
                    (meta.fwd_pkt_count + meta.bwd_pkt_count) >= 2) {

                // Apply PC transformations
                pca_component1.apply();
                pca_component2.apply();
                pca_component3.apply();

                // Apply classifier
                meta.rf_votes = 24w0;
                rf_tree_0.apply();
                rf_tree_1.apply();
                rf_tree_2.apply();
                rf_tree_3.apply();
                rf_tree_4.apply();
                rf_tree_5.apply();
                rf_tree_6.apply();
                rf_tree_7.apply();
                rf_vote_classify.apply();

                meta.send_digest = 1w1;
            } // end if flow_ended

            if (hdr.ipv4.isValid()) {
                ipv4_lpm.apply();
            }
        }
    }
}

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
                meta.pc1_code,
                meta.pc2_code,
                meta.pc3_code,
                meta.min_iat,
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
