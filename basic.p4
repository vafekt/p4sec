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
#define FLOW_TIMEOUT 15000000  // 15 seconds in microseconds

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
    // Flow identification (5-tuple)
    ip4Addr_t src_ip;
    ip4Addr_t dst_ip;
    port_t src_port;
    port_t dst_port;
    bit<8>  protocol;
    
    // Flow state tracking
    bit<32> flow_hash;
    bit<32> flow_hash_2;
    bit<1>  is_first_packet;
    bit<1>  hash_collision;
    
    // Flow-based features
    iat_t iat;
    duration_t duration;
    bytes_t total_bytes;
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
    pca_code_t pc6_code;
    pca_code_t pc7_code;
    pca_code_t pc8_code;
    
    // Classification result
    inference_result_t ml_result;
    
    // Timestamp
    bit<48> ingress_timestamp;
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
    iat_t iat;
    duration_t duration;
    bytes_t total_bytes;
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
    pca_code_t pc6_code;
    pca_code_t pc7_code;
    pca_code_t pc8_code;
    
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
    register<bytes_t>(MAX_REGISTER_ENTRIES) reg_total_bytes;      // Total bytes in flow
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_syn;        // SYN flag seen
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_ack;        // ACK flag seen
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_fin;        // FIN flag seen
    register<flags_t>(MAX_REGISTER_ENTRIES) reg_flags_rst;        // RST flag seen
    register<iat_t>(MAX_REGISTER_ENTRIES) reg_sum_iat;            // Sum of IATs for averaging
    register<bit<32>>(MAX_REGISTER_ENTRIES) reg_pkt_count;        // Packet count in flow
    
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

    // Helper action to compute flow hash
    action compute_flow_hash() {
        hash(meta.flow_hash, HashAlgorithm.crc16, (bit<16>)0, 
            {meta.src_ip, meta.dst_ip, meta.src_port, meta.dst_port, meta.protocol},
            (bit<32>)MAX_REGISTER_ENTRIES);
        
        hash(meta.flow_hash_2, HashAlgorithm.crc32, (bit<16>)0,
            {meta.src_ip, meta.dst_ip, meta.src_port, meta.dst_port, meta.protocol},
            (bit<32>)MAX_REGISTER_ENTRIES);
        
        // Mark in bloom filter
        bloom_filter.write(meta.flow_hash, 1);
    }

    // Helper to extract TCP flags
    action extract_tcp_flags() {
        if (meta.protocol == TYPE_TCP) {
            meta.flags_syn = hdr.tcp.ctrl[5:5];   // SYN bit
            meta.flags_ack = hdr.tcp.ctrl[4:4];   // ACK bit
            meta.flags_fin = hdr.tcp.ctrl[0:0];   // FIN bit
            meta.flags_rst = hdr.tcp.ctrl[2:2];   // RST bit
        }
    }

    // Helper to update flow state
    action update_flow_state() {
        bit<48> current_time = standard_metadata.ingress_global_timestamp;
        bit<48> time_first;
        bit<48> time_last;
        bit<32> pkt_count;
        bytes_t total_bytes;
        iat_t sum_iat;
        
        // Read current state
        reg_time_first_pkt.read(time_first, meta.flow_hash);
        reg_time_last_pkt.read(time_last, meta.flow_hash);
        reg_total_bytes.read(total_bytes, meta.flow_hash);
        reg_pkt_count.read(pkt_count, meta.flow_hash);
        reg_sum_iat.read(sum_iat, meta.flow_hash);
        
        if (time_first == 0) {
            // First packet of flow
            meta.is_first_packet = 1;
            reg_time_first_pkt.write(meta.flow_hash, current_time);
            reg_pkt_count.write(meta.flow_hash, 1);
            meta.iat = 0;
            meta.duration = 0;
            meta.total_bytes = (bytes_t)standard_metadata.packet_length;
        } else {
            // Subsequent packet
            meta.is_first_packet = 0;
            
            // Calculate IAT (inter-arrival time)
            iat_t current_iat = current_time - time_last;
            
            // Update sum of IATs
            sum_iat = sum_iat + current_iat;
            reg_sum_iat.write(meta.flow_hash, sum_iat);
            
            // Use sum_iat directly (right-shift as approximation for averaging)
            // This maintains statistical properties for ML without runtime division
            meta.iat = sum_iat >> 2;  // Approximate average by right-shifting
            
            // Update total bytes
            total_bytes = total_bytes + (bytes_t)standard_metadata.packet_length;
            meta.total_bytes = total_bytes;
            
            // Update duration (from first to current packet)
            meta.duration = current_time - time_first;
            
            // Write updated state
            reg_pkt_count.write(meta.flow_hash, pkt_count + 1);
            reg_total_bytes.write(meta.flow_hash, total_bytes);
        }
        
        // Always update last packet time and flags
        reg_time_last_pkt.write(meta.flow_hash, current_time);
        
        // Update TCP flags (use bitwise OR to aggregate)
        if (meta.flags_syn == 1) {
            reg_flags_syn.write(meta.flow_hash, 1);
        }
        if (meta.flags_ack == 1) {
            reg_flags_ack.write(meta.flow_hash, 1);
        }
        if (meta.flags_fin == 1) {
            reg_flags_fin.write(meta.flow_hash, 1);
        }
        if (meta.flags_rst == 1) {
            reg_flags_rst.write(meta.flow_hash, 1);
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
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
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
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
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
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
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
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
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
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
        }
        actions = {
            set_pc5_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 6 transformation
    action set_pc6_code(pca_code_t code) {
        meta.pc6_code = code;
    }

    table pca_component6 {
        key = {
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
        }
        actions = {
            set_pc6_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 7 transformation
    action set_pc7_code(pca_code_t code) {
        meta.pc7_code = code;
    }

    table pca_component7 {
        key = {
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
        }
        actions = {
            set_pc7_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA component 8 transformation
    action set_pc8_code(pca_code_t code) {
        meta.pc8_code = code;
    }

    table pca_component8 {
        key = {
            meta.iat         : range;
            meta.duration    : range;
            meta.src_port    : range;
            meta.dst_port    : range;
            meta.total_bytes : range;
            meta.flags_syn   : range;
            meta.flags_ack   : range;
            meta.flags_fin   : range;
            meta.flags_rst   : range;
        }
        actions = {
            set_pc8_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // Decision Tree classification using PCA components
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }

    table ml_code {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
            meta.pc3_code : range;
            meta.pc4_code : range;
            meta.pc5_code : range;
            meta.pc6_code : range;
            meta.pc7_code : range;
            meta.pc8_code : range;
        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    apply {
        if (hdr.ipv4.isValid() && (meta.protocol == TYPE_TCP || meta.protocol == TYPE_UDP)) {
            // Step 1: Compute flow hash
            compute_flow_hash();
            
            // Step 2: Extract TCP flags (if TCP)
            extract_tcp_flags();
            
            // Step 3: Update flow state and calculate features
            update_flow_state();
            
            // Step 4: Read aggregated flow features
            read_flow_features();
            
            // Step 5: Apply PCA transformations
            pca_component1.apply();
            pca_component2.apply();
            pca_component3.apply();
            pca_component4.apply();
            pca_component5.apply();
            pca_component6.apply();
            pca_component7.apply();
            pca_component8.apply();
            
            // Step 6: Apply Decision Tree classification
            ml_code.apply();
            
            // Step 7: Send digest with flow features and classification result
            digest<digest_t>(1, {
                meta.src_ip,
                meta.dst_ip,
                meta.src_port,
                meta.dst_port,
                meta.protocol,
                meta.iat,
                meta.duration,
                meta.total_bytes,
                meta.flags_syn,
                meta.flags_ack,
                meta.flags_fin,
                meta.flags_rst,
                meta.pc1_code,
                meta.pc2_code,
                meta.pc3_code,
                meta.pc4_code,
                meta.pc5_code,
                meta.pc6_code,
                meta.pc7_code,
                meta.pc8_code,
                meta.ml_result
            });
            
            // Step 8: Forward packet
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
