/* -*- P4_16 -*- */

#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x800;
const bit<8>  TYPE_TCP  = 6;
const bit<8>  TYPE_UDP  = 17;

const bit<32> NB_ENTRIES = 8192;

//write and read the first element of a register (which contains an array of elements)
#define FIRST_INDEX ((bit<32>)0)
#define WRITE_REG(r, v) r.write(FIRST_INDEX, v)
#define READ_REG(r,  v) r.read(v, FIRST_INDEX)

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;
typedef bit<64> feature1_t;       // IAT (inter-arrival time)
typedef bit<16> feature2_t;       // packet length
typedef bit<32> feature3_t;       // diff of packet length
typedef bit<16> pca_code_t;       // PCA component code (quantized)
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
    bit<16>   totalLen;  // feature2: packet length
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

// tcp header
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

/* UDP header */
header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> udpTotalLen;
    bit<16> checksum;
}

struct metadata {
    // Flow identification (5-tuple)
    bit<16> srcPort;
    bit<16> dstPort;
    
    // Raw features (extracted from packets)
    feature1_t iat;      // inter-arrival time (nanoseconds)
    feature2_t pkt_len;  // packet length (frame length including Ethernet)
    feature3_t diffLen;  // difference in packet length
    
    // PCA-transformed features (quantized)
    pca_code_t pc1_code;
    pca_code_t pc2_code;
    
    // Classification result
    inference_result_t ml_result;
}

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;
    udp_t        udp;
}

struct digest_t {
    //flow ID is a 5-tuples
    ip4Addr_t srcAddr;  //32 bits
    ip4Addr_t dstAddr;
    bit<16> srcPort;
    bit<16> destPort;
    bit<8> protocol;
    feature1_t iat;
    feature2_t len;
    feature3_t diffLen;
    pca_code_t pc1_code; // PCA component 1
    pca_code_t pc2_code; // PCA component 2
    inference_result_t class_value; //class of traffic in this flow
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
        transition select(hdr.ipv4.protocol) {
            TYPE_TCP: parse_tcp;
            TYPE_UDP: parse_udp;
            default : accept;
        }
    }

    state parse_tcp {
        packet.extract(hdr.tcp);
        //remember src and dst ports to identify this flow
        meta.dstPort = hdr.tcp.dstPort;
        meta.srcPort = hdr.tcp.srcPort;
        transition accept;
    }

    state parse_udp {
        packet.extract(hdr.udp);
        meta.dstPort = hdr.udp.dstPort;
        meta.srcPort = hdr.udp.srcPort;
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

    /* default table and its actions for packet forwarding */
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

    // PCA transformation: map 3 raw features to PC1 quantized code
    action set_pc1_code(pca_code_t code) {
        meta.pc1_code = code;
    }
    table pca_component1 {
        key = {
            meta.iat      : range;
            meta.pkt_len  : range;
            meta.diffLen  : range;
        }
        actions = {
            set_pc1_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // PCA transformation: map 3 raw features to PC2 quantized code
    action set_pc2_code(pca_code_t code) {
        meta.pc2_code = code;
    }
    table pca_component2 {
        key = {
            meta.iat      : range;
            meta.pkt_len  : range;
            meta.diffLen  : range;
        }
        actions = {
            set_pc2_code;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // Decision Tree classification: map PCA components to traffic class
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }
    table ml_code {
        key = {
            meta.pc1_code : range;
            meta.pc2_code : range;
        }
        actions = {
            set_result;
            NoAction;
        }
        size = NB_ENTRIES;
    }

    // Extract IAT (inter-arrival time) from packet timestamps
    register<feature1_t>(1) last_ts_reg;
    action get_iat() {
        feature1_t last;
        feature1_t now = (feature1_t) standard_metadata.ingress_global_timestamp * 1000;
        READ_REG(last_ts_reg, last);
        
        if (last != 0) {
            meta.iat = now - last;
        } else {
            // First packet: set IAT to 0
            meta.iat = 0;
        }
        
        WRITE_REG(last_ts_reg, now);
    }

    // Extract packet length (full frame length including Ethernet)
    // IMPORTANT: This must match data_extraction.py which uses packet.length (full frame)
    action get_pkt_len() {
        meta.pkt_len = (feature2_t) standard_metadata.packet_length;
    }

    // Extract diffLen (difference in packet length from previous packet)
    // IMPORTANT: Must match data_extraction.py formula: diff_len = (now - last) + 65535
    register<feature3_t>(1) last_len_reg;
    action get_diff_len() {
        feature3_t last;
        feature3_t now = (feature3_t) hdr.ipv4.totalLen;
        READ_REG(last_len_reg, last);
        
        if (last != 0) {
            // Match data_extraction.py: diff_len = (now - last) + 65535
            // This can produce negative intermediate values, so use bit arithmetic
            meta.diffLen = (now + 0xFFFF - last);
        } else {
            // First packet: set diffLen to 0
            meta.diffLen = 0;
        }
        
        WRITE_REG(last_len_reg, now);
    }
    
    apply {
        if (hdr.ipv4.isValid()) {
            // Step 1: Extract raw features from current and previous packets
            get_iat();
            get_pkt_len();
            get_diff_len();
            
            // Step 2: Transform 3 raw features to PCA component codes
            pca_component1.apply();
            pca_component2.apply();
            
            // Step 3: Use PCA components for Decision Tree classification
            ml_code.apply();
            
            // Send digest to controller with classification result and features
            digest<digest_t>(1, {
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr,
                meta.srcPort,
                meta.dstPort,
                hdr.ipv4.protocol,
                meta.iat,
                meta.pkt_len,
                meta.diffLen,
                meta.pc1_code,
                meta.pc2_code,
                meta.ml_result
            });
            
            // Forward packet based on destination IP
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
