#!/usr/bin/env python3
"""
P4 Code Generator for Scalable PCA-based ML Classification
Automatically generates basic.p4 with support for N PCA components
"""

import json
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class P4CodeGenerator:
    def __init__(self, n_components=2, bits=16, output_file='basic.p4'):
        self.n_components = n_components
        self.bits = bits
        self.output_file = output_file
        
    def generate_header(self):
        """Generate P4 file header with includes and constants."""
        return '''/* -*- P4_16 -*- */

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
typedef bit<''' + str(self.bits) + '''> pca_code_t;       // PCA component code (quantized)
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
'''

    def generate_metadata(self):
        """Generate metadata struct with dynamic PCA component codes."""
        code = '''
struct metadata {
    // Flow identification (5-tuple)
    bit<16> srcPort;
    bit<16> dstPort;
    
    // Raw features (extracted from packets)
    feature1_t iat;      // inter-arrival time (nanoseconds)
    feature2_t pkt_len;  // packet length (frame length including Ethernet)
    feature3_t diffLen;  // difference in packet length
    
    // PCA-transformed features (quantized)
'''
        # Add PCA component codes dynamically
        for i in range(1, self.n_components + 1):
            code += f'    pca_code_t pc{i}_code;\n'
        
        code += '''    
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
'''
        # Add PCA component codes to digest
        for i in range(1, self.n_components + 1):
            code += f'    pca_code_t pc{i}_code; // PCA component {i}\n'
        
        code += '''    inference_result_t class_value; //class of traffic in this flow
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
'''

    def generate_ingress_forwarding(self):
        """Generate basic forwarding logic."""
        return '''
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
'''

    def generate_pca_tables(self):
        """Generate PCA component transformation tables dynamically."""
        code = ''
        for i in range(1, self.n_components + 1):
            code += f'''
    // PCA transformation: map 3 raw features to PC{i} quantized code
    action set_pc{i}_code(pca_code_t code) {{
        meta.pc{i}_code = code;
    }}
    table pca_component{i} {{
        key = {{
            meta.iat      : range;
            meta.pkt_len  : range;
            meta.diffLen  : range;
        }}
        actions = {{
            set_pc{i}_code;
            NoAction;
        }}
        size = NB_ENTRIES;
    }}
'''
        return code

    def generate_ml_table(self):
        """Generate decision tree classification table with dynamic PCA keys."""
        code = '''
    // Decision Tree classification: map PCA components to traffic class
    action set_result(inference_result_t val) {
        meta.ml_result = val;
    }
    table ml_code {
        key = {
'''
        # Add all PCA component codes as keys
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
        return code

    def generate_feature_extraction(self):
        """Generate feature extraction actions."""
        return '''
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
'''

    def generate_apply_block(self):
        """Generate apply block with dynamic PCA table applications."""
        code = '''    
    apply {
        if (hdr.ipv4.isValid()) {
            // Step 1: Extract raw features from current and previous packets
            get_iat();
            get_pkt_len();
            get_diff_len();
            
            // Step 2: Transform 3 raw features to PCA component codes
'''
        # Apply PCA component tables dynamically
        for i in range(1, self.n_components + 1):
            code += f'            pca_component{i}.apply();\n'
        
        code += '''            
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
'''
        # Add PCA component codes to digest call
        for i in range(1, self.n_components + 1):
            code += f'                meta.pc{i}_code,\n'
        
        code += '''                meta.ml_result
            });
            
            // Forward packet based on destination IP
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
        logger.info(f"Generating P4 code with {self.n_components} PCA components")
        
        code = self.generate_header()
        code += self.generate_metadata()
        code += self.generate_parser()
        code += self.generate_ingress_forwarding()
        code += self.generate_pca_tables()
        code += self.generate_ml_table()
        code += self.generate_feature_extraction()
        code += self.generate_apply_block()
        code += self.generate_egress_and_tail()
        
        return code

    def write_to_file(self):
        """Write generated P4 code to file."""
        code = self.generate()
        
        with open(self.output_file, 'w') as f:
            f.write(code)
        
        logger.info(f"Successfully generated {self.output_file}")
        logger.info(f"  - PCA components: {self.n_components}")
        logger.info(f"  - PCA tables: pca_component1 to pca_component{self.n_components}")
        logger.info(f"  - ML table keys: pc1_code to pc{self.n_components}_code")


def detect_n_components(params_file='tables/pca_encoding_params.json', 
                        commands_file='tables/s1-commands.txt'):
    """
    Automatically detect the number of PCA components from existing files.
    
    Priority:
    1. Read from pca_encoding_params.json if exists
    2. Parse s1-commands.txt to count pca_component tables
    """
    # Try reading from JSON params file
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
                n_components = params.get('n_components')
                bits = params.get('bits', 16)
                if n_components:
                    logger.info(f"Detected {n_components} PCA components with {bits} bits from {params_file}")
                    return n_components, bits
        except Exception as e:
            logger.warning(f"Could not read {params_file}: {e}")
    
    # Try parsing s1-commands.txt to count pca_component tables
    if os.path.exists(commands_file):
        try:
            with open(commands_file, 'r') as f:
                component_tables = set()
                for line in f:
                    if 'pca_component' in line:
                        # Extract table name like "MyIngress.pca_component1"
                        parts = line.split()
                        for part in parts:
                            if 'pca_component' in part:
                                # Extract number from pca_componentN
                                import re
                                match = re.search(r'pca_component(\d+)', part)
                                if match:
                                    component_tables.add(int(match.group(1)))
                
                if component_tables:
                    n_components = max(component_tables)
                    logger.info(f"Detected {n_components} PCA components from {commands_file} (bits defaulting to 16)")
                    return n_components, 16
        except Exception as e:
            logger.warning(f"Could not parse {commands_file}: {e}")
    
    logger.warning("Could not auto-detect number of PCA components, defaulting to 2 components with 16 bits")
    return 2, 16


def main():
    parser = argparse.ArgumentParser(
        description='Generate scalable P4 code for PCA-based ML classification'
    )
    parser.add_argument(
        '--output',
        default='../basic.p4',
        help='Output P4 file path (default: ../basic.p4)'
    )
    parser.add_argument(
        '--params-file',
        default='tables/pca_encoding_params.json',
        help='PCA encoding parameters file for auto-detection'
    )
    parser.add_argument(
        '--commands-file',
        default='tables/s1-commands.txt',
        help='S1 commands file for auto-detection'
    )
    
    args = parser.parse_args()
    
    # Always auto-detect number of components and bits
    n_components, bits = detect_n_components(args.params_file, args.commands_file)
    
    # Generate P4 code
    generator = P4CodeGenerator(n_components=n_components, bits=bits, output_file=args.output)
    generator.write_to_file()
    
    logger.info("\nGeneration complete!")
    logger.info("PCA components were auto-detected from existing files.")
    logger.info("To compile: make")
    logger.info(f"To view generated file: cat {args.output}")


if __name__ == '__main__':
    main()
