#!/usr/bin/env python3
"""
P4 Runtime Controller for PCA + Decision Tree Traffic Classification.

This controller:
1. Loads P4 program rules into a BMv2 switch via simple_switch_CLI
2. Listens for digest messages containing packet features and classifications
3. Records predictions to CSV for analysis
"""

import argparse
import os
import sys
import time
import grpc
import subprocess
import json
from time import sleep
from queue import Queue, Empty
from threading import Thread

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  '../../../utils/')))
import p4runtime_lib.bmv2
import p4runtime_lib.helper
from p4runtime_lib.switch import ShutdownAllSwitchConnections
from p4.v1 import p4runtime_pb2, p4runtime_pb2_grpc

# Traffic class label mapping (matches training labels, unknown as fallback)
CLASS_LABELS = {
    0: "skype",
    1: "webex",
    2: "whasapp"
}

def load_switch_cli(sw, runtime_cli, thrift_port=9090):
    """Load P4 table rules via simple_switch_CLI."""
    print(f"Loading P4 rules from {runtime_cli}...")
    os.makedirs('logs', exist_ok=True)
    try:
        with open(runtime_cli, 'r') as fin, open('logs/cli_output.log', 'w') as fout:
            subprocess.Popen(
                ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.STDOUT
            )
        print(f"Rules loaded. CLI output logged to logs/cli_output.log")
    except FileNotFoundError as e:
        print(f"ERROR: Runtime CLI file not found: {e}")
        raise

def build_digest_entry(p4info_helper, digest_name):
    """Build a DigestEntry configuration for the switch."""
    de = p4runtime_pb2.DigestEntry()
    de.digest_id = p4info_helper.get_digests_id(digest_name)
    de.config.max_timeout_ns = 0
    de.config.max_list_size = 1
    de.config.ack_timeout_ns = 1
    return de

def install_digest(p4info_helper, sw, digest_name):
    """Install digest configuration on the switch."""
    de = build_digest_entry(p4info_helper, digest_name)
    req = p4runtime_pb2.WriteRequest()
    req.device_id = sw.device_id
    req.election_id.low = 1
    u = req.updates.add()
    u.type = p4runtime_pb2.Update.INSERT
    u.entity.digest_entry.CopyFrom(de)
    try:
        sw.client_stub.Write(req)
        print(f"Digest '{digest_name}' installed successfully.")
    except grpc.RpcError as e:
        print(f"WARNING: Digest installation failed: {e.code().name} - {e.details()}")
        raise

def bytes_to_int(bb):
    """Convert byte array to integer (big-endian)."""
    v = 0
    for b in bb:
        v = (v << 8) + int(b)
    return v

def bytes_to_ip(bb):
    """Convert byte array to IPv4 address string."""
    return '.'.join(str(b) for b in bb)

class DigestClient:
    """Dedicated P4Runtime client for receiving digest messages from the switch.
    
    Uses a separate gRPC channel with higher election ID to avoid conflicts
    with the control connection.
    """
    def __init__(self, address, device_id, election_id=2):
        self.address = address
        self.device_id = device_id
        self.election_id = election_id
        self.channel = None
        self.stub = None
        self.req_q = None
        self.resp_it = None
        self.out_q = Queue()
        self._stop = False
        self._t = None

    def _req_iter(self):
        while not self._stop:
            req = self.req_q.get()
            if req is None:
                break
            yield req

    def start(self):
        self._stop = False
        self.channel = grpc.insecure_channel(self.address)
        self.stub = p4runtime_pb2_grpc.P4RuntimeStub(self.channel)
        self.req_q = Queue()
        self.resp_it = self.stub.StreamChannel(self._req_iter())

        # Send arbitration with higher election id so this client becomes master for digests
        arb = p4runtime_pb2.StreamMessageRequest()
        arb.arbitration.device_id = self.device_id
        arb.arbitration.election_id.low = self.election_id
        self.req_q.put(arb)

        # Spin a reader thread
        self._t = Thread(target=self._reader_loop, daemon=True)
        self._t.start()

    def _reader_loop(self):
        """Continuously read digest messages from the switch."""
        backoff = 0.5
        while not self._stop:
            try:
                msg = next(self.resp_it)
            except StopIteration:
                if self._stop:
                    return
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            except Exception as e:
                if self._stop:
                    return
                print(f"Digest stream error: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue

            if msg.WhichOneof('update') == 'arbitration':
                backoff = 0.5
                continue

            if msg.WhichOneof('update') != 'digest':
                continue

            self.out_q.put(msg)

            # ACK the digest immediately
            try:
                ack = p4runtime_pb2.StreamMessageRequest()
                ack.digest_ack.digest_id = msg.digest.digest_id
                ack.digest_ack.list_id = msg.digest.list_id
                self.req_q.put(ack)
            except Exception as e:
                print(f"Failed to ACK digest: {e}")

    def get_digest(self, timeout=0.1):
        try:
            return self.out_q.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self._stop = True
        try:
            if self.req_q:
                self.req_q.put(None)
        except Exception:
            pass
        if self._t and self._t.is_alive():
            self._t.join(timeout=1.0)
        try:
            if self.channel:
                self.channel.close()
        except Exception:
            pass

def main(p4info_file_path, bmv2_file_path, runtime_cli_path):
    """Main controller logic: load rules, listen for digests, record predictions."""
    p4info_helper = p4runtime_lib.helper.P4InfoHelper(p4info_file_path)
    digest_name = "digest_t"

    print("\n=== P4Runtime Controller for Traffic Classification ===")
    print(f"P4Info: {p4info_file_path}")
    print(f"BMv2 JSON: {bmv2_file_path}")
    print(f"Runtime CLI: {runtime_cli_path}\n")

    # Load PCA configuration to determine number of components
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pca_config_path = os.path.join(script_dir, 'tables/pca_encoding_params.json')
    try:
        with open(pca_config_path, 'r') as f:
            pca_config = json.load(f)
            n_components = pca_config.get('n_components', 2)
            pca_bits = pca_config.get('bits', 16)
    except FileNotFoundError:
        print(f"WARNING: PCA config not found at {pca_config_path}, defaulting to 2 components")
        n_components = 2
        pca_bits = 16
    except json.JSONDecodeError:
        print(f"WARNING: Invalid JSON in {pca_config_path}, defaulting to 2 components")
        n_components = 2
        pca_bits = 16
    
    print(f"PCA Components: {n_components} (bits={pca_bits})\n")

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Control connection for programming
    s1 = p4runtime_lib.bmv2.Bmv2SwitchConnection(
        name='s1',
        address='127.0.0.1:50051',
        device_id=0,
        proto_dump_file='logs/s1-p4runtime-requests.log'
    )

    print("Connecting to P4 switch...")
    s1.MasterArbitrationUpdate()

    # Load rules via CLI
    load_switch_cli(s1, runtime_cli_path)
    sleep(2)
    print("Waiting for rules to be installed...\n")

    # Install digest configuration
    install_digest(p4info_helper, s1, digest_name)

    os.makedirs('logs', exist_ok=True)
    print("Listening for traffic digests...\n")

    # Dedicated digest client
    dclient = DigestClient(address='127.0.0.1:50051', device_id=0, election_id=2)
    dclient.start()

    try:
        os.makedirs('logs', exist_ok=True)
        with open("logs/predictions.csv", "w") as out:
            # Build CSV header dynamically based on number of PCA components
            pca_headers = ','.join([f'pc{i}_code' for i in range(1, n_components + 1)])
            out.write(f"src_ip,src_port,dst_ip,dst_port,proto,iat,pkt_len,diff_len,{pca_headers},class_id,class_label\n")
            packet_id = 0
            
            while True:
                msg = dclient.get_digest(timeout=0.5)
                if msg is None:
                    continue

                digest = msg.digest
                # Validate digest name
                name = p4info_helper.get_digests_name(digest.digest_id)
                if name != digest_name:
                    continue

                for el in digest.data:
                    st = el.struct.members
                    src_ip = bytes_to_ip(st[0].bitstring)
                    dst_ip = bytes_to_ip(st[1].bitstring)
                    src_port = bytes_to_int(st[2].bitstring)
                    dst_port = bytes_to_int(st[3].bitstring)
                    proto = bytes_to_int(st[4].bitstring)
                    iat = bytes_to_int(st[5].bitstring)
                    pkt_len = bytes_to_int(st[6].bitstring)
                    diff_len = bytes_to_int(st[7].bitstring)
                    
                    # Extract PCA component codes dynamically
                    pca_codes = []
                    for i in range(n_components):
                        pca_codes.append(bytes_to_int(st[8 + i].bitstring))
                    
                    # Class ID is at position 8 + n_components
                    class_id = bytes_to_int(st[8 + n_components].bitstring)

                    packet_id += 1
                    class_label = CLASS_LABELS.get(class_id, "unknown")
                    
                    # Build PCA display string and CSV values
                    pca_width = len(str((1 << pca_bits) - 1))
                    pca_display = ' '.join([f'PCA{i+1}={pca_codes[i]:<{pca_width}}' for i in range(n_components)])
                    pca_csv = ','.join([str(code) for code in pca_codes])
                    
                    print(f"[{packet_id:<4}] {src_ip:>15}:{src_port:<5} -> {dst_ip:>15}:{dst_port:<5} | "
                          f"IAT={iat:<12} Len={pkt_len:<4} DiffLen={diff_len:<5} | "
                          f"{pca_display} | "
                          f"Class={class_label}({class_id})")
                    
                    out.write(f"{src_ip},{src_port},{dst_ip},{dst_port},{proto},"
                              f"{iat},{pkt_len},{diff_len},{pca_csv},{class_id},{class_label}\n")
                    out.flush()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        dclient.stop()
        ShutdownAllSwitchConnections()

if __name__ == '__main__':
    # Default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_p4info = os.path.join(script_dir, '../basic.p4info')
    default_bmv2_json = os.path.join(script_dir, '../build/basic.json')
    default_runtime_cli = os.path.join(script_dir, 'tables/s1-commands.txt')
    
    parser = argparse.ArgumentParser(
        description='P4 Runtime Controller for Traffic Classification')
    parser.add_argument('--p4info', type=str, default=default_p4info,
                        help=f'Path to P4info file (default: {default_p4info})')
    parser.add_argument('--bmv2-json', type=str, default=default_bmv2_json,
                        help=f'Path to BMv2 JSON file (default: {default_bmv2_json})')
    parser.add_argument('--runtime-cli', type=str, default=default_runtime_cli,
                        help=f'Path to P4 table_add commands file (default: {default_runtime_cli})')
    args = parser.parse_args()

    # Validate input files
    for file_path, name in [(args.p4info, 'P4Info'),
                             (args.bmv2_json, 'BMv2 JSON'),
                             (args.runtime_cli, 'Runtime CLI')]:
        if not os.path.exists(file_path):
            print(f"ERROR: {name} file not found: {file_path}")
            sys.exit(1)

    try:
        main(args.p4info, args.bmv2_json, args.runtime_cli)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)