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
import pandas as pd
import pickle
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

# Traffic class label mapping will be loaded dynamically from model
CLASS_LABELS = {}

class FlowAggregator:
    """Aggregate per-packet digests into flow-level outputs."""
    def __init__(self, timeout_s=5.0):
        self.timeout_s = timeout_s
        self.flows = {}

    def update(self, key, features, pca_codes, class_id, class_label, flags):
        now = time.time()
        entry = self.flows.get(key)
        if entry is None:
            entry = {
                "last_seen": now,
                "features": features,
                "pca_codes": pca_codes,
                "class_id": class_id,
                "class_label": class_label,
                "flags": flags,
            }
            self.flows[key] = entry
        else:
            entry["last_seen"] = now
            entry["features"] = features
            entry["pca_codes"] = pca_codes
            entry["class_id"] = class_id
            entry["class_label"] = class_label
            entry["flags"] = flags

        # If FIN/RST seen, flush immediately
        if flags.get("fin") == 1 or flags.get("rst") == 1:
            self.flows.pop(key, None)
            return [entry]

        return []

    def flush_expired(self):
        now = time.time()
        expired = []
        for key, entry in list(self.flows.items()):
            if (now - entry["last_seen"]) >= self.timeout_s:
                expired.append(entry)
                del self.flows[key]
        return expired

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

def load_class_labels(model_path):
    """Load class labels from trained DecisionTree model.
    
    Dynamically extracts class labels from the model to support flexible
    training with any number of classes or different label names.
    
    Args:
        model_path: Path to the pickled DecisionTree model file
        
    Returns:
        Dictionary mapping class_id (int) to class_label (str)
    """
    try:
        dt = pd.read_pickle(model_path)
        label_mapping = {idx: label for idx, label in enumerate(dt.classes_)}
        print("=== Class Label Mapping (from model) ===")
        for class_id, label in sorted(label_mapping.items()):
            print(f"  {class_id}: {label}")
        print()
        return label_mapping
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {model_path}")
        print("Using empty label mapping - will default to 'unknown' for all classes")
        return {}
    except Exception as e:
        print(f"WARNING: Error loading model from {model_path}: {e}")
        print("Using empty label mapping - will default to 'unknown' for all classes")
        return {}

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

def main(p4info_file_path, bmv2_file_path, runtime_cli_path, flow_timeout_s):
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
    flow_aggregator = FlowAggregator(timeout_s=flow_timeout_s)

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

    # Load class labels dynamically from trained model
    model_path = os.path.join(script_dir, 'model/dt.model')
    global CLASS_LABELS
    CLASS_LABELS = load_class_labels(model_path)

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
            out.write(f"src_ip,src_port,dst_ip,dst_port,proto,iat,duration,total_bytes,pkt_count,flags_syn,flags_ack,flags_fin,flags_rst,{pca_headers},class_id,class_label\n")
            packet_id = 0
            
            while True:
                msg = dclient.get_digest(timeout=0.5)
                if msg is None:
                    # Periodically flush expired flows
                    for entry in flow_aggregator.flush_expired():
                        # Only output flows with 2+ packets (match offline extraction)
                        if entry["features"]["pkt_count"] < 2:
                            continue
                        
                        packet_id += 1
                        pca_width = len(str((1 << pca_bits) - 1))
                        pca_display = ' '.join([f'PCA{i+1}={entry["pca_codes"][i]:<{pca_width}}' for i in range(n_components)])
                        f = entry["features"]
                        flags = entry["flags"]
                        class_id = entry["class_id"]
                        class_label = entry["class_label"]

                        print(f"[{packet_id:<4}] {f['src_ip']:>15}:{f['src_port']:<5} -> {f['dst_ip']:>15}:{f['dst_port']:<5} | "
                              f"IAT={f['iat']:<12} Dur={f['duration']:<12} Bytes={f['total_bytes']:<8} Pkts={f['pkt_count']:<4} | "
                              f"Flags(S/A/F/R)={flags['syn']}/{flags['ack']}/{flags['fin']}/{flags['rst']} | "
                              f"{pca_display} | "
                              f"Class={class_label}({class_id})")

                        out.write(f"{f['src_ip']},{f['src_port']},{f['dst_ip']},{f['dst_port']},{f['proto']},"
                                  f"{f['iat']},{f['duration']},{f['total_bytes']},{f['pkt_count']},{flags['syn']},{flags['ack']},{flags['fin']},{flags['rst']},"
                                  f"{','.join([str(code) for code in entry['pca_codes']])},{class_id},{class_label}\n")
                        out.flush()
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
                    
                    # Extract flow-based features
                    iat = bytes_to_int(st[5].bitstring)           # Inter-Arrival Time
                    duration = bytes_to_int(st[6].bitstring)      # Flow duration
                    total_bytes = bytes_to_int(st[7].bitstring)   # Total bytes
                    pkt_count = bytes_to_int(st[8].bitstring)     # Packet count
                    flags_syn = bytes_to_int(st[9].bitstring)     # SYN flag
                    flags_ack = bytes_to_int(st[10].bitstring)    # ACK flag
                    flags_fin = bytes_to_int(st[11].bitstring)    # FIN flag
                    flags_rst = bytes_to_int(st[12].bitstring)    # RST flag
                    
                    # Extract PCA component codes dynamically
                    pca_codes = []
                    for i in range(n_components):
                        pca_codes.append(bytes_to_int(st[13 + i].bitstring))
                    
                    # Class ID is at position 12 + n_components
                    class_id = bytes_to_int(st[13 + n_components].bitstring)

                    class_label = CLASS_LABELS.get(class_id, "unknown")

                    features = {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "proto": proto,
                        "iat": iat,
                        "duration": duration,
                        "total_bytes": total_bytes,
                        "pkt_count": pkt_count,
                    }
                    flags = {
                        "syn": flags_syn,
                        "ack": flags_ack,
                        "fin": flags_fin,
                        "rst": flags_rst,
                    }

                    for entry in flow_aggregator.update(
                        (src_ip, dst_ip, src_port, dst_port, proto),
                        features,
                        pca_codes,
                        class_id,
                        class_label,
                        flags,
                    ):
                        # Only output flows with 2+ packets (match offline extraction)
                        if entry["features"]["pkt_count"] < 2:
                            continue
                        packet_id += 1
                        pca_width = len(str((1 << pca_bits) - 1))
                        pca_display = ' '.join([f'PCA{i+1}={entry["pca_codes"][i]:<{pca_width}}' for i in range(n_components)])
                        f = entry["features"]
                        flags = entry["flags"]
                        class_id = entry["class_id"]
                        class_label = entry["class_label"]

                        print(f"[{packet_id:<4}] {f['src_ip']:>15}:{f['src_port']:<5} -> {f['dst_ip']:>15}:{f['dst_port']:<5} | "
                              f"IAT={f['iat']:<12} Dur={f['duration']:<12} Bytes={f['total_bytes']:<8} Pkts={f['pkt_count']:<4} | "
                              f"Flags(S/A/F/R)={flags['syn']}/{flags['ack']}/{flags['fin']}/{flags['rst']} | "
                              f"{pca_display} | "
                              f"Class={class_label}({class_id})")

                        out.write(f"{f['src_ip']},{f['src_port']},{f['dst_ip']},{f['dst_port']},{f['proto']},"
                                  f"{f['iat']},{f['duration']},{f['total_bytes']},{f['pkt_count']},{flags['syn']},{flags['ack']},{flags['fin']},{flags['rst']},"
                                  f"{','.join([str(code) for code in entry['pca_codes']])},{class_id},{class_label}\n")
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
    parser.add_argument('--flow-timeout', type=float, default=5.0,
                        help='Flow inactivity timeout in seconds before printing a flow summary (default: 5.0)')
    args = parser.parse_args()

    # Validate input files
    for file_path, name in [(args.p4info, 'P4Info'),
                             (args.bmv2_json, 'BMv2 JSON'),
                             (args.runtime_cli, 'Runtime CLI')]:
        if not os.path.exists(file_path):
            print(f"ERROR: {name} file not found: {file_path}")
            sys.exit(1)

    try:
        main(args.p4info, args.bmv2_json, args.runtime_cli, args.flow_timeout)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)