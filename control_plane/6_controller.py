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
import numpy as np
import math
import re
from time import sleep
from queue import Queue, Empty
from threading import Thread
from google.protobuf import text_format
from p4.config.v1 import p4info_pb2

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  '../../../utils/')))
import p4runtime_lib.bmv2
import p4runtime_lib.helper
from p4runtime_lib.switch import ShutdownAllSwitchConnections
from p4.v1 import p4runtime_pb2, p4runtime_pb2_grpc

# Traffic class label mapping will be loaded dynamically from model
CLASS_LABELS = {}

def make_canonical_key(src_ip, src_port, dst_ip, dst_port, proto):
    """
    Return a direction-normalised 5-tuple key so that both A->B and B->A
    digest messages aggregate into the same flow entry.

    Uses INTEGER IP comparison — matches P4 (bit<32> comparison) and
    1_data_extraction.py (ipaddress.ip_address integer conversion) exactly.
    String comparison (e.g. '192.168.1.100' < '192.168.1.52') disagrees with
    integer comparison and would split a bidirectional flow into two entries.
    """
    import ipaddress
    src_int = int(ipaddress.ip_address(src_ip))
    dst_int = int(ipaddress.ip_address(dst_ip))
    if src_int < dst_int:
        return (src_ip, src_port, dst_ip, dst_port, proto)
    elif src_int > dst_int:
        return (dst_ip, dst_port, src_ip, src_port, proto)
    else:
        if src_port <= dst_port:
            return (src_ip, src_port, dst_ip, dst_port, proto)
        else:
            return (dst_ip, dst_port, src_ip, src_port, proto)


class FlowAggregator:
    """Aggregate per-packet digests into flow-level outputs."""
    def __init__(self, timeout_s=5.0):
        self.timeout_s = timeout_s
        self.flows = {}

    def update(self, key, features, pca_codes, class_id, class_label, flags, xgb_scores=None, rf_votes=None):
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
                "xgb_scores": xgb_scores or {},
                "rf_votes": rf_votes or [],
            }
            self.flows[key] = entry
        else:
            entry["last_seen"] = now
            entry["features"] = features
            entry["pca_codes"] = pca_codes
            entry["class_id"] = class_id
            entry["class_label"] = class_label
            entry["flags"] = flags
            entry["xgb_scores"] = xgb_scores or {}
            entry["rf_votes"] = rf_votes or []

        # If FIN/RST seen, flush immediately
        if flags.get("fin", 0) > 0 or flags.get("rst", 0) > 0:
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

    def flush_all(self):
        """Flush every remaining flow regardless of timeout — call at shutdown."""
        remaining = list(self.flows.values())
        self.flows.clear()
        return remaining

def load_switch_cli(sw, runtime_cli, thrift_port=9090):
    """Load P4 table rules via simple_switch_CLI.

    Waits for the CLI process to finish so all table entries are fully
    installed before the controller starts listening for digests.
    With ~73 k entries this can take 60–90 seconds.
    """
    print(f"Loading P4 rules from {runtime_cli}...")
    n_entries = 0
    try:
        with open(runtime_cli, 'r') as f:
            for line in f:
                if line.lstrip().startswith('table_'):
                    n_entries += 1
    except Exception:
        pass
    if n_entries:
        print(f"  ({n_entries:,} table entries to install — please wait, this may take ~{max(10, n_entries // 800)} seconds)")

    os.makedirs('logs', exist_ok=True)
    try:
        with open(runtime_cli, 'r') as fin, open('logs/cli_output.log', 'w') as fout:
            proc = subprocess.Popen(
                ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.STDOUT
            )
            proc.wait()   # <-- BLOCK until all entries are installed
        # Quick sanity check
        errors = 0
        with open('logs/cli_output.log', 'r') as log:
            for line in log:
                if line.startswith('RuntimeCmd: Error'):
                    errors += 1
        if errors:
            print(f"WARNING: {errors} CLI errors during rule loading (check logs/cli_output.log)")
        else:
            print(f"Rules loaded successfully. CLI output logged to logs/cli_output.log")
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
    """Load class labels from a trained sklearn model.

    Args:
        model_path: Path to the pickled model file

    Returns:
        Dictionary mapping class_id (int) to class_label (str)
    """
    try:
        model = pd.read_pickle(model_path)
        label_mapping = {idx: label for idx, label in enumerate(model.classes_)}
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

def load_class_labels_from_rf_params(params_path, model_path):
    """Load class labels for RF from params JSON, fallback to model if needed."""
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        classes = params.get("classes")
        if classes:
            label_mapping = {idx: label for idx, label in enumerate(classes)}
            print("=== Class Label Mapping (from rf_params.json) ===")
            for class_id, label in sorted(label_mapping.items()):
                print(f"  {class_id}: {label}")
            print()
            return label_mapping
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: Error reading RF params from {params_path}: {e}")

    # Fallback to model if params are missing or invalid
    return load_class_labels(model_path)

def load_class_labels_from_xgb_params(params_path, model_path):
    """Load class labels for XGB from params JSON, fallback to model if needed."""
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        classes = params.get("classes")
        if classes:
            label_mapping = {idx: label for idx, label in enumerate(classes)}
            print("=== Class Label Mapping (from xgb_params.json) ===")
            for class_id, label in sorted(label_mapping.items()):
                print(f"  {class_id}: {label}")
            print()
            return label_mapping
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: Error reading XGB params from {params_path}: {e}")

    # Fallback to model if params are missing or invalid
    return load_class_labels(model_path)

def load_digest_schema(p4info_path, digest_name):
    """Load digest field names from P4Info.

    Returns a list of field names in order, or [] if not found.
    """
    try:
        p4info = p4info_pb2.P4Info()
        with open(p4info_path, 'r') as f:
            text_format.Merge(f.read(), p4info)
        digest = None
        for d in p4info.digests:
            if d.preamble.name == digest_name or d.preamble.alias == digest_name:
                digest = d
                break
        if digest is None:
            return []

        type_spec = digest.type_spec
        type_kind = type_spec.WhichOneof("type_spec")
        if type_kind != "struct":
            return []
        struct_name = type_spec.struct.name
        if not struct_name:
            return []
        if struct_name not in p4info.type_info.structs:
            return []
        struct_spec = p4info.type_info.structs[struct_name]
        return [m.name for m in struct_spec.members]
    except Exception as e:
        print(f"WARNING: Unable to load digest schema from P4Info: {e}")
        return []

def build_digest_field_index(digest_fields):
    return {name: idx for idx, name in enumerate(digest_fields)}

def find_first_field(name_to_index, candidates):
    for name in candidates:
        if name in name_to_index:
            return name
    return None

def parse_pca_field_names(digest_fields):
    """Return PCA field names sorted by numeric suffix (pc1_code, pc2_code, ...)."""
    pca = []
    for name in digest_fields:
        m = re.match(r"^pc(\d+)_code$", name)
        if m:
            pca.append((int(m.group(1)), name))
    if not pca:
        return []
    return [name for _, name in sorted(pca, key=lambda x: x[0])]

def parse_score_field_names(digest_fields):
    """Return score fields mapped by class index (e.g., score_c0, xgb_c1)."""
    scores = []
    patterns = [
        r"^(?:xgb_)?score[_-]?c?(\d+)$",
        r"^(?:xgb_)?c(\d+)$",
        r"^score_c(\d+)$",
    ]
    for name in digest_fields:
        for pat in patterns:
            m = re.match(pat, name)
            if m:
                scores.append((int(m.group(1)), name))
                break
    if not scores:
        return []
    return [name for _, name in sorted(scores, key=lambda x: x[0])]

def format_score_debug(scores):
    if not scores:
        return ""
    def sort_key(k):
        m = re.match(r"c(\d+)$", k)
        return int(m.group(1)) if m else k
    parts = [f"{k}={scores.get(k, '?')}" for k in sorted(scores.keys(), key=sort_key)]
    return f"  [Scores: {' '.join(parts)}]"

def format_votes_debug(votes, class_labels):
    """Format RF per-tree votes for display."""
    if not votes:
        return ""
    vote_strs = [f"T{i}={class_labels.get(v, f'{v}')}" for i, v in enumerate(votes)]
    return f"  [Tree Votes: {' '.join(vote_strs)}]"

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

def drain_p4_registers(p4info_helper, stub, device_id, pca_config,
                       model_type, rf_model, xgb_model, xgb_class_mapping,
                       CLASS_LABELS, n_components, pca_bits, out, pca_field_names):
    """
    Read every P4 register slot via P4Runtime and classify flows that are still
    sitting in registers (no FIN/RST seen, 120-second timeout never fired because
    no new packets arrived after tcpreplay ended).
    Appends recovered flows to `out` (already-open CSV file in append mode).
    Returns the number of flows drained.
    """
    # PCA transform parameters saved by 2_pca_generating_entries.py
    pca_components = pca_config.get('pca_components')  # list[list[float]], shape (k, F)
    pca_mean       = pca_config.get('pca_mean')        # list[float], shape (F,)
    pc_min         = pca_config.get('pc_min')          # list[float], shape (k,)
    pc_range       = pca_config.get('pc_range')        # list[float], shape (k,)
    max_val        = pca_config.get('max_val', (1 << pca_bits) - 1)

    can_classify = all(x is not None for x in [pca_components, pca_mean, pc_min, pc_range])
    if not can_classify:
        print("  NOTE: PCA components missing from pca_encoding_params.json — "
              "re-run 2_pca_generating_entries.py, then restart the controller "
              "to enable full drain classification.")

    REGS = {
        'time_first':    'MyIngress.reg_time_first_pkt',
        'time_last':     'MyIngress.reg_time_last_pkt',
        'max_iat':       'MyIngress.reg_max_iat',
        'urg_count':     'MyIngress.reg_urg_count',
        'fwd_pkt_count': 'MyIngress.reg_fwd_pkt_count',
        'bwd_pkt_count': 'MyIngress.reg_bwd_pkt_count',
        'fwd_bytes':     'MyIngress.reg_fwd_bytes',
        'bwd_bytes':     'MyIngress.reg_bwd_bytes',
        'max_win_size':  'MyIngress.reg_max_win_size',
        'flags_syn':     'MyIngress.reg_flags_syn',
        'flags_ack':     'MyIngress.reg_flags_ack',
        'flags_fin':     'MyIngress.reg_flags_fin',
        'flags_rst':     'MyIngress.reg_flags_rst',
    }

    # Read all entries of each register (no index set = read all slots)
    reg_data = {}
    for key, full_name in REGS.items():
        try:
            reg_id = p4info_helper.get_id('registers', full_name)
            req = p4runtime_pb2.ReadRequest(device_id=device_id)
            entity = req.entities.add()
            entity.register_entry.register_id = reg_id
            values = {}
            for resp in stub.Read(req):
                for ent in resp.entities:
                    if ent.HasField('register_entry'):
                        re = ent.register_entry
                        idx = re.index.index
                        val = bytes_to_int(re.data.bitstring)
                        if val != 0:
                            values[idx] = val
            reg_data[key] = values
        except Exception as e:
            print(f"  WARNING: Could not read register {full_name}: {e}")
            reg_data[key] = {}

    time_first_map = reg_data.get('time_first', {})
    fwd_count_map  = reg_data.get('fwd_pkt_count', {})
    bwd_count_map  = reg_data.get('bwd_pkt_count', {})

    active_slots = [
        slot for slot, tf in time_first_map.items()
        if tf != 0 and (fwd_count_map.get(slot, 0) + bwd_count_map.get(slot, 0)) >= 2
    ]
    if not active_slots:
        return 0

    print(f"  Found {len(active_slots)} stuck flows in P4 registers.")
    pca_width = len(str(max_val))
    drained = 0

    for slot in active_slots:
        time_first = time_first_map.get(slot, 0)
        time_last  = reg_data.get('time_last',    {}).get(slot, 0)
        duration   = time_last - time_first if time_last >= time_first else 0
        max_iat    = reg_data.get('max_iat',       {}).get(slot, 0)
        urg_count  = reg_data.get('urg_count',     {}).get(slot, 0)
        fwd        = fwd_count_map.get(slot, 0)
        bwd        = bwd_count_map.get(slot, 0)
        fwd_bytes  = reg_data.get('fwd_bytes',    {}).get(slot, 0)
        bwd_bytes  = reg_data.get('bwd_bytes',    {}).get(slot, 0)
        win        = reg_data.get('max_win_size', {}).get(slot, 0)
        f_syn      = reg_data.get('flags_syn',    {}).get(slot, 0)
        f_ack      = reg_data.get('flags_ack',    {}).get(slot, 0)
        f_fin      = reg_data.get('flags_fin',    {}).get(slot, 0)
        f_rst      = reg_data.get('flags_rst',    {}).get(slot, 0)

        class_id    = -1
        class_label = 'unclassified'
        pca_codes   = [0] * n_components

        if can_classify:
            try:
                # Guess protocol from TCP flags; fall back to TCP (6) if any flag set
                proto_guess = 6 if (f_syn or f_ack or f_fin or f_rst) else 17
                # Feature order MUST match P4_FEATURE_COLS in 2_pca_generating_entries.py:
                # Protocol, Duration, MaxIAT, UrgCount, FwdPktCount, BwdPktCount,
                # FwdBytes, BwdBytes, MaxWinSize, FlagsSyn, FlagsAck, FlagsFin, FlagsRst
                raw = np.array([[proto_guess, duration, max_iat, urg_count,
                                  fwd, bwd, fwd_bytes, bwd_bytes, win,
                                  f_syn, f_ack, f_fin, f_rst]], dtype=np.float64)
                mean_vec      = np.array(pca_mean)
                comps_mat     = np.array(pca_components)       # (k, n_features)
                pc_floats     = (raw - mean_vec) @ comps_mat.T  # (1, k)
                pc_min_arr    = np.array(pc_min)
                pc_range_arr  = np.array(pc_range)
                pc_range_safe = np.where(pc_range_arr == 0, 1, pc_range_arr)
                pc_norm       = (pc_floats - pc_min_arr) / pc_range_safe
                pc_int        = np.rint(np.clip(pc_norm * max_val, 0, max_val)).astype(int)[0]
                pca_codes     = pc_int.tolist()[:n_components]
                if len(pca_codes) < n_components:
                    pca_codes += [0] * (n_components - len(pca_codes))

                if model_type == 'rf' and rf_model is not None:
                    x_np        = np.array([pca_codes[:rf_model.n_features_in_]], dtype=np.float64)
                    pred_label  = rf_model.predict(x_np)[0]
                    class_id    = list(rf_model.classes_).index(pred_label)
                    class_label = pred_label
                elif model_type == 'xgb' and xgb_model is not None:
                    x_np        = np.array([pca_codes[:xgb_model.n_features_in_]], dtype=np.float64)
                    class_id    = int(xgb_model.predict(x_np)[0])
                    class_label = (xgb_class_mapping or {}).get(class_id, f'unknown({class_id})')
            except Exception as _e:
                class_label = 'drain-err'

        pca_display = ' '.join([f'PCA{i+1}={pca_codes[i]:<{pca_width}}' for i in range(n_components)])
        print(f"  [REG-SLOT {slot:<6}] "
              f"Dur={duration:<12} MaxIAT={max_iat:<12} Urg={urg_count:<3} "
              f"Fwd={fwd:<4} Bwd={bwd:<4} FwdB={fwd_bytes:<8} BwdB={bwd_bytes:<8} Win={win:<6} | "
              f"Flags(S/A/F/R)={f_syn}/{f_ack}/{f_fin}/{f_rst} | "
              f"{pca_display} | Class={class_label}({class_id}) [REG-DRAIN]", flush=True)
        out.write(
            f"reg-slot-{slot},0,reg-drain,0,unknown,"
            f"{duration},{max_iat},{urg_count},{fwd},{bwd},"
            f"{fwd_bytes},{bwd_bytes},{win},"
            f"{f_syn},{f_ack},{f_fin},{f_rst},"
            f"{','.join(str(c) for c in pca_codes)},"
            f"{class_id},{class_label}\n"
        )
        drained += 1

    return drained


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
    
    # Load digest schema from P4Info for dynamic parsing
    digest_fields = load_digest_schema(p4info_file_path, digest_name)
    name_to_index = build_digest_field_index(digest_fields) if digest_fields else {}
    pca_field_names = parse_pca_field_names(digest_fields)
    score_field_names = parse_score_field_names(digest_fields)

    if pca_field_names:
        n_components = len(pca_field_names)

    if digest_fields:
        print(f"Digest schema loaded: {len(digest_fields)} fields")
    else:
        print("WARNING: Digest schema not found; using fallback parsing")

    print(f"PCA Components: {n_components} (bits={pca_bits})\n")

    # Load class labels dynamically from trained model (DT, RF, or XGB)
    global CLASS_LABELS
    dt_params_path = os.path.join(script_dir, 'tables/dt_params.json')
    rf_params_path = os.path.join(script_dir, 'tables/rf_params.json')
    xgb_params_path = os.path.join(script_dir, 'tables/xgb_params.json')
    dt_model_path = os.path.join(script_dir, 'model/dt.model')
    rf_model_path = os.path.join(script_dir, 'model/rf.model')
    xgb_model_path = os.path.join(script_dir, 'model/xgb.model')

    # Infer model type from digest structure (most reliable when multiple params exist)
    model_type = None
    if digest_fields:
        if any(re.match(r"^xgb_score_c\d+$", f) for f in digest_fields):
            model_type = "xgb"
        elif "rf_votes" in digest_fields:
            model_type = "rf"
    
    # Fallback: check params files (prioritize rf > xgb > dt)
    if model_type is None:
        for params_path, mtype in [
            (rf_params_path, "rf"),
            (xgb_params_path, "xgb"),
            (dt_params_path, "dt"),
        ]:
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                    if params.get("model_type") == mtype:
                        model_type = mtype
                        break
                except Exception:
                    pass
    
    # Final default
    if model_type is None:
        model_type = "dt"

    rf_model = None
    rf_params = None
    rf_vote_bits = None
    xgb_model = None
    xgb_params = None
    xgb_class_mapping = None  # Maps encoded int to class name for XGB
    
    if model_type == "rf":
        CLASS_LABELS = load_class_labels_from_rf_params(rf_params_path, rf_model_path)
        try:
            rf_model = pd.read_pickle(rf_model_path)
            print("RF model loaded for runtime verification.")
        except Exception as e:
            print(f"WARNING: Unable to load RF model for verification: {e}")
            rf_model = None
        try:
            with open(rf_params_path, 'r') as f:
                rf_params = json.load(f)
            rf_vote_bits = rf_params.get('vote_bits')
        except Exception:
            rf_params = None
            rf_vote_bits = None
    elif model_type == "xgb":
        CLASS_LABELS = load_class_labels_from_xgb_params(xgb_params_path, xgb_model_path)
        try:
            xgb_model = pd.read_pickle(xgb_model_path)
            print("XGB model loaded for runtime verification.")
        except Exception as e:
            print(f"WARNING: Unable to load XGB model for verification: {e}")
            xgb_model = None
        try:
            with open(xgb_params_path, 'r') as f:
                xgb_params = json.load(f)
            # Build class mapping: encoded_int -> class_name
            classes = xgb_params.get("classes", [])
            xgb_class_mapping = {idx: cls for idx, cls in enumerate(classes)}
        except Exception:
            xgb_params = None
            xgb_class_mapping = None
    else:
        CLASS_LABELS = load_class_labels(dt_model_path)

    # -----------------------------------------------------------------------
    # Stale-model check: verify that the loaded model's feature count matches
    # the number of PCA components the current digest/P4 program produces.
    # A mismatch means the pipeline (2_ → 3_ → 4_ → 5_ → make) was not fully
    # rerun after adding/removing features.  Warn once here instead of crashing
    # on every single flow during runtime verification.
    # -----------------------------------------------------------------------
    _verify_model = rf_model if model_type == 'rf' else (xgb_model if model_type == 'xgb' else None)
    model_pca_mismatch = False
    if _verify_model is not None and hasattr(_verify_model, 'n_features_in_'):
        model_n_feat = _verify_model.n_features_in_
        if model_n_feat != n_components:
            model_pca_mismatch = True
            print(f"\n{'='*65}")
            print(f"WARNING: Model feature mismatch detected!")
            print(f"  Digest/P4 produces {n_components} PCA component(s).")
            print(f"  Loaded {model_type.upper()} model expects {model_n_feat} feature(s).")
            print(f"  Runtime verification will be DISABLED until the pipeline")
            print(f"  is rerun in order:")
            print(f"    1. python3 2_pca_generating_entries.py")
            print(f"    2. python3 3_{model_type}_training_model.py")
            print(f"    3. python3 4_{model_type}_generating_entries.py")
            print(f"    4. python3 5_generating_p4_code.py --model-type {model_type}")
            print(f"    5. make")
            print(f"{'='*65}\n")

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

    # Reset all flow-state registers so stale state from any previous run
    # (e.g. tcpreplay before the controller started) doesn't produce ghost flows.
    print("Resetting switch flow-state registers...")
    reset_registers = [
        "reg_time_first_pkt", "reg_time_last_pkt", "reg_max_iat",
        "reg_urg_count", "reg_fwd_pkt_count", "reg_bwd_pkt_count",
        "reg_fwd_bytes", "reg_bwd_bytes", "reg_max_win_size",
        "reg_flags_syn", "reg_flags_ack", "reg_flags_fin", "reg_flags_rst",
        "bloom_filter",
    ]
    reset_cmds = "\n".join(f"register_reset MyIngress.{r}" for r in reset_registers) + "\n"
    try:
        import subprocess as _sp, tempfile as _tf
        with _tf.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as _f:
            _f.write(reset_cmds)
            _reset_path = _f.name
        with open(_reset_path, 'r') as _fin, open(os.devnull, 'w') as _devnull:
            _p = _sp.Popen(['simple_switch_CLI', '--thrift-port', '9090'],
                           stdin=_fin, stdout=_devnull, stderr=_devnull)
            _p.wait()
        os.unlink(_reset_path)
        print("Registers reset.\n")
    except Exception as _e:
        print(f"WARNING: Could not reset registers: {_e}\n")

    # Load rules via CLI
    load_switch_cli(s1, runtime_cli_path)
    print("Rules installed. Installing digest listener...\n")

    # Install digest configuration
    install_digest(p4info_helper, s1, digest_name)

    os.makedirs('logs', exist_ok=True)
    print("Listening for traffic digests...\n")

    # Dedicated digest client
    dclient = DigestClient(address='127.0.0.1:50051', device_id=0, election_id=2)
    dclient.start()

    _digest_count_raw  = 0   # total DigestList messages received
    _digest_count_flow = 0   # flows actually printed
    _filtered_count    = 0   # flows received but filtered (< 1 packets)

    os.makedirs('logs', exist_ok=True)
    try:
        with open("logs/predictions.csv", "w") as out:
            # Build CSV header dynamically based on number of PCA components
            if pca_field_names:
                pca_headers = ','.join(pca_field_names)
            else:
                pca_headers = ','.join([f'pc{i}_code' for i in range(1, n_components + 1)])
            out.write(f"src_ip,src_port,dst_ip,dst_port,proto,duration,max_iat,urg_count,fwd_pkt_count,bwd_pkt_count,fwd_bytes,bwd_bytes,max_win_size,flags_syn,flags_ack,flags_fin,flags_rst,{pca_headers},class_id,class_label\n")
            packet_id = 0
            warn_once = {
                "missing_fields": False,
                "missing_class": False,
            }
            
            while True:
                msg = dclient.get_digest(timeout=0.5)

                if msg is None:
                    # Periodically flush expired flows
                    for entry in flow_aggregator.flush_expired():
                        # Skip flows with < 2 packets (matches training data filter)
                        if (entry["features"]["fwd_pkt_count"] + entry["features"]["bwd_pkt_count"]) < 2:
                            _filtered_count += 1
                            continue
                        
                        packet_id += 1
                        pca_width = len(str((1 << pca_bits) - 1))
                        pca_display = ' '.join([f'PCA{i+1}={entry["pca_codes"][i]:<{pca_width}}' for i in range(n_components)])
                        f = entry["features"]
                        flags = entry["flags"]
                        class_id = entry["class_id"]
                        class_label = entry["class_label"]

                        verify_note = ""
                        votes_or_scores_debug = ""
                        if model_type == "rf" and rf_model is not None and not model_pca_mismatch:
                            try:
                                x_np = np.array([entry["pca_codes"][:rf_model.n_features_in_]], dtype=np.float64)
                                if x_np.shape[1] != rf_model.n_features_in_:
                                    raise ValueError(f"X has {x_np.shape[1]} features, model expects {rf_model.n_features_in_}")
                                pred_label = rf_model.predict(x_np)[0]
                                pred_id = list(rf_model.classes_).index(pred_label)

                                # Per-tree votes and packed value
                                class_to_index = {label: idx for idx, label in enumerate(rf_model.classes_)}
                                votes = []
                                for est in rf_model.estimators_:
                                    v = est.predict(x_np)[0]
                                    if v in class_to_index:
                                        v_id = class_to_index[v]
                                    else:
                                        v_id = int(v)
                                    votes.append(v_id)

                                # Display per-tree votes
                                votes_or_scores_debug = format_votes_debug(votes, CLASS_LABELS)

                                vote_bits = rf_vote_bits
                                if vote_bits is None:
                                    n_cls = len(rf_model.classes_)
                                    vote_bits = max(1, math.ceil(math.log2(n_cls))) if n_cls > 1 else 1
                                packed = 0
                                for i, v in enumerate(votes):
                                    packed |= (v << (i * vote_bits))

                                if pred_id != class_id:
                                    verify_note = (
                                        f" | RF-VERIFY={pred_label}({pred_id})"
                                        f" PACK={packed} VOTES={votes}"
                                    )
                            except Exception as _ve:
                                verify_note = f" | RF-VERIFY=error({type(_ve).__name__}: {_ve})"
                        elif model_type == "xgb" and xgb_model is not None and not model_pca_mismatch:
                            try:
                                x_np = np.array([entry["pca_codes"][:xgb_model.n_features_in_]], dtype=np.float64)
                                pred_id = int(xgb_model.predict(x_np)[0])
                                # XGBoost returns encoded int; decode to class name using mapping
                                pred_label = xgb_class_mapping.get(pred_id, f"unknown({pred_id})")

                                if pred_id != class_id:
                                    verify_note = f" | XGB-VERIFY={pred_label}({pred_id})"
                            except Exception as _ve:
                                verify_note = f" | XGB-VERIFY=error({type(_ve).__name__}: {_ve})"

                            # Debug: show accumulated scores from dataplane (any class count)
                            votes_or_scores_debug = format_score_debug(entry.get("xgb_scores", {}))

                        print(f"[{packet_id:<4}] {f['src_ip']:>15}:{f['src_port']:<5} -> {f['dst_ip']:>15}:{f['dst_port']:<5} | "
                              f"Dur={f['duration']:<12} MaxIAT={f['max_iat']:<12} Urg={f['urg_count']:<4} "
                              f"FwdPkts={f['fwd_pkt_count']:<4} BwdPkts={f['bwd_pkt_count']:<4} "
                              f"FwdBytes={f['fwd_bytes']:<8} BwdBytes={f['bwd_bytes']:<8} Win={f['max_win_size']:<6} | "
                              f"Flags(S/A/F/R)={flags['syn']}/{flags['ack']}/{flags['fin']}/{flags['rst']} | "
                              f"{pca_display} | "
                              f"Class={class_label}({class_id}){verify_note}")
                        if votes_or_scores_debug:
                            print(votes_or_scores_debug)

                        out.write(f"{f['src_ip']},{f['src_port']},{f['dst_ip']},{f['dst_port']},{f['proto']},"
                                  f"{f['duration']},{f['max_iat']},{f['urg_count']},{f['fwd_pkt_count']},{f['bwd_pkt_count']},"
                                  f"{f['fwd_bytes']},{f['bwd_bytes']},{f['max_win_size']},{flags['syn']},{flags['ack']},{flags['fin']},{flags['rst']},"
                                  f"{','.join([str(code) for code in entry['pca_codes']])},{class_id},{class_label}\n")
                        out.flush()
                    continue

                _digest_count_raw += 1

                digest = msg.digest
                # Validate digest name
                try:
                    name = p4info_helper.get_digests_name(digest.digest_id)
                except Exception:
                    name = f"id={digest.digest_id}"
                if name != digest_name:
                    print(f"[debug] Unexpected digest name '{name}' (expected '{digest_name}'), skipping")
                    continue

                for el in digest.data:
                    st = el.struct.members

                    if digest_fields:
                        def get_idx(candidates):
                            return find_first_field(name_to_index, candidates)

                        src_ip_field = get_idx(["srcAddr", "src_ip", "src"])
                        dst_ip_field = get_idx(["dstAddr", "dst_ip", "dst"])
                        src_port_field = get_idx(["srcPort", "src_port"])
                        dst_port_field = get_idx(["dstPort", "dst_port"])
                        proto_field = get_idx(["protocol", "proto"])

                        def get_val(field_name, default=0, is_ip=False):
                            if field_name is None:
                                return default
                            idx = name_to_index.get(field_name)
                            if idx is None or idx >= len(st):
                                return default
                            return bytes_to_ip(st[idx].bitstring) if is_ip else bytes_to_int(st[idx].bitstring)

                        src_ip = get_val(src_ip_field, default="0.0.0.0", is_ip=True)
                        dst_ip = get_val(dst_ip_field, default="0.0.0.0", is_ip=True)
                        src_port = get_val(src_port_field, default=0)
                        dst_port = get_val(dst_port_field, default=0)
                        proto = get_val(proto_field, default=0)

                        duration     = get_val(get_idx(["duration"]))
                        max_iat      = get_val(get_idx(["max_iat"]))
                        urg_count    = get_val(get_idx(["urg_count"]))
                        fwd_pkt_count= get_val(get_idx(["fwd_pkt_count"]))
                        bwd_pkt_count= get_val(get_idx(["bwd_pkt_count"]))
                        fwd_bytes    = get_val(get_idx(["fwd_bytes"]))
                        bwd_bytes    = get_val(get_idx(["bwd_bytes"]))
                        max_win_size = get_val(get_idx(["max_win_size"]))
                        flags_syn = get_val(get_idx(["flags_syn", "syn"]))
                        flags_ack = get_val(get_idx(["flags_ack", "ack"]))
                        flags_fin = get_val(get_idx(["flags_fin", "fin"]))
                        flags_rst = get_val(get_idx(["flags_rst", "rst"]))

                        pca_codes = []
                        for name in (pca_field_names or []):
                            idx = name_to_index.get(name)
                            if idx is None or idx >= len(st):
                                pca_codes.append(0)
                            else:
                                pca_codes.append(bytes_to_int(st[idx].bitstring))
                        if not pca_codes:
                            pca_codes = [0] * n_components

                        xgb_scores = {}
                        for name in (score_field_names or []):
                            idx = name_to_index.get(name)
                            if idx is None or idx >= len(st):
                                continue
                            m = re.search(r"(\d+)$", name)
                            if m:
                                xgb_scores[f"c{int(m.group(1))}"] = bytes_to_int(st[idx].bitstring)

                        class_field = get_idx(["ml_result", "class_id", "class", "label", "result"])
                        class_id = get_val(class_field, default=0)
                        if class_field is None and not warn_once["missing_class"]:
                            warn_once["missing_class"] = True
                            print("WARNING: Class field not found in digest schema; defaulting class_id=0")
                    else:
                        if len(st) < 17:
                            if not warn_once["missing_fields"]:
                                warn_once["missing_fields"] = True
                                print(f"WARNING: Digest has insufficient fields ({len(st)}); skipping")
                            continue

                        src_ip = bytes_to_ip(st[0].bitstring)
                        dst_ip = bytes_to_ip(st[1].bitstring)
                        src_port = bytes_to_int(st[2].bitstring)
                        dst_port = bytes_to_int(st[3].bitstring)
                        proto = bytes_to_int(st[4].bitstring)

                        # Extract flow-based features
                        duration      = bytes_to_int(st[5].bitstring)   # Flow duration
                        max_iat       = bytes_to_int(st[6].bitstring)   # Max inter-arrival time
                        urg_count     = bytes_to_int(st[7].bitstring)   # URG count
                        fwd_pkt_count = bytes_to_int(st[8].bitstring)   # Forward packet count
                        bwd_pkt_count = bytes_to_int(st[9].bitstring)   # Backward packet count
                        fwd_bytes     = bytes_to_int(st[10].bitstring)  # Forward bytes
                        bwd_bytes     = bytes_to_int(st[11].bitstring)  # Backward bytes
                        max_win_size  = bytes_to_int(st[12].bitstring)  # Max window size
                        flags_syn = bytes_to_int(st[13].bitstring)      # SYN flag
                        flags_ack = bytes_to_int(st[14].bitstring)      # ACK flag
                        flags_fin = bytes_to_int(st[15].bitstring)      # FIN flag
                        flags_rst = bytes_to_int(st[16].bitstring)      # RST flag

                        base_fields = 17
                        remaining = len(st) - base_fields
                        if remaining <= 0:
                            if not warn_once["missing_fields"]:
                                warn_once["missing_fields"] = True
                                print(f"WARNING: Digest has no PCA/class fields ({len(st)}); skipping")
                            continue

                        # Determine layout: either [PCA... , class_id] or [PCA..., xgb_scores(4), class_id]
                        has_xgb_scores = False
                        if remaining >= 5 and (remaining - 1) not in (n_components, 9):
                            if remaining >= n_components + 5:
                                has_xgb_scores = True

                        if has_xgb_scores:
                            pca_count = min(n_components, remaining - 5)
                            scores_start = base_fields + pca_count
                            class_index = scores_start + 4
                        else:
                            pca_count = min(n_components, max(0, remaining - 1))
                            class_index = base_fields + (remaining - 1)

                        # Extract PCA component codes (pad if digest has fewer than expected)
                        pca_codes = []
                        for i in range(pca_count):
                            idx = base_fields + i
                            if idx >= len(st):
                                break
                            pca_codes.append(bytes_to_int(st[idx].bitstring))
                        if len(pca_codes) < n_components:
                            pca_codes.extend([0] * (n_components - len(pca_codes)))

                        # Extract XGB scores if present
                        xgb_scores = {}
                        if has_xgb_scores and (class_index < len(st)):
                            if scores_start + 3 < len(st):
                                xgb_scores = {
                                    'c0': bytes_to_int(st[scores_start].bitstring),
                                    'c1': bytes_to_int(st[scores_start + 1].bitstring),
                                    'c2': bytes_to_int(st[scores_start + 2].bitstring),
                                    'c3': bytes_to_int(st[scores_start + 3].bitstring),
                                }

                        # Class ID is the last field in both layouts
                        if class_index >= len(st):
                            if not warn_once["missing_fields"]:
                                warn_once["missing_fields"] = True
                                print(f"WARNING: Digest class_id index out of range ({class_index} >= {len(st)}); skipping")
                            continue
                        class_id = bytes_to_int(st[class_index].bitstring)

                    class_label = CLASS_LABELS.get(class_id, "unknown")

                    features = {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "proto": proto,
                        "duration": duration,
                        "max_iat": max_iat,
                        "urg_count": urg_count,
                        "fwd_pkt_count": fwd_pkt_count,
                        "bwd_pkt_count": bwd_pkt_count,
                        "fwd_bytes": fwd_bytes,
                        "bwd_bytes": bwd_bytes,
                        "max_win_size": max_win_size,
                    }
                    flags = {
                        "syn": flags_syn,
                        "ack": flags_ack,
                        "fin": flags_fin,
                        "rst": flags_rst,
                    }

                    for entry in flow_aggregator.update(
                        make_canonical_key(src_ip, src_port, dst_ip, dst_port, proto),
                        features,
                        pca_codes,
                        class_id,
                        class_label,
                        flags,
                        xgb_scores,
                    ):
                        # Skip flows with < 2 packets (matches training data filter)
                        if (entry["features"]["fwd_pkt_count"] + entry["features"]["bwd_pkt_count"]) < 2:
                            _filtered_count += 1
                            continue
                        packet_id += 1
                        _digest_count_flow += 1
                        pca_width = len(str((1 << pca_bits) - 1))
                        pca_display = ' '.join([f'PCA{i+1}={entry["pca_codes"][i]:<{pca_width}}' for i in range(n_components)])
                        f = entry["features"]
                        flags = entry["flags"]
                        class_id = entry["class_id"]
                        class_label = entry["class_label"]

                        verify_note = ""
                        votes_or_scores_debug = ""
                        if model_type == "rf" and rf_model is not None and not model_pca_mismatch:
                            try:
                                x_np = np.array([entry["pca_codes"][:rf_model.n_features_in_]], dtype=np.float64)
                                if x_np.shape[1] != rf_model.n_features_in_:
                                    raise ValueError(f"X has {x_np.shape[1]} features, model expects {rf_model.n_features_in_}")
                                pred_label = rf_model.predict(x_np)[0]
                                pred_id = list(rf_model.classes_).index(pred_label)

                                class_to_index = {label: idx for idx, label in enumerate(rf_model.classes_)}
                                votes = []
                                for est in rf_model.estimators_:
                                    v = est.predict(x_np)[0]
                                    if v in class_to_index:
                                        v_id = class_to_index[v]
                                    else:
                                        v_id = int(v)
                                    votes.append(v_id)

                                # Display per-tree votes
                                votes_or_scores_debug = format_votes_debug(votes, CLASS_LABELS)

                                vote_bits = rf_vote_bits
                                if vote_bits is None:
                                    n_cls = len(rf_model.classes_)
                                    vote_bits = max(1, math.ceil(math.log2(n_cls))) if n_cls > 1 else 1
                                packed = 0
                                for i, v in enumerate(votes):
                                    packed |= (v << (i * vote_bits))

                                if pred_id != class_id:
                                    verify_note = (
                                        f" | RF-VERIFY={pred_label}({pred_id})"
                                        f" PACK={packed} VOTES={votes}"
                                    )
                            except Exception as _ve:
                                verify_note = f" | RF-VERIFY=error({type(_ve).__name__}: {_ve})"
                        elif model_type == "xgb" and xgb_model is not None and not model_pca_mismatch:
                            try:
                                x_np = np.array([entry["pca_codes"][:xgb_model.n_features_in_]], dtype=np.float64)
                                pred_id = int(xgb_model.predict(x_np)[0])
                                # XGBoost returns encoded int; decode to class name using mapping
                                pred_label = xgb_class_mapping.get(pred_id, f"unknown({pred_id})")

                                if pred_id != class_id:
                                    verify_note = f" | XGB-VERIFY={pred_label}({pred_id})"
                            except Exception as _ve:
                                verify_note = f" | XGB-VERIFY=error({type(_ve).__name__}: {_ve})"

                            # Debug: show accumulated scores from dataplane (any class count)
                            votes_or_scores_debug = format_score_debug(entry.get("xgb_scores", {}))

                        print(f"[{packet_id:<4}] {f['src_ip']:>15}:{f['src_port']:<5} -> {f['dst_ip']:>15}:{f['dst_port']:<5} | "
                              f"Dur={f['duration']:<12} MaxIAT={f['max_iat']:<12} Urg={f['urg_count']:<4} "
                              f"FwdPkts={f['fwd_pkt_count']:<4} BwdPkts={f['bwd_pkt_count']:<4} "
                              f"FwdBytes={f['fwd_bytes']:<8} BwdBytes={f['bwd_bytes']:<8} Win={f['max_win_size']:<6} | "
                              f"Flags(S/A/F/R)={flags['syn']}/{flags['ack']}/{flags['fin']}/{flags['rst']} | "
                              f"{pca_display} | "
                              f"Class={class_label}({class_id}){verify_note}", flush=True)
                        if votes_or_scores_debug:
                            print(votes_or_scores_debug, flush=True)

                        out.write(f"{f['src_ip']},{f['src_port']},{f['dst_ip']},{f['dst_port']},{f['proto']},"
                                  f"{f['duration']},{f['max_iat']},{f['urg_count']},{f['fwd_pkt_count']},{f['bwd_pkt_count']},"
                                  f"{f['fwd_bytes']},{f['bwd_bytes']},{f['max_win_size']},{flags['syn']},{flags['ack']},{flags['fin']},{flags['rst']},"
                                  f"{','.join([str(code) for code in entry['pca_codes']])},{class_id},{class_label}\n")
                        out.flush()

        print("\nShutting down...")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        # Flush any flows still in memory that never got a FIN/RST or timed out.
        # Reopen in append mode — the `with open()` above has already closed the file.
        remaining = flow_aggregator.flush_all()
        with open("logs/predictions.csv", "a") as final_out:
            if remaining:
                print(f"Flushing {len(remaining)} remaining in-memory flows...")
                for entry in remaining:
                    if (entry["features"]["fwd_pkt_count"] + entry["features"]["bwd_pkt_count"]) < 2:
                        continue
                    packet_id += 1
                    f = entry["features"]
                    flags = entry["flags"]
                    class_id = entry["class_id"]
                    class_label = entry["class_label"]
                    pca_width = len(str((1 << pca_bits) - 1))
                    pca_display = ' '.join([f'PCA{i+1}={entry["pca_codes"][i]:<{pca_width}}' for i in range(n_components)])
                    print(f"[{packet_id:<4}] {f['src_ip']:>15}:{f['src_port']:<5} -> "
                          f"{f['dst_ip']:>15}:{f['dst_port']:<5} | "
                          f"Dur={f['duration']:<12} MaxIAT={f['max_iat']:<12} Urg={f['urg_count']:<4} "
                          f"FwdPkts={f['fwd_pkt_count']:<4} BwdPkts={f['bwd_pkt_count']:<4} "
                          f"FwdBytes={f['fwd_bytes']:<8} BwdBytes={f['bwd_bytes']:<8} Win={f['max_win_size']:<6} | "
                          f"Flags(S/A/F/R)={flags['syn']}/{flags['ack']}/{flags['fin']}/{flags['rst']} | "
                          f"{pca_display} | Class={class_label}({class_id}) [FINAL-FLUSH]")
                    final_out.write(
                        f"{f['src_ip']},{f['src_port']},{f['dst_ip']},{f['dst_port']},{f['proto']},"
                        f"{f['duration']},{f['max_iat']},{f['urg_count']},{f['fwd_pkt_count']},{f['bwd_pkt_count']},"
                        f"{f['fwd_bytes']},{f['bwd_bytes']},{f['max_win_size']},"
                        f"{flags['syn']},{flags['ack']},{flags['fin']},{flags['rst']},"
                        f"{','.join([str(code) for code in entry['pca_codes']])},"
                        f"{class_id},{class_label}\n"
                    )
                print(f"In-memory flush complete. Total flows so far: {packet_id}")

            # Drain any flows still in P4 registers that never got a FIN/RST or
            # 120-second timeout (the timeout only fires when a new packet arrives
            # at the same register slot — after tcpreplay ends, nothing does).
            print("\nDraining stuck flows from P4 registers...")
            try:
                _rf_m   = rf_model   if model_type == 'rf'  else None
                _xgb_m  = xgb_model  if model_type == 'xgb' else None
                _xgb_cm = xgb_class_mapping if model_type == 'xgb' else None
                drained = drain_p4_registers(
                    p4info_helper, s1.client_stub, s1.device_id,
                    pca_config, model_type, _rf_m, _xgb_m, _xgb_cm,
                    CLASS_LABELS, n_components, pca_bits, final_out, pca_field_names
                )
                if drained:
                    packet_id += drained
                    print(f"Register drain complete. Total flows recorded: {packet_id}")
                else:
                    print("Register drain: no stuck flows found.")
            except Exception as _de:
                print(f"WARNING: Register drain failed: {_de}")
        dclient.stop()
        ShutdownAllSwitchConnections()

if __name__ == '__main__':
    # Default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer build/basic.p4.p4info.txtpb — always freshly written by `make`.
    # Fall back to basic.p4info (project root) only if the build file is absent.
    _build_p4info = os.path.join(script_dir, '../build/basic.p4.p4info.txtpb')
    _root_p4info  = os.path.join(script_dir, '../basic.p4info')
    default_p4info = _build_p4info if os.path.exists(_build_p4info) else _root_p4info
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