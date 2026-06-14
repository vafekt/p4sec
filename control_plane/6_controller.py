#!/usr/bin/env python3
"""
P4 Runtime Controller for ML Traffic Classification.

Supports the three reduction methods (PCA / LDA / Autoencoder) and the two
deployable classifier back-ends (DT / RF).

This controller:
1. Loads P4 program rules into a BMv2 switch via simple_switch_CLI
2. Listens for digest messages containing flow features and classifications
3. Records predictions to CSV for analysis
"""

import argparse
import sys
import os
import time
import grpc
import subprocess
import json
import pandas as pd
import numpy as np
import math
import re
import ipaddress
import warnings
from time import sleep
from queue import Queue, Empty
from threading import Thread

# Suppress sklearn warning when predicting with a numpy array on a model
# fitted with a DataFrame (feature names mismatch is cosmetic, not a bug).
warnings.filterwarnings('ignore', message='.*does not have valid feature names.*',
                        category=UserWarning)
from google.protobuf import text_format
from p4.config.v1 import p4info_pb2

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                  '../../../utils/')))
import p4runtime_lib.bmv2
import p4runtime_lib.helper
from p4runtime_lib.switch import ShutdownAllSwitchConnections
from p4.v1 import p4runtime_pb2, p4runtime_pb2_grpc

# gRPC keepalive so a long-idle digest StreamChannel (e.g. while waiting out the
# 20s flow idle-timeout before a drain) is not dropped by the switch's grpc
# server as an idle HTTP/2 connection. Without this the digest stream returns
# UNAVAILABLE / "Socket closed" under the Mininet-spawned switch, which delivers
# zero digests. Interval is kept >= the grpc server's default min ping interval
# to avoid a "too_many_pings" GOAWAY.
GRPC_KEEPALIVE_OPTS = [
    ('grpc.keepalive_time_ms', 600000),                      # ping every 10min (BMv2 server min is 5min)
    ('grpc.keepalive_timeout_ms', 20000),                    # wait 20s for ack
    ('grpc.keepalive_permit_without_calls', 1),              # ping on an idle stream
    ('grpc.http2.max_pings_without_data', 0),                # no cap on data-less pings
    ('grpc.http2.min_time_between_pings_ms', 600000),
    ('grpc.http2.min_ping_interval_without_data_ms', 600000),
    ('grpc.max_connection_idle_ms', 2147483647),             # never treat as idle
    ('grpc.max_connection_age_ms', 2147483647),
]

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

# Traffic class label mapping will be loaded dynamically from model
CLASS_LABELS = {}

def make_canonical_key(src_ip, src_port, dst_ip, dst_port, proto):
    """
    Return a direction-normalised 5-tuple key so that both A->B and B->A
    digest messages aggregate into the same flow entry.

    Uses INTEGER IP comparison — matches P4 (bit<32> comparison) and
    1_extract_dataset.py (ipaddress.ip_address integer conversion) exactly.
    String comparison (e.g. '192.168.1.100' < '192.168.1.52') disagrees with
    integer comparison and would split a bidirectional flow into two entries.
    """
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
    """Pass-through: every digest from the P4 switch is a complete,
    already-classified flow (timeout / FIN-RST / scan-drain handled in
    the data plane).  No controller-side aggregation or timeout needed."""

    @staticmethod
    def update(key, features, pca_codes, class_id, class_label, flags, rf_votes=None):
        return [{
            "features": features,
            "pca_codes": pca_codes,
            "class_id": class_id,
            "class_label": class_label,
            "flags": flags,
            "rf_votes": rf_votes or [],
        }]

def _run_cli_chunk(lines, thrift_port, log_fh):
    """Run simple_switch_CLI with the given lines written to a temp file.

    Returns (returncode, elapsed_seconds).
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        tf.writelines(lines)
        tmp_path = tf.name
    try:
        t0 = time.time()
        with open(tmp_path, 'r') as fin:
            proc = subprocess.Popen(
                ['simple_switch_CLI', '--thrift-port', str(thrift_port)],
                stdin=fin, stdout=log_fh, stderr=subprocess.STDOUT
            )
            proc.wait()
        return proc.returncode, time.time() - t0
    finally:
        os.unlink(tmp_path)


def load_switch_cli(sw, runtime_cli, thrift_port=9090, chunk_size=50000):
    """Load P4 table rules via simple_switch_CLI.

    Blocks until all entries are installed before the controller starts
    listening for digests.  ml_code (classifier) entries are placed at the
    top of s1-commands.txt by step 4, so the classifier is ready as soon as
    those first lines load — no separate pre-load file is needed.

    Large tables (e.g. rf_vote_classify with 1.68 M entries) are loaded in
    chunks of `chunk_size` entries each, each with a fresh Thrift connection,
    to avoid BMv2 dropping the connection mid-load.

    Returns (table_counts, load_time_seconds) where table_counts is a dict
    mapping table name → entry count (plus a '__total__' key).
    """
    print(f"Loading P4 rules from {runtime_cli}...")
    table_counts = {}
    all_lines = []
    try:
        with open(runtime_cli, 'r') as f:
            all_lines = f.readlines()
        for line in all_lines:
            stripped = line.lstrip()
            if stripped.startswith('table_add'):
                parts = stripped.split()
                if len(parts) >= 2:
                    tbl = parts[1]
                    table_counts[tbl] = table_counts.get(tbl, 0) + 1
    except Exception:
        pass
    n_entries = sum(table_counts.values())
    table_counts['__total__'] = n_entries
    if n_entries:
        print(f"  ({n_entries:,} table entries to install — timing will be reported on completion)")

    os.makedirs('logs', exist_ok=True)

    # Partition lines: everything that is NOT a large-table entry goes into
    # base_lines; the large exact-match RF vote table is separated and loaded
    # in chunks.
    LARGE_TABLE_KEYWORDS = ('rf_vote_classify',)
    base_lines = []
    large_lines = []
    for line in all_lines:
        stripped = line.lstrip()
        if stripped.startswith('table_add') and any(kw in stripped for kw in LARGE_TABLE_KEYWORDS):
            large_lines.append(line)
        else:
            base_lines.append(line)

    load_time = 0.0
    try:
        with open('logs/cli_output.log', 'w') as fout:
            t_start = time.time()

            # --- Pass 1: base rules (pca, ml_code, rf_tree, etc.) in chunks ---
            # BMv2's Thrift connection drops when too many commands are sent
            # in a single session, silently losing entries.  Chunk all rules.
            all_to_load = base_lines + large_lines
            n_chunks = math.ceil(len(all_to_load) / chunk_size)
            print(f"  Loading {len(all_to_load):,} lines in {n_chunks} chunk(s) of {chunk_size:,}...")
            for chunk_idx in range(n_chunks):
                chunk = all_to_load[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
                rc, dt = _run_cli_chunk(chunk, thrift_port, fout)
                load_time += dt
                if rc not in (0, 1):
                    print(f"  WARNING: chunk {chunk_idx+1}/{n_chunks} exited with code {rc}")
                else:
                    print(f"  Chunk {chunk_idx+1}/{n_chunks} ({len(chunk):,} entries): {dt:.1f}s")

            load_time = time.time() - t_start  # use wall-clock total

        errors = 0
        table_full = 0
        with open('logs/cli_output.log', 'r') as log:
            for line in log:
                if line.startswith('RuntimeCmd: Error'):
                    errors += 1
                if 'TABLE_FULL' in line or 'table is full' in line.lower():
                    table_full += 1
        if table_full:
            print(f"ERROR: {table_full} TABLE_FULL rejections — increase NB_ENTRIES in basic.p4 and recompile!")
        if errors:
            print(f"WARNING: {errors} CLI errors during rule loading (check logs/cli_output.log)")
        if not errors and not table_full:
            print(f"Rules loaded successfully in {load_time:.2f}s. CLI output logged to logs/cli_output.log")

        # Verify actual entry counts match expected counts
        added = 0
        with open('logs/cli_output.log', 'r') as log:
            for line in log:
                if 'Entry has been added' in line:
                    added += 1
        if added < n_entries:
            print(f"WARNING: only {added:,}/{n_entries:,} entries confirmed added — "
                  f"{n_entries - added:,} may have been silently dropped!")
        else:
            print(f"  Verified: all {added:,} entries confirmed in switch.")
    except FileNotFoundError as e:
        print(f"ERROR: Runtime CLI file not found: {e}")
        raise

    return table_counts, load_time

def build_digest_entry(p4info_helper, digest_name):
    """Build a DigestEntry configuration for the switch."""
    de = p4runtime_pb2.DigestEntry()
    de.digest_id = p4info_helper.get_digests_id(digest_name)
    de.config.max_timeout_ns = 0
    de.config.max_list_size = 1
    de.config.ack_timeout_ns = 1
    return de

def install_digest(p4info_helper, sw, digest_name, election_id=2):
    """Install digest configuration on the switch.

    BMv2's P4Runtime server always returns UNKNOWN as the top-level gRPC
    status for any write error (actual code is in Status.details).  The most
    common case is ALREADY_EXISTS when the digest was left configured from a
    previous controller run.  We handle this by retrying with MODIFY.
    """
    de = build_digest_entry(p4info_helper, digest_name)

    def _write(update_type):
        req = p4runtime_pb2.WriteRequest()
        req.device_id = sw.device_id
        req.election_id.low = election_id
        u = req.updates.add()
        u.type = update_type
        u.entity.digest_entry.CopyFrom(de)
        sw.client_stub.Write(req)

    # Try INSERT first; if that fails (e.g. ALREADY_EXISTS from a previous
    # session), fall back to MODIFY which overwrites the existing config.
    try:
        _write(p4runtime_pb2.Update.INSERT)
        print(f"Digest '{digest_name}' installed successfully.")
        return
    except grpc.RpcError:
        pass

    try:
        _write(p4runtime_pb2.Update.MODIFY)
        print(f"Digest '{digest_name}' updated (MODIFY) successfully.")
        return
    except grpc.RpcError as e:
        print(f"WARNING: Digest installation failed (INSERT+MODIFY both failed): "
              f"{e.code().name} - {e.details()}")
        raise

def bytes_to_int(bb):
    """Convert byte array to integer (big-endian)."""
    v = 0
    for b in bb:
        v = (v << 8) + int(b)
    return v

def bytes_to_ip(bb):
    """Convert byte array to IPv4 address string.
    Handles P4Runtime bitstrings which may omit leading zero-bytes."""
    v = 0
    for b in bb:
        v = (v << 8) + int(b)
    return f"{(v>>24)&0xff}.{(v>>16)&0xff}.{(v>>8)&0xff}.{v&0xff}"

def bytes_to_mac(bb):
    """Convert byte array to MAC address integer (48-bit big-endian).
    Returns the integer representation, same as bytes_to_int."""
    v = 0
    for b in bb:
        v = (v << 8) + int(b)
    return v

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

def load_class_labels_from_model_params(params_path, model_path):
    """Load class labels from universal model_params.json, fallback to model if needed."""
    try:
        with open(params_path, 'r') as f:
            params = json.load(f)
        classes = params.get("classes")
        if classes:
            label_mapping = {idx: label for idx, label in enumerate(classes)}
            mt = params.get("model_type", "unknown")
            print(f"=== Class Label Mapping (from model_params.json, type={mt}) ===")
            for class_id, label in sorted(label_mapping.items()):
                print(f"  {class_id}: {label}")
            print()
            return label_mapping
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"WARNING: Error reading model params from {params_path}: {e}")

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

def detect_model_type_from_p4info(p4info_path):
    """Detect model type by inspecting compiled P4 table names in the p4info.

    Returns 'rf' or None (meaning DT/unknown).
    This is the authoritative check because rf_votes is INTERNAL metadata
    that is never placed in digest_t, so digest-field inspection alone cannot
    distinguish RF from DT.
    """
    try:
        p4info = p4info_pb2.P4Info()
        with open(p4info_path, 'r') as f:
            text_format.Merge(f.read(), p4info)
        table_aliases = {t.preamble.alias for t in p4info.tables}
        table_names   = {t.preamble.name  for t in p4info.tables}
        all_names = table_aliases | table_names
        if any('rf_vote' in n for n in all_names):
            return 'rf'
        return None
    except Exception:
        return None

def build_digest_field_index(digest_fields):
    return {name: idx for idx, name in enumerate(digest_fields)}

def find_first_field(name_to_index, candidates):
    for name in candidates:
        if name in name_to_index:
            return name
    return None

def parse_transform_field_names(digest_fields):
    """Return code field names sorted by numeric suffix.

    Detects PCA (pc1_code, pc2_code, ...), LDA (ld1_code, ld2_code, ...),
    and Autoencoder (ae1_code, ae2_code, ...).
    """
    codes = []
    for name in digest_fields:
        m = re.match(r"^(pc|ld|ae)(\d+)_code$", name, re.IGNORECASE)
        if m:
            codes.append((int(m.group(2)), name))
    if not codes:
        return []
    return [name for _, name in sorted(codes, key=lambda x: x[0])]

def format_votes_debug(votes, class_labels):
    """Format RF per-tree votes for display."""
    if not votes:
        return ""
    vote_strs = [f"T{i}={class_labels.get(v, f'{v}')}" for i, v in enumerate(votes)]
    return f"  [Tree Votes: {' '.join(vote_strs)}]"

def format_codes_display(codes, display_names, bits=16):
    """Universal display formatter for transform codes or empty for FS."""
    if not codes or not display_names:
        return ""
    width = len(str((1 << bits) - 1)) if bits else 8
    parts = [f'{display_names[i]}={codes[i]:<{width}}' for i in range(min(len(codes), len(display_names)))]
    return ' '.join(parts)

def format_codes_csv(codes):
    """Format codes for CSV output. Returns empty string if no codes."""
    if not codes:
        return ""
    return ','.join(str(c) for c in codes) + ','


# Map canonical feature names (paper Table 2) to entry["features"] / entry["flags"] keys
_FEATURE_TO_ENTRY_KEY = {
    "SrcIP":            ("features", "src_ip"),
    "DstIP":            ("features", "dst_ip"),
    "SrcMAC":           ("features", "src_mac"),
    "DstMAC":           ("features", "dst_mac"),
    "Protocol":         ("features", "proto"),
    "SrcPort":          ("features", "src_port"),
    "DstPort":          ("features", "dst_port"),
    "Duration":         ("features", "duration"),
    "MaxIAT":           ("features", "max_iat"),
    "FwdPktCount":      ("features", "fwd_pkt_count"),
    "BwdPktCount":      ("features", "bwd_pkt_count"),
    "FwdBytes":         ("features", "fwd_bytes"),
    "BwdBytes":         ("features", "bwd_bytes"),
    "FwdMaxPktLen":     ("features", "fwd_max_pkt_len"),
    "BwdMaxPktLen":     ("features", "bwd_max_pkt_len"),
    "FlagsSyn":         ("flags", "syn"),
    "FlagsAck":         ("flags", "ack"),
    "FlagsFin":         ("flags", "fin"),
    "FlagsRst":         ("flags", "rst"),
    "FlagsPsh":         ("features", "flags_psh"),
    "MaxWinSize":       ("features", "max_win_size"),
    "InitFwdWinBytes":  ("features", "init_fwd_win"),
}


def build_model_input(entry, needs_transform, reduction_feature_columns):
    """
    Build the correct numpy model input from a flow entry.

    - PCA / LDA / Autoencoder / UMAP (needs_transform=True): returns entry["pca_codes"] (the quantized codes).
    - Feature Selection (needs_transform=False): returns the selected raw features
      extracted from entry["features"] + entry["flags"] in the correct order.

    Returns a list of numeric values ready for np.array([...]).
    """
    if needs_transform:
        return list(entry.get("pca_codes", []))

    # Feature Selection: build input from raw features
    if not reduction_feature_columns:
        return list(entry.get("pca_codes", []))

    values = []
    for feat_name in reduction_feature_columns:
        mapping = _FEATURE_TO_ENTRY_KEY.get(feat_name)
        if mapping:
            section, key = mapping
            values.append(entry.get(section, {}).get(key, 0))
        else:
            # Unknown feature name — try lowercase in features dict
            values.append(entry.get("features", {}).get(feat_name.lower(), 0))
    return values


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
        self.channel = grpc.insecure_channel(self.address, options=GRPC_KEEPALIVE_OPTS)
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

    # Load reduction / transform configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pca_config_path = os.path.join(script_dir, 'tables/encoding_params.json')
    reduction_config_path = os.path.join(script_dir, 'tables/reduction_config.json')
    flow_aggregator = FlowAggregator()

    # Load universal reduction config (determines method: pca/lda/autoencoder/umap/feature_selection)
    reduction_method = 'pca'   # default fallback
    needs_transform = True
    reduction_feature_columns = None
    try:
        with open(reduction_config_path, 'r') as f:
            reduction_cfg = json.load(f)
            reduction_method = reduction_cfg.get('method', 'pca')
            needs_transform = reduction_cfg.get('needs_transform_tables', True)
            reduction_feature_columns = reduction_cfg.get('feature_columns')
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Determine display prefix: PCA / LD / AE / UM / raw feature names
    if reduction_method == 'lda':
        code_display_prefix = 'LD'
    elif reduction_method == 'autoencoder':
        code_display_prefix = 'AE'
    elif reduction_method == 'umap':
        code_display_prefix = 'UM'
    elif reduction_method == 'feature_selection':
        code_display_prefix = ''   # will use actual feature names
    else:
        code_display_prefix = 'PCA'

    pca_config = {}
    # Try canonical encoding_params.json first; fall back to legacy pca_encoding_params.json
    _enc_candidates = [
        pca_config_path,
        os.path.join(script_dir, 'tables/pca_encoding_params.json'),
    ]
    _enc_loaded = False
    for _enc_path in _enc_candidates:
        if not os.path.exists(_enc_path):
            continue
        try:
            with open(_enc_path, 'r') as f:
                pca_config = json.load(f)
            # Normalise legacy key names: pca_* → transform_* so all methods share same keys
            for _old, _new in [('pca_components', 'transform_components'),
                                ('pca_mean',       'transform_mean'),
                                ('pc_min',         'transform_min'),
                                ('pc_max',         'transform_max'),
                                ('pc_range',       'transform_range')]:
                if _old in pca_config and _new not in pca_config:
                    pca_config[_new] = pca_config[_old]
            n_components = pca_config.get('n_components', 2)
            pca_bits = pca_config.get('bits', 16) or 16
            _enc_loaded = True
            print(f"Loaded encoding config: {_enc_path}")
            break
        except json.JSONDecodeError:
            print(f"WARNING: Invalid JSON in {_enc_path}, skipping")
    if not _enc_loaded:
        print(f"WARNING: No encoding config found, defaulting to 2 components")
        n_components = 2
        pca_bits = 16

    # Load digest schema from P4Info for dynamic parsing
    digest_fields = load_digest_schema(p4info_file_path, digest_name)
    name_to_index = build_digest_field_index(digest_fields) if digest_fields else {}
    transform_field_names = parse_transform_field_names(digest_fields)

    if transform_field_names:
        n_components = len(transform_field_names)

    # Build universal display names for the code/feature columns.
    # PCA: ["PCA1", ...], LDA: ["LD1", ...], AE: ["AE1", ...], FS: []
    if transform_field_names:
        # Transform codes are in digest
        code_csv_headers = transform_field_names
        code_display_names = []
        for name in transform_field_names:
            m = re.match(r"^(pc|ld|ae|um)(\d+)_code$", name, re.IGNORECASE)
            if m:
                pfx = {'pc': 'PCA', 'ld': 'LD', 'ae': 'AE', 'um': 'UM'}.get(m.group(1).lower(), m.group(1).upper())
                code_display_names.append(f"{pfx}{m.group(2)}")
            else:
                code_display_names.append(name)
    elif reduction_feature_columns and not needs_transform:
        # Feature Selection — no codes in digest, classifier ran on raw features
        code_csv_headers = []   # no extra columns in digest
        code_display_names = [] # nothing to display beyond raw features
        n_components = 0        # no transform components
    else:
        # Fallback: naming based on reduction method
        prefix_map = {'pca': 'pc', 'lda': 'ld', 'autoencoder': 'ae', 'umap': 'um'}
        code_prefix = prefix_map.get(reduction_method, 'pc')
        code_csv_headers = [f'{code_prefix}{i}_code' for i in range(1, n_components + 1)]
        code_display_names = [f'{code_display_prefix or "PCA"}{i}' for i in range(1, n_components + 1)]

    if digest_fields:
        print(f"Digest schema loaded: {len(digest_fields)} fields")
    else:
        print("WARNING: Digest schema not found; using fallback parsing")

    print(f"Reduction method: {reduction_method.upper()} | Components: {n_components} (bits={pca_bits})\n")

    # Load class labels dynamically from trained model
    global CLASS_LABELS
    model_params_path = os.path.join(script_dir, 'tables/model_params.json')

    # Infer model type from the compiled P4 program (p4info tables) — authoritative.
    # rf_votes is INTERNAL metadata in P4 and is never placed in digest_t,
    # so we inspect table names: rf_vote_classify → RF, otherwise DT.
    p4info_model_type = detect_model_type_from_p4info(p4info_file_path)

    # Read universal model_params.json to get model_type field
    _mp = {}
    if os.path.exists(model_params_path):
        try:
            with open(model_params_path, 'r') as f:
                _mp = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not read {model_params_path}: {e}")

    # Resolve to dt or rf
    pmt = _mp.get("model_type", "")
    if p4info_model_type == "rf" or pmt == "rf":
        model_type = "rf"
    else:
        model_type = "dt"

    rf_model = None
    rf_params = None
    rf_vote_bits = None

    # Determine model path
    _model_path = os.path.join(script_dir, f'model/{model_type}.model')

    # Load class labels from universal params
    CLASS_LABELS = load_class_labels_from_model_params(model_params_path, _model_path)

    if model_type == "rf":
        try:
            rf_model = pd.read_pickle(_model_path)
            print("RF model loaded for runtime verification.")
        except Exception as e:
            print(f"WARNING: Unable to load RF model for verification: {e}")
            rf_model = None
        rf_params = _mp if _mp.get("model_type") == "rf" else None
        rf_vote_bits = rf_params.get('vote_bits') if rf_params else None

    # -----------------------------------------------------------------------
    # Stale-model check: verify that the loaded model's feature count matches
    # the number of PCA components the current digest/P4 program produces.
    # A mismatch means the pipeline (2_ → 3_ → 4_ → 5_ → make) was not fully
    # rerun after adding/removing features.  Warn once here instead of crashing
    # on every single flow during runtime verification.
    # -----------------------------------------------------------------------
    _verify_model = rf_model if model_type == 'rf' else None
    model_pca_mismatch = False
    if _verify_model is not None and hasattr(_verify_model, 'n_features_in_'):
        model_n_feat = _verify_model.n_features_in_
        # For transform methods: model features should match n_components
        # For FS: model features should match len(reduction_feature_columns)
        expected_feat = n_components if needs_transform else len(reduction_feature_columns or [])
        if expected_feat > 0 and model_n_feat != expected_feat:
            model_pca_mismatch = True
            print(f"\n{'='*65}")
            print(f"WARNING: Model feature mismatch detected!")
            print(f"  Pipeline produces {expected_feat} feature(s) ({reduction_method.upper()}).")
            print(f"  Loaded {model_type.upper()} model expects {model_n_feat} feature(s).")
            print(f"  Runtime verification will be DISABLED until the pipeline")
            print(f"  is rerun in order (step 2 → step 3 → step 4 → step 5 → make).")
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
    # Force mastership with election_id=2 to beat any stale run_exercise.py
    # connection (which uses election_id=1).  run_exercise.py keeps the gRPC
    # channel alive (HTTP/2 connection pooling) even after closing the P4Runtime
    # stream, so the switch may still hold election_id=1 mastership for it.
    _force_arb = p4runtime_pb2.StreamMessageRequest()
    _force_arb.arbitration.device_id = s1.device_id
    _force_arb.arbitration.election_id.low = 2
    s1.requests_stream.put(_force_arb)
    s1.dispatcher.arbitration_queue.get()  # wait for ack

    # Reset all flow-state registers so stale state from any previous run
    # (e.g. tcpreplay before the controller started) doesn't produce ghost flows.
    print("Resetting switch flow-state registers...")
    reset_registers = [
        "reg_time_first_pkt", "reg_time_last_pkt", "reg_max_iat",
        "reg_fwd_pkt_count", "reg_bwd_pkt_count",
        "reg_fwd_bytes", "reg_bwd_bytes", "reg_max_win_size",
        "reg_flags_syn", "reg_flags_ack", "reg_flags_fin", "reg_flags_rst",
        "reg_fwd_max_pkt_len", "reg_bwd_max_pkt_len",
        "reg_flags_psh", "reg_init_fwd_win",
        "reg_flow_hash_2",
        "reg_canon_src_port",
        "reg_canon_dst_port",
        "reg_canon_src_ip",
        "reg_canon_dst_ip",
        "reg_canon_src_mac",
        "reg_canon_dst_mac",
        "reg_protocol",
        "bloom_filter_1",
        "bloom_filter_2",
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
    table_counts, rules_load_time = load_switch_cli(s1, runtime_cli_path)
    print("Rules installed. Installing digest listener...\n")

    # Write run metadata. Classify each table as "transform" or "classifier".
    # Transform tables: pca_component* (PCA), lda_component* (LDA),
    #                   ae_component* (Autoencoder)
    # Classifier tables: ml_code (DT),
    #                    rf_tree_* / rf_vote_classify (RF)
    _TRANSFORM_RE  = re.compile(r'(pca|lda|ae)_component\d+$')
    _CLASSIFIER_RE = re.compile(r'ml_code|rf_tree_|rf_vote_classify')

    _transform_per_component = {}  # table_name -> count (one entry per component table)
    _n_model_entries = 0
    for _tbl, _cnt in table_counts.items():
        if _tbl == '__total__':
            continue
        if _TRANSFORM_RE.search(_tbl):
            _transform_per_component[_tbl] = _cnt
        elif _CLASSIFIER_RE.search(_tbl):
            _n_model_entries += _cnt

    # All component tables should have the same count; record a single per-component value
    _pca_entries_per_component = next(iter(_transform_per_component.values()), 0)
    _pca_total_entries = sum(_transform_per_component.values())

    # Determine on-disk size of the active model
    try:
        _model_size_bytes = os.path.getsize(_model_path)
    except OSError:
        _model_size_bytes = 0

    # Transform config size (encoding_params.json)
    try:
        _transform_config_size_bytes = os.path.getsize(pca_config_path)
    except OSError:
        _transform_config_size_bytes = 0

    _run_metadata = {
        "reduction_method":              reduction_method,
        "n_components":                  n_components,
        "pca_bits":                      pca_bits,
        "transform_entries_per_component": _pca_entries_per_component,
        "transform_total_entries":       _pca_total_entries,
        "model_type":                    model_type,
        "model_entries":                 _n_model_entries,
        "model_file":                    os.path.basename(_model_path),
        "model_size_bytes":              _model_size_bytes,
        "transform_config_size_bytes":   _transform_config_size_bytes,
        "total_memory_bytes":            _model_size_bytes + _transform_config_size_bytes,
        "total_table_entries":           table_counts.get('__total__', 0),
        "rules_load_time_s":             round(rules_load_time, 3),
        "table_counts":                  {k: v for k, v in table_counts.items() if k != '__total__'},
        "runtime_cli":                   runtime_cli_path,
    }
    os.makedirs('logs', exist_ok=True)
    _metadata_path = os.path.join('logs', 'run_metadata.json')
    with open(_metadata_path, 'w') as _mf:
        json.dump(_run_metadata, _mf, indent=2)
    print(f"Run metadata written to {_metadata_path}\n")

    # Dedicated digest client first — make it primary BEFORE installing
    # the DigestEntry so BMv2 routes digests to this stream.
    dclient = DigestClient(address='127.0.0.1:50051', device_id=0, election_id=3)
    dclient.start()
    time.sleep(1.0)  # let arbitration settle

    # Install digest configuration under the DigestClient's election_id
    install_digest(p4info_helper, s1, digest_name, election_id=3)

    os.makedirs('logs', exist_ok=True)
    print("Listening for traffic digests...\n")

    _digest_count_raw  = 0   # total DigestList messages received
    _digest_count_flow = 0   # flows actually printed
    _filtered_count    = 0   # flows received but filtered (< 1 packets)

    os.makedirs('logs', exist_ok=True)

    try:
        with open("logs/predictions.csv", "w") as out:
            # Build CSV header dynamically based on reduction method
            if code_csv_headers:
                extra_headers = ','.join(code_csv_headers) + ','
            else:
                extra_headers = ''   # Feature Selection: no code columns
            out.write(f"src_ip,dst_ip,src_mac,dst_mac,src_port,dst_port,proto,duration,max_iat,fwd_pkt_count,bwd_pkt_count,fwd_bytes,bwd_bytes,fwd_max_pkt_len,bwd_max_pkt_len,flags_syn,flags_ack,flags_fin,flags_rst,flags_psh,max_win_size,init_fwd_win,{extra_headers}class_id,class_label\n")
            packet_id = 0
            # IPs used by run_demo.sh drain-trigger probes — filter them out
            _DRAIN_IPS = {int(ipaddress.ip_address("10.255.255.254")),
                          int(ipaddress.ip_address("10.255.255.253"))}
            warn_once = {
                "missing_fields": False,
                "missing_class": False,
            }

            while True:
                msg = dclient.get_digest(timeout=0.5)

                if msg is None:
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
                        src_mac_field = get_idx(["srcMAC", "canon_src_mac", "src_mac"])
                        dst_mac_field = get_idx(["dstMAC", "canon_dst_mac", "dst_mac"])
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

                        src_ip_str = get_val(src_ip_field, default="0.0.0.0", is_ip=True)
                        dst_ip_str = get_val(dst_ip_field, default="0.0.0.0", is_ip=True)
                        src_ip = int(ipaddress.ip_address(src_ip_str))
                        dst_ip = int(ipaddress.ip_address(dst_ip_str))
                        src_mac = get_val(src_mac_field, default=0)
                        dst_mac = get_val(dst_mac_field, default=0)
                        src_port = get_val(src_port_field, default=0)
                        dst_port = get_val(dst_port_field, default=0)
                        proto = get_val(proto_field, default=0)

                        duration           = get_val(get_idx(["duration"]))
                        max_iat            = get_val(get_idx(["max_iat"]))
                        fwd_pkt_count      = get_val(get_idx(["fwd_pkt_count"]))
                        bwd_pkt_count      = get_val(get_idx(["bwd_pkt_count"]))
                        fwd_bytes          = get_val(get_idx(["fwd_bytes"]))
                        bwd_bytes          = get_val(get_idx(["bwd_bytes"]))
                        fwd_max_pkt_len    = get_val(get_idx(["fwd_max_pkt_len"]))
                        bwd_max_pkt_len    = get_val(get_idx(["bwd_max_pkt_len"]))
                        flags_syn          = get_val(get_idx(["flags_syn", "syn"]))
                        flags_ack          = get_val(get_idx(["flags_ack", "ack"]))
                        flags_fin          = get_val(get_idx(["flags_fin", "fin"]))
                        flags_rst          = get_val(get_idx(["flags_rst", "rst"]))
                        flags_psh          = get_val(get_idx(["flags_psh"]))
                        max_win_size       = get_val(get_idx(["max_win_size"]))
                        init_fwd_win       = get_val(get_idx(["init_fwd_win"]))

                        pca_codes = []
                        for name in (transform_field_names or []):
                            idx = name_to_index.get(name)
                            if idx is None or idx >= len(st):
                                pca_codes.append(0)
                            else:
                                pca_codes.append(bytes_to_int(st[idx].bitstring))
                        if not pca_codes:
                            pca_codes = [0] * n_components

                        class_field = get_idx(["ml_result", "class_id", "class", "label", "result"])
                        class_id = get_val(class_field, default=0)
                        if class_field is None and not warn_once["missing_class"]:
                            warn_once["missing_class"] = True
                            print("WARNING: Class field not found in digest schema; defaulting class_id=0")
                    else:
                        if len(st) < 24:
                            if not warn_once["missing_fields"]:
                                warn_once["missing_fields"] = True
                                print(f"WARNING: Digest has insufficient fields ({len(st)}); skipping")
                            continue

                        src_ip = bytes_to_int(st[0].bitstring)
                        dst_ip = bytes_to_int(st[1].bitstring)
                        src_mac = bytes_to_int(st[2].bitstring)        # Src MAC
                        dst_mac = bytes_to_int(st[3].bitstring)        # Dst MAC
                        src_port = bytes_to_int(st[4].bitstring)
                        dst_port = bytes_to_int(st[5].bitstring)
                        proto = bytes_to_int(st[6].bitstring)

                        # Extract flow-based features in P4 digest_t order (paper Table 2)
                        duration           = bytes_to_int(st[7].bitstring)   # Flow duration
                        max_iat            = bytes_to_int(st[8].bitstring)   # Max inter-arrival time
                        fwd_pkt_count      = bytes_to_int(st[9].bitstring)
                        bwd_pkt_count      = bytes_to_int(st[10].bitstring)
                        fwd_bytes          = bytes_to_int(st[11].bitstring)
                        bwd_bytes          = bytes_to_int(st[12].bitstring)
                        fwd_max_pkt_len    = bytes_to_int(st[13].bitstring)
                        bwd_max_pkt_len    = bytes_to_int(st[14].bitstring)
                        flags_syn          = bytes_to_int(st[15].bitstring)
                        flags_ack          = bytes_to_int(st[16].bitstring)
                        flags_fin          = bytes_to_int(st[17].bitstring)
                        flags_rst          = bytes_to_int(st[18].bitstring)
                        flags_psh          = bytes_to_int(st[19].bitstring)
                        max_win_size       = bytes_to_int(st[20].bitstring)
                        init_fwd_win       = bytes_to_int(st[21].bitstring)

                        base_fields = 22
                        remaining = len(st) - base_fields
                        if remaining <= 0:
                            if not warn_once["missing_fields"]:
                                warn_once["missing_fields"] = True
                                print(f"WARNING: Digest has no transform/class fields ({len(st)}); skipping")
                            continue

                        # Layout: [22 raw flow fields, K transform codes, class_id]
                        pca_count = min(n_components, max(0, remaining - 1))
                        class_index = base_fields + (remaining - 1)

                        # Extract transform codes (pad if digest has fewer than expected)
                        pca_codes = []
                        for i in range(pca_count):
                            idx = base_fields + i
                            if idx >= len(st):
                                break
                            pca_codes.append(bytes_to_int(st[idx].bitstring))
                        if len(pca_codes) < n_components:
                            pca_codes.extend([0] * (n_components - len(pca_codes)))

                        # Class ID is the last field
                        if class_index >= len(st):
                            if not warn_once["missing_fields"]:
                                warn_once["missing_fields"] = True
                                print(f"WARNING: Digest class_id index out of range ({class_index} >= {len(st)}); skipping")
                            continue
                        class_id = bytes_to_int(st[class_index].bitstring)

                    class_label = CLASS_LABELS.get(class_id, "unknown")

                    # Skip drain-trigger probe flows (run_demo.sh sends UDP
                    # probes from 10.255.255.254 → 10.255.255.253 to force
                    # scan_and_drain; these create fake flows that corrupt
                    # per-pcap accuracy metrics).
                    if src_ip in _DRAIN_IPS and dst_ip in _DRAIN_IPS:
                        continue

                    features = {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_mac": src_mac,
                        "dst_mac": dst_mac,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "proto": proto,
                        "duration": duration,
                        "max_iat": max_iat,
                        "fwd_pkt_count": fwd_pkt_count,
                        "bwd_pkt_count": bwd_pkt_count,
                        "fwd_bytes": fwd_bytes,
                        "bwd_bytes": bwd_bytes,
                        "fwd_max_pkt_len": fwd_max_pkt_len,
                        "bwd_max_pkt_len": bwd_max_pkt_len,
                        "flags_psh": flags_psh,
                        "max_win_size": max_win_size,
                        "init_fwd_win": init_fwd_win,
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
                    ):
                        # Skip flows with < 1 packet (matches training data filter)
                        if (entry["features"]["fwd_pkt_count"] + entry["features"]["bwd_pkt_count"]) < 1:
                            _filtered_count += 1
                            continue
                        packet_id += 1
                        _digest_count_flow += 1
                        pca_display = format_codes_display(entry['pca_codes'], code_display_names, pca_bits)
                        f = entry["features"]
                        flags = entry["flags"]
                        class_id = entry["class_id"]
                        class_label = entry["class_label"]

                        verify_note = ""
                        votes_debug = ""
                        if model_type == "rf" and rf_model is not None and not model_pca_mismatch:
                            try:
                                x_np = np.array([build_model_input(entry, needs_transform, reduction_feature_columns)[:rf_model.n_features_in_]], dtype=np.float64)
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
                                votes_debug = format_votes_debug(votes, CLASS_LABELS)

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

                        _code_sfx = (pca_display + " | ") if pca_display else ""
                        print(f"[{packet_id:<4}] {str(ipaddress.ip_address(f['src_ip'])):>15}:{f['src_port']:<5} -> {str(ipaddress.ip_address(f['dst_ip'])):>15}:{f['dst_port']:<5} | "
                              f"SrcMAC={f['src_mac']:#014x} DstMAC={f['dst_mac']:#014x} | "
                              f"Dur={f['duration']:<12} MaxIAT={f['max_iat']:<12} "
                              f"FwdPkts={f['fwd_pkt_count']:<4} BwdPkts={f['bwd_pkt_count']:<4} "
                              f"FwdBytes={f['fwd_bytes']:<8} BwdBytes={f['bwd_bytes']:<8} Win={f['max_win_size']:<6} | "
                              f"Flags(S/A/F/R)={flags['syn']}/{flags['ack']}/{flags['fin']}/{flags['rst']} | "
                            f"{_code_sfx}"
                              f"Class={class_label}({class_id}){verify_note}", flush=True)
                        if votes_debug:
                            print(votes_debug, flush=True)

                        out.write(f"{f['src_ip']},{f['dst_ip']},{f['src_mac']},{f['dst_mac']},{f['src_port']},{f['dst_port']},{f['proto']},"
                                  f"{f['duration']},{f['max_iat']},{f['fwd_pkt_count']},{f['bwd_pkt_count']},"
                                  f"{f['fwd_bytes']},{f['bwd_bytes']},"
                                  f"{f.get('fwd_max_pkt_len',0)},{f.get('bwd_max_pkt_len',0)},"
                                  f"{flags['syn']},{flags['ack']},{flags['fin']},{flags['rst']},"
                                  f"{f.get('flags_psh',0)},{f['max_win_size']},{f.get('init_fwd_win',0)},"
                                  f"{format_codes_csv(entry['pca_codes'])}{class_id},{class_label}\n")
                        out.flush()

        print("\nShutting down...")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        print(f"Total digests received: {_digest_count_raw}, flows written: {_digest_count_flow}, filtered (<2 pkts): {_filtered_count}")
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
    
    parser = P4secArgumentParser(
        description='P4 Runtime Controller for Traffic Classification',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Notes:\n"
            "  - Reduction method is read from tables/reduction_config.json.\n"
            "  - Logs predictions to logs/predictions.csv.\n"
        )
    )
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
