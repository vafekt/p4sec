#!/usr/bin/env python3
"""
Shared pipeline utilities for the P4 ML classification pipeline.

Provides universal feature/config detection that works with any step 2 method:
    - PCA         (2_pca_generate_entries.py)         → PC*_code columns
    - LDA         (2_lda_generate_entries.py)         → LD*_code columns
    - Autoencoder (2_autoencoder_generate_entries.py) → AE*_code columns

The contract between step 2 and steps 3/4/5 is the file:
    tables/reduction_config.json

All step 2 scripts write this file.  All step 3/4 scripts read it.
"""

import os
import glob
import json
import pandas as pd

# P4 field widths for raw flow features (used when classifying directly
# on raw features without PCA/LDA transformation).
P4_FEATURE_MAX = {
    "SrcIP":           (2**32 - 1),   # bit<32> (IPv4 as integer)
    "DstIP":           (2**32 - 1),   # bit<32> (IPv4 as integer)
    "Protocol":        (2**8  - 1),   # bit<8>
    "SrcPort":         (2**16 - 1),   # port_t / bit<16>
    "DstPort":         (2**16 - 1),   # port_t / bit<16>
    "Duration":        (2**48 - 1),   # bit<48>
    "MaxIAT":          (2**48 - 1),   # bit<48>
    "FwdPktCount":     (2**32 - 1),   # bit<32>
    "BwdPktCount":     (2**32 - 1),   # bit<32>
    "FwdBytes":        (2**32 - 1),   # bit<32>
    "BwdBytes":        (2**32 - 1),   # bit<32>
    "MaxWinSize":      (2**16 - 1),   # bit<16>
    "FlagsSyn":        (2**32 - 1),   # bit<32>
    "FlagsAck":        (2**32 - 1),   # bit<32>
    "FlagsFin":        (2**32 - 1),   # bit<32>
    "FlagsRst":        (2**32 - 1),   # bit<32>
    "FwdMaxPktLen":    (2**16 - 1),   # bit<16>
    "BwdMaxPktLen":    (2**16 - 1),   # bit<16>
    "FlagsPsh":        (2**32 - 1),   # bit<32>
    "InitFwdWinBytes": (2**16 - 1),   # bit<16>
}

# All 20 P4 raw flow feature names (ML classifier input), in canonical order
ALL_RAW_FEATURES = list(P4_FEATURE_MAX.keys())

# Columns that are never ML features
NON_FEATURE_COLS = {'Label'}

# ─── Feature quantization for Tofino range-match ────────────────────────
# Tofino limits range match to fields ≤ 20 bits (5 PHV nibbles).
# We pre-quantize features >20 bits via right-shift so the PCA surrogate DT
# trains on the same reduced-precision values that the data plane sees.
# Both BMv2 and Tofino use the same quantized values for consistency.
#
# Format: { feature_name: (shift_amount, quantized_bits) }
# The quantized value = raw_value >> shift_amount, clamped to quantized_bits.
# Features not listed here are already ≤ 16 bits and need no quantization.
FEATURE_QUANTIZE = {
    "Duration":    (20, 16),  # 48b ns → 16b (~1ms granularity, max ~65s)
    "MaxIAT":      (20, 16),  # 48b ns → 16b (~1ms granularity)
    "FwdPktCount": (0,  16),  # 32b → 16b (truncate lower 16 bits; max 65535)
    "BwdPktCount": (0,  16),  # 32b → 16b
    "FwdBytes":    (4,  16),  # 32b → 16b (16-byte granularity; max ~1MB)
    "BwdBytes":    (4,  16),  # 32b → 16b
    "FlagsSyn":    (0,   8),  # 32b → 8b (SYN count rarely > 255)
    "FlagsAck":    (0,  16),  # 32b → 16b
    "FlagsFin":    (0,   8),  # 32b → 8b
    "FlagsRst":    (0,   8),  # 32b → 8b
    "FlagsPsh":    (0,  16),  # 32b → 16b
}

# P4 field widths AFTER quantization (for range-match key sizing)
P4_FEATURE_MAX_QUANTIZED = {}
for _feat, _max in P4_FEATURE_MAX.items():
    if _feat in FEATURE_QUANTIZE:
        _shift, _qbits = FEATURE_QUANTIZE[_feat]
        P4_FEATURE_MAX_QUANTIZED[_feat] = (2**_qbits - 1)
    else:
        P4_FEATURE_MAX_QUANTIZED[_feat] = _max


def quantize_features(X_df):
    """
    Apply the same quantization that the P4 data plane does (right-shift + truncate).
    Input: DataFrame with raw feature columns.
    Returns: DataFrame with quantized columns (same column names).
    """
    X_q = X_df.copy()
    for feat, (shift, qbits) in FEATURE_QUANTIZE.items():
        if feat in X_q.columns:
            max_val = 2**qbits - 1
            if shift > 0:
                X_q[feat] = (X_q[feat] // (2**shift)).clip(upper=max_val).astype(int)
            else:
                X_q[feat] = X_q[feat].clip(upper=max_val).astype(int)
    return X_q

TRANSFORM_METHOD_TO_PREFIX = {
    'pca': 'PC',
    'lda': 'LD',
    'autoencoder': 'AE',
}


def find_dataset_csv(script_file=None):
    """
    Locate the dataset CSV.  Searches control_plane/dataset/ then ../dataset/.
    Prefers a file named 'dataset.csv'; falls back to the most-recently modified CSV.

    Pass __file__ from the calling script so the search is relative to that script,
    or omit it to search relative to this utils file.
    """
    base = os.path.dirname(os.path.abspath(script_file or __file__))
    root = os.path.abspath(os.path.join(base, ".."))
    for d in [os.path.join(base, "dataset"), os.path.join(root, "dataset")]:
        if not os.path.isdir(d):
            continue
        csvs = glob.glob(os.path.join(d, "*.csv"))
        if not csvs:
            continue
        preferred = [p for p in csvs if os.path.basename(p).lower() == "dataset.csv"]
        if preferred:
            return preferred[0]
        return sorted(csvs, key=os.path.getmtime, reverse=True)[0]
    raise FileNotFoundError("No CSV dataset found in any dataset/ directory")


def load_reduction_config(tables_dir):
    """
    Load reduction_config.json written by step 2.
    Returns the config dict, or None if the file does not exist.
    """
    path = os.path.join(tables_dir, 'reduction_config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def detect_feature_columns(csv_path, tables_dir=None):
    """
    Detect feature column names from config or CSV.

    Priority:
      1. reduction_config.json (written by any step 2)
    2. Auto-detect *_code columns (backward compat with old PCA)
      3. All non-Label, non-identifier, non-*_float columns
    """
    if tables_dir is None:
        tables_dir = os.path.dirname(csv_path)

    config = load_reduction_config(tables_dir)
    if config is not None:
        return config['feature_columns']

    # Fallback: auto-detect from CSV header
    df = pd.read_csv(csv_path, nrows=1)
    code_cols = [c for c in df.columns if c.endswith('_code')]
    if code_cols:
        return code_cols

    # Raw feature columns
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and not c.endswith('_float')]


def detect_feature_max_values(tables_dir):
    """
    Return a dict {feature_name: max_int_value} for P4 range matching.

    For PCA/LDA/Autoencoder codes: all features share MAX_VAL = 2^bits - 1.
    For raw features:  each feature has its own P4 field width.
    """
    config = load_reduction_config(tables_dir)
    if config is not None and 'feature_max_values' in config:
        # Convert values to int (JSON may have stored them as float)
        return {k: int(v) for k, v in config['feature_max_values'].items()}

    # Fallback: try encoding_params.json (canonical) or legacy pca_encoding_params.json
    for _enc_name in ['encoding_params.json', 'pca_encoding_params.json']:
        enc_params_path = os.path.join(tables_dir, _enc_name)
        if not os.path.exists(enc_params_path):
            continue
        with open(enc_params_path) as f:
            enc_params = json.load(f)
        bits = enc_params.get('bits', 16)
        max_val = 2 ** bits - 1
        n_comp = enc_params.get('n_components', 2)
        method = str(enc_params.get('method', 'pca')).lower()
        prefix = TRANSFORM_METHOD_TO_PREFIX.get(method, 'PC')
        return {f"{prefix}{j+1}_code": max_val for j in range(n_comp)}

    # Ultimate fallback: assume 16-bit codes
    return {}


def detect_needs_transform(tables_dir):
    """
    Return True if the pipeline uses PCA/LDA/Autoencoder transform tables,
    False if classifier operates directly on raw features.
    """
    config = load_reduction_config(tables_dir)
    if config is not None:
        return config.get('needs_transform_tables', True)
    # Default: assume PCA (backward compat)
    return True


def detect_bits(tables_dir):
    """Return the quantisation bit width for the active transform method."""
    config = load_reduction_config(tables_dir)
    if config is not None:
        return config.get('bits', None)
    # Fallback
    enc_path = os.path.join(tables_dir, 'encoding_params.json')
    if os.path.exists(enc_path):
        with open(enc_path) as f:
            return int(json.load(f).get('bits', 16))
    return 16
