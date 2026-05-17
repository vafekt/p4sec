#!/usr/bin/env python3
"""
Raw-feature baseline (paper Section 4.2 "raw-feature range-match DT").

Implements the no-PCA branch of the paper's pipeline: the 20 quantised flow
features feed the downstream classifier directly via range-match P4 entries,
with no PCA projection in between.  Output is byte-compatible with what
2_pca_generate_entries.py produces, so steps 3-5 work unchanged.

Inputs : dataset/dataset.csv
Outputs:
  tables/transform_mapping.csv     quantised 20-feature vectors + Label
  tables/reduction_config.json     method='raw', needs_transform_tables=False
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from pipeline_utils import (
    P4_FEATURE_MAX, P4_FEATURE_MAX_QUANTIZED,
    find_dataset_csv, quantize_features, FEATURE_QUANTIZE,
)

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


P4_FEATURE_COLS = [
    "Protocol",
    "SrcPort", "DstPort",
    "Duration", "MaxIAT",
    "FwdPktCount", "BwdPktCount", "FwdBytes", "BwdBytes",
    "FwdMaxPktLen", "BwdMaxPktLen",
    "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst", "FlagsPsh",
    "MaxWinSize", "InitFwdWinBytes",
    "FlowCountPerSrc", "SynCountPerDst",
]


def main():
    parser = P4secArgumentParser(
        description="Raw-feature baseline (no PCA) — paper Section 4.2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.parse_args()

    tables_dir = os.path.join(os.path.dirname(__file__), "tables")
    os.makedirs(tables_dir, exist_ok=True)

    csv_path = find_dataset_csv(__file__)
    print("Using dataset:", csv_path)
    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

    for col in P4_FEATURE_COLS:
        if col not in df_clean.columns:
            sys.exit(f"ERROR: feature column '{col}' missing from {csv_path}")
        bad = pd.to_numeric(df_clean[col], errors='coerce').isna()
        if bad.any():
            print(f"WARNING: dropping {bad.sum()} row(s) with non-numeric '{col}'")
            df_clean = df_clean[~bad]

    X_raw = df_clean[P4_FEATURE_COLS].astype(int)
    X_q = quantize_features(X_raw)

    out_df = X_q.copy()
    out_df["Label"] = df_clean[label_col].values

    out_path = os.path.join(tables_dir, "transform_mapping.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")

    config = {
        "method": "raw",
        "needs_transform_tables": False,
        "feature_columns": P4_FEATURE_COLS,
        "feature_max_values": {
            col: int(P4_FEATURE_MAX_QUANTIZED.get(col, P4_FEATURE_MAX[col]))
            for col in P4_FEATURE_COLS
        },
        "bits": None,
        "n_components": len(P4_FEATURE_COLS),
        "quantize": {col: list(FEATURE_QUANTIZE[col])
                     for col in P4_FEATURE_COLS if col in FEATURE_QUANTIZE},
    }
    cfg_path = os.path.join(tables_dir, "reduction_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {cfg_path}")

    # Reset s1-commands.txt so steps 4/5 start from a clean slate
    s1 = os.path.join(tables_dir, "s1-commands.txt")
    with open(s1, "w") as f:
        f.write("# Raw-feature baseline (no PCA transform entries).\n")
        f.write("# Classifier entries appended by 4_generate_model_entries.py.\n")
    print(f"Initialised {s1}")


if __name__ == "__main__":
    main()
