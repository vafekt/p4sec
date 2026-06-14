#!/usr/bin/env python3
"""
PCA-linear reduction (paper Section: "additive per-feature projection").

Unlike 2_pca_generate_entries.py, which approximates the PCA *rotation* with a
multivariate DecisionTreeRegressor surrogate (one wide 18-field range key per
leaf, ~10^5 entries), this generator exploits the fact that a PCA projection is
*linear*:

    code_j = clamp( round( SUM_i  A'[j][i] * x_i  +  INIT_j ) , 0, 2^B - 1 )

so each component is a SUM of independent per-feature contributions.  Every
feature can therefore be looked up on its OWN single-field range table, and the
K component codes are accumulated in K signed metadata registers.  No feature
ever shares a match key with another feature.

Result vs the surrogate:
    * widest match key   : 1 field (<=16 bit)   vs   18 fields / 256 bit
    * transform entries  : SUM_i distinct(x_i)   vs   ~n_components * 10^4
    * projection fidelity: exact linear map      vs   axis-aligned staircase

The fixed-point contract embedded in tables/reduction_config.json["linear"]:
    acc_j (int<ACC_W>)  starts at INIT_j  (already scaled by 2^FP_SHIFT)
    each matched feature table adds its precomputed deltas (scaled by 2^FP_SHIFT)
    code_j = clamp( acc_j >> FP_SHIFT , 0, 2^B - 1 )

5_generating_p4_code.py reads that block and emits the matching P4.

Outputs (same filenames / same contract as every other step 2):
    tables/s1-commands.txt        per-feature contribution table_add entries
    tables/transform_mapping.csv  PC*_code (data-plane-exact) + Label
    tables/encoding_params.json   PCA params + ["linear"] fixed-point block
    tables/reduction_config.json  method = "lda_linear"
    tables/transform_metrics.json raw / float-PCA / linear-code accuracies
Code prefix: PC*_code
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

from pipeline_utils import (find_dataset_csv, quantize_features,
                            FEATURE_QUANTIZE, P4_FEATURE_MAX_QUANTIZED)

# ── Feature → P4 metadata field (mirrors 5_generating_p4_code.py) ───────────
FEATURE_TO_META = {
    "Protocol":        "protocol",
    "SrcPort":         "canon_src_port",
    "DstPort":         "canon_dst_port",
    "Duration":        "duration",
    "MaxIAT":          "max_iat",
    "FwdPktCount":     "fwd_pkt_count",
    "BwdPktCount":     "bwd_pkt_count",
    "FwdBytes":        "fwd_bytes",
    "BwdBytes":        "bwd_bytes",
    "FwdMaxPktLen":    "fwd_max_pkt_len",
    "BwdMaxPktLen":    "bwd_max_pkt_len",
    "FlagsSyn":        "flags_syn",
    "FlagsAck":        "flags_ack",
    "FlagsFin":        "flags_fin",
    "FlagsRst":        "flags_rst",
    "FlagsPsh":        "flags_psh",
    "MaxWinSize":      "max_win_size",
    "InitFwdWinBytes": "init_fwd_win",
}

# 18 ML features, paper Table 2 order (SrcIP/DstIP are flow ids, not features)
P4_FEATURE_COLS = list(FEATURE_TO_META.keys())

ACC_W   = 64      # signed accumulator width (int<64>)
DELTA_W = 64      # signed action-data width (int<64>)
FP_CAP  = 20      # max fractional bits


def meta_field(feat):
    """Quantized metadata field name the data plane carries for `feat`."""
    base = FEATURE_TO_META[feat]
    return base + "_q" if feat in FEATURE_QUANTIZE else base


def parse_args():
    p = argparse.ArgumentParser(
        description="LDA-linear: additive per-feature projection (no surrogate)")
    p.add_argument("--components", "-k", type=int, default=3,
                   help="Number of LDA components (default: 3 = max for 4-class data)")
    p.add_argument("--bits", "-b", type=int, default=32,
                   help="Quantization bits for the LDA codes (default: 32)")
    return p.parse_args()


def main():
    args = parse_args()
    k    = args.components
    bits = args.bits
    MAXCODE = (1 << bits) - 1

    here       = os.path.dirname(__file__)
    TABLES_DIR = os.path.join(here, "tables")
    os.makedirs(TABLES_DIR, exist_ok=True)

    csv_path = find_dataset_csv(__file__)
    print("Using dataset:", csv_path)
    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    for col in P4_FEATURE_COLS:
        if col not in df.columns:
            sys.exit(f"ERROR: feature column '{col}' missing from {csv_path}")
        bad = pd.to_numeric(df[col], errors="coerce").isna()
        if bad.any():
            df = df[~bad]

    # ── Quantize FIRST: the projection is defined on exactly the values the
    #    data plane carries (raw >> shift), so the in-network codes are exact.
    X_raw = df[P4_FEATURE_COLS].astype(np.int64)
    X_q   = quantize_features(X_raw).astype(np.int64).values   # (N, 18)
    y     = df[label_col].values
    N, F  = X_q.shape
    print(f"Samples: {N}  Features: {F}  Components: {k}  Bits: {bits}")

    # ── Fit StandardScaler + LDA on the QUANTIZED features ───────────────────
    # LDA is supervised: the projection directions maximise
    # between-class / within-class scatter (Fisher's criterion).  Max k for
    # n-class data is n_classes - 1.
    n_classes = len(np.unique(y))
    k_eff = min(k, n_classes - 1, X_q.shape[1])
    if k_eff != k:
        print(f"  capping k from {k} to {k_eff} "
              f"(LDA max = min(n_classes-1, n_features) = {k_eff})")
        k = k_eff
    scaler = StandardScaler()
    Z      = scaler.fit_transform(X_q.astype(float))
    lda    = LinearDiscriminantAnalysis(n_components=k, solver="svd")
    PC     = lda.fit_transform(Z, y)                  # (N, k)
    pc_min = PC.min(axis=0)
    pc_max = PC.max(axis=0)
    pc_rng = np.where((pc_max - pc_min) == 0, 1.0, pc_max - pc_min)

    # LDA's scalings_ has shape (n_features, max_components).  Take only the
    # first k columns and transpose to match the (k, n_features) convention.
    W      = lda.scalings_[:, :k].T                   # (k, 18) in scaled space
    pcamn  = lda.xbar_                                 # (18,)  class-pooled mean of scaled X
    mean   = scaler.mean_                             # (18,)
    scale  = scaler.scale_                            # (18,)

    G = MAXCODE / pc_rng                              # (k,)
    # code_j = SUM_i A[j,i] * x_i + INITreal_j
    A        = (G[:, None]) * W / scale[None, :]      # (k, 18)
    c        = W @ (mean / scale + pcamn)             # (k,)
    INITreal = -G * (c + pc_min)                      # (k,)

    # ── Choose FP_SHIFT so the worst-case accumulator stays inside int<63> ───
    qmax = np.array([P4_FEATURE_MAX_QUANTIZED[f] for f in P4_FEATURE_COLS],
                    dtype=float)
    term_max = np.abs(INITreal) + (np.abs(A) * qmax[None, :]).sum(axis=1)
    total_max = float(term_max.max()) if term_max.size else 1.0
    if total_max <= 0:
        FP_SHIFT = FP_CAP
    else:
        FP_SHIFT = int(np.clip(np.floor(np.log2((1 << 61) / total_max)),
                               0, FP_CAP))
    SCALE_FP = 1 << FP_SHIFT
    print(f"FP_SHIFT = {FP_SHIFT}  (worst-case |acc| ~ {total_max:.3e})")

    INIT = [int(round(INITreal[j] * SCALE_FP)) for j in range(k)]

    def delta_vec(i, v):
        """K fixed-point contributions of feature i taking quantised value v."""
        return [int(round(A[j, i] * v * SCALE_FP)) for j in range(k)]

    # ── Data-plane-exact code simulation (matches P4: sum, >>FP, clamp) ──────
    def codes_for(Xq):
        out = np.empty((Xq.shape[0], k), dtype=np.int64)
        for n in range(Xq.shape[0]):
            acc = list(INIT)
            for i in range(F):
                d = delta_vec(i, int(Xq[n, i]))
                for j in range(k):
                    acc[j] += d[j]
            for j in range(k):
                cj = acc[j] >> FP_SHIFT          # arithmetic floor shift
                cj = 0 if cj < 0 else (MAXCODE if cj > MAXCODE else cj)
                out[n, j] = cj
        return out

    PC_code = codes_for(X_q)
    for j in range(k):
        print(f"  LD{j+1}_code range: {PC_code[:, j].min()} -> {PC_code[:, j].max()}")

    # ── Per-feature contribution tables: disjoint midpoint tiles over [0,qmax]
    prio = [0]
    lines = []
    rules = []
    total_entries = 0
    for i, feat in enumerate(P4_FEATURE_COLS):
        mf   = meta_field(feat)
        vmax = int(qmax[i])
        vals = sorted(set(int(v) for v in X_q[:, i].tolist()))
        if not vals:
            vals = [0]
        # tile boundaries: each observed value owns the interval to the
        # midpoint with its neighbours; ends extend to [0, vmax].
        for idx, v in enumerate(vals):
            lo = 0 if idx == 0 else (vals[idx - 1] + v) // 2 + 1
            hi = vmax if idx == len(vals) - 1 else (v + vals[idx + 1]) // 2
            if hi < lo:
                hi = lo
            d = delta_vec(i, v)
            d_cli = " ".join(str(x & ((1 << DELTA_W) - 1)) for x in d)  # 2's-comp
            prio[0] += 1
            lines.append(f"table_add MyIngress.featc_{mf} addc_{mf} "
                         f"{lo}->{hi} => {d_cli} {prio[0]}\n")
            rules.append(f"{feat:16s} in [{lo:>6},{hi:>6}] (rep={v:>6}) -> "
                         + " ".join(f"LD{j+1}+={d[j]}" for j in range(k)) + "\n")
            total_entries += 1
    print(f"Total transform entries: {total_entries} "
          f"(widest key: 1 field; vs surrogate's 18-field key)")

    with open(os.path.join(TABLES_DIR, "s1-commands.txt"), "w") as f:
        f.write("# PCA-linear per-feature contribution entries.\n")
        f.writelines(lines)
    with open(os.path.join(TABLES_DIR, "transform_rules.txt"), "w") as f:
        f.write("# PCA-linear contribution rules (feature range -> PC deltas)\n")
        f.write(f"# INIT = {INIT}   FP_SHIFT = {FP_SHIFT}\n\n")
        f.writelines(rules)

    # ── transform_mapping.csv (classifier trains on these exact codes) ───────
    mp = {f"LD{j+1}_float": PC[:, j] for j in range(k)}
    mp.update({f"LD{j+1}_code": PC_code[:, j] for j in range(k)})
    mp["Label"] = y
    pd.DataFrame(mp).to_csv(
        os.path.join(TABLES_DIR, "transform_mapping.csv"), index=False)

    # ── encoding_params.json (PCA params + linear fixed-point block) ─────────
    encoding_params = {
        "method":         "lda_linear",
        "n_components":   int(k),
        "bits":           int(bits),
        "max_val":        int(MAXCODE),
        "transform_min":  pc_min.tolist(),
        "transform_max":  pc_max.tolist(),
        "transform_range": (pc_max - pc_min).tolist(),
        "transform_components": W.tolist(),
        "transform_mean": pcamn.tolist(),
        "scaler_mean":    mean.tolist(),
        "scaler_scale":   scale.tolist(),
        "linear": {
            "fp_shift":  int(FP_SHIFT),
            "acc_width": ACC_W,
            "delta_width": DELTA_W,
            "maxcode":   int(MAXCODE),
            "init":      INIT,
            "features":  [meta_field(f) for f in P4_FEATURE_COLS],
        },
    }
    with open(os.path.join(TABLES_DIR, "encoding_params.json"), "w") as f:
        json.dump(encoding_params, f, indent=2)

    # ── reduction_config.json (drives steps 3/4/5); linear block included ────
    feature_columns = [f"LD{j+1}_code" for j in range(k)]
    reduction_config = {
        "method": "lda_linear",
        "feature_columns": feature_columns,
        "feature_max_values": {c: int(MAXCODE) for c in feature_columns},
        "needs_transform_tables": True,
        "n_components": int(k),
        "bits": int(bits),
        "linear": encoding_params["linear"],
    }
    with open(os.path.join(TABLES_DIR, "reduction_config.json"), "w") as f:
        json.dump(reduction_config, f, indent=2)

    # ── transform_metrics.json: raw vs float-PCA vs linear-code ──────────────
    labels = sorted(np.unique(y))

    def fit_acc(Xtr):
        clf = DecisionTreeClassifier(max_depth=100, random_state=42)
        clf.fit(Xtr, y)
        return clf.predict(Xtr)

    blocks = {}
    for name, Xfit in [("raw_features", X_q),
                       ("original_pca", PC),
                       ("linear_codes", PC_code)]:
        yp = fit_acc(Xfit)
        blocks[name] = {
            "accuracy": float(accuracy_score(y, yp)),
            "classification_report": classification_report(
                y, yp, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, yp, labels=labels).tolist(),
        }
    with open(os.path.join(TABLES_DIR, "transform_metrics.json"), "w") as f:
        json.dump({"labels": labels, **blocks}, f, indent=2)

    print("\n=== DecisionTreeClassifier accuracy (resubstitution) ===")
    for name in ("raw_features", "original_pca", "linear_codes"):
        print(f"  {name:16s}: {blocks[name]['accuracy']:.4f}")

    print("\nNext:")
    print("  python3 3_train_model.py -m dt")
    print("  python3 4_generate_model_entries.py -m dt")
    print("  python3 5_generating_p4_code.py -m dt")


if __name__ == "__main__":
    main()
