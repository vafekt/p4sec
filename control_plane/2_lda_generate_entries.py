#!/usr/bin/env python3
"""
LDA (Linear Discriminant Analysis) + surrogate DT pipeline with quantization.

Drop-in replacement for 2_pca_generate_entries.py.

Outputs (same locations, compatible with ALL step 3/4 scripts):
  tables/transform_mapping.csv       Columns: LD1_float…LDk_float, LD1_code…LDk_code, Label
  tables/encoding_params.json        Encoding metadata (LDA transform, min/max/range)
  tables/reduction_config.json       Universal config for steps 3/4
  tables/s1-commands.txt             P4 table_add commands for LDA transform tables
  tables/transform_rules.txt         Human-readable surrogate tree rules
  tables/transform_metrics.json      Classification accuracy comparison

Why LDA?
--------
PCA maximises variance regardless of labels.  LDA maximises the ratio of
between-class to within-class scatter, so reduced features are optimised
for class separation.  This typically yields better accuracy with fewer
components, especially when classes overlap in the raw feature space.

LDA produces at most min(n_features, n_classes-1) components.

P4 deployment is identical to PCA: a surrogate DecisionTreeRegressor maps
raw features -> quantised LDA codes via lda_component* range-match tables.
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score, accuracy_score, confusion_matrix, classification_report,
)

from pipeline_utils import P4_FEATURE_MAX, ALL_RAW_FEATURES, find_dataset_csv

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

# ==========================================================
# 0. CONFIG
# ==========================================================
parser = P4secArgumentParser(
    description="LDA pipeline with quantization (works with any classifier in step 3)",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
            "Outputs:\n"
            "  tables/s1-commands.txt           (P4 transform entries)\n"
            "  tables/encoding_params.json      (transform params)\n"
            "  tables/transform_mapping.csv     (codes + labels)\n"
            "  tables/reduction_config.json     (universal config)\n"
            "Code prefix: LD*_code\n"
        )
    )
parser.add_argument("--components", "-k", type=int, default=None,
                    help="Number of LDA components (default: n_classes-1)")
parser.add_argument("--bits", "-b", type=int, default=16,
                    help="Quantization bits for LDA codes (default: 16)")
parser.add_argument("--solver", choices=["svd", "lsqr", "eigen"],
                    default="svd", help="LDA solver (default: svd)")
parser.add_argument("--max-leaf-nodes", "-l", type=int, default=300000,
                    help="Maximum leaf nodes in the surrogate tree (default: 300000)")
parser.add_argument("--surrogate", "-s", choices=["dt", "rf", "gbr"],
                    default="dt",
                    help=(
                        "Surrogate regressor for mapping raw features → quantised LDA codes.\n"
                        "  dt   DecisionTreeRegressor (baseline, fast)\n"
                        "  rf   RandomForest teacher → distil to DT (better codes, ~2× slower)\n"
                        "  gbr  GradientBoosting teacher → distil to DT (best accuracy, ~50× slower)\n"
                        "(default: dt)"
                    ))
args = parser.parse_args()

USER_K = args.components
BITS = args.bits
SOLVER = args.solver
SURROGATE = args.surrogate
MAX_LEAF_NODES = args.max_leaf_nodes if args.max_leaf_nodes and args.max_leaf_nodes > 0 else None

if BITS not in [8, 16, 24, 32]:
    print(f"WARNING: BITS={BITS}. Recommended values are 8, 16, 24, 32.")

MAX_VAL = 2**BITS - 1

# ==========================================================
# 1. Load dataset
# ==========================================================
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

csv_path = find_dataset_csv(__file__)
print("Using dataset:", csv_path)
df = pd.read_csv(csv_path)
label_col = df.columns[-1]
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

P4_FEATURE_COLS = [
    "Protocol", "SrcPort", "DstPort",
    "Duration", "MaxIAT", "UrgCount",
    "FwdPktCount", "BwdPktCount", "FwdBytes", "BwdBytes",
    "MaxWinSize", "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst",
    "MinIAT", "FwdMaxPktLen", "BwdMaxPktLen", "FlagsPsh", "InitFwdWinBytes",
]
X_df = df_clean[P4_FEATURE_COLS].astype(int)
feature_cols = X_df.columns.tolist()
X = X_df.values
y = df_clean[label_col].values

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print("Labels:", np.unique(y)[:10])

# ==========================================================
# 2. Determine number of LDA components
# ==========================================================
n_classes = len(np.unique(y))
max_lda_k = min(X.shape[1], n_classes - 1)
if USER_K is not None:
    k = min(USER_K, max_lda_k)
    if USER_K > max_lda_k:
        print(f"WARNING: requested {USER_K} but LDA max is {max_lda_k}. Capping.")
else:
    k = max_lda_k
print(f"LDA components: {k} (max possible: {max_lda_k})")

# ==========================================================
# 3. Fit LDA on raw features
# ==========================================================
X_train = X_test = X
y_train = y_test = y

lda = LinearDiscriminantAnalysis(n_components=k, solver=SOLVER)
LD_train = lda.fit_transform(X_train, y_train)
LD_test = lda.transform(X_test)

print(f"LD_train shape: {LD_train.shape}")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
print(f"Total explained variance: {sum(lda.explained_variance_ratio_):.4f}")

# ==========================================================
# 4. Quantize LDA codes -> [0, MAX_VAL]
# ==========================================================
ld_min = LD_train.min(axis=0)
ld_max = LD_train.max(axis=0)
ld_range = ld_max - ld_min
ld_range_safe = np.where(ld_range == 0, 1, ld_range)

def encode(F, fmin, frange, maxv):
    F = np.asarray(F)
    norm = (F - fmin.reshape(1, -1)) / frange.reshape(1, -1)
    codes = np.clip(np.rint(norm * maxv), 0, maxv)
    return codes.astype(int)

def decode(codes, fmin, frange, maxv):
    codes = np.asarray(codes)
    return (codes / maxv) * frange.reshape(1, -1) + fmin.reshape(1, -1)

LD_code_train = encode(LD_train, ld_min, ld_range_safe, MAX_VAL)
LD_code_test  = encode(LD_test,  ld_min, ld_range_safe, MAX_VAL)

for j in range(k):
    print(f"LD{j+1}_code range: {LD_code_train[:, j].min()} -> {LD_code_train[:, j].max()}")

# Quantization R2
for j in range(k):
    approx = decode(LD_code_train, ld_min, ld_range_safe, MAX_VAL)
    r2 = r2_score(LD_train[:, j], approx[:, j])
    print(f"LD{j+1} quantization R2: {r2:.6f}")

# ==========================================================
# 5. Surrogate regressor: raw features -> quantized LDA codes
#    --surrogate dt   : single DT (baseline)
#    --surrogate rf   : RF teacher → refined targets → DT student
#    --surrogate gbr  : GBR teacher → refined targets → DT student
# ==========================================================
print(f"\nSurrogate method: {SURROGATE}")

if SURROGATE == 'rf':
    teacher = RandomForestRegressor(
        n_estimators=20, max_depth=None, min_samples_leaf=1,
        max_features='sqrt', random_state=42, n_jobs=-1)
    teacher.fit(X_train, LD_code_train)
    refined_targets = np.clip(np.rint(teacher.predict(X_train)), 0, MAX_VAL).astype(int)
    print(f"RF teacher trained (20 trees). Distilling to DT student...")
elif SURROGATE == 'gbr':
    refined_targets = np.zeros_like(LD_code_train)
    for j in range(k):
        gbr_j = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        gbr_j.fit(X_train, LD_code_train[:, j])
        refined_targets[:, j] = np.clip(
            np.rint(gbr_j.predict(X_train)), 0, MAX_VAL).astype(int)
    print(f"GBR teacher trained ({k} components × 100 trees). Distilling to DT student...")
else:
    refined_targets = LD_code_train

tree = DecisionTreeRegressor(
    max_depth=None, min_samples_leaf=1, max_leaf_nodes=MAX_LEAF_NODES, random_state=42)
tree.fit(X_train, refined_targets)

leaf_ids = tree.apply(X_train)
unique_leaves = np.unique(leaf_ids)
print(f"Surrogate tree leaves: {len(unique_leaves)}")

leaf_to_codes = {}
for lid in unique_leaves:
    idx = np.where(leaf_ids == lid)[0]
    labels_in_leaf = y_train[idx]
    values, counts = np.unique(labels_in_leaf, return_counts=True)
    majority_class = values[np.argmax(counts)]
    mask = labels_in_leaf == majority_class
    leaf_to_codes[lid] = np.rint(refined_targets[idx][mask].mean(axis=0)).astype(int)

def get_tree_codes(dt, l2c, X, k):
    ids = dt.apply(X)
    codes = np.zeros((X.shape[0], k), dtype=int)
    for i, lid in enumerate(ids):
        codes[i, :] = l2c[lid]
    return codes

LD_code_tree_train = get_tree_codes(tree, leaf_to_codes, X_train, k)
LD_code_tree_test  = get_tree_codes(tree, leaf_to_codes, X_test, k)

# ==========================================================
# 6. Classification comparison
# ==========================================================
clf_raw = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, clf_raw.predict(X_test))

clf_lda = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_lda.fit(LD_train, y_train)
acc_lda = accuracy_score(y_test, clf_lda.predict(LD_test))

clf_code = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_code.fit(LD_code_train, y_train)
acc_code = accuracy_score(y_test, clf_code.predict(LD_code_test))

clf_tree = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_tree.fit(LD_code_tree_train, y_train)
acc_tree = accuracy_score(y_test, clf_tree.predict(LD_code_tree_test))

print(f"\n=== Classification Accuracy ===")
print(f"Raw features        : {acc_raw}")
print(f"LDA float           : {acc_lda}")
print(f"Quantized LDA codes : {acc_code}")
print(f"Tree-approx codes   : {acc_tree}")

labels = sorted(np.unique(y))
metrics = {
    "method": "LDA", "n_components": k, "solver": SOLVER,
    "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
    "labels": labels,
    "raw_features":      {"accuracy": float(acc_raw)},
    "original_lda":      {"accuracy": float(acc_lda)},
    "quantized_codes":   {"accuracy": float(acc_code)},
    "tree_approx_codes": {"accuracy": float(acc_tree)},
}
with open(os.path.join(TABLES_DIR, "transform_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ==========================================================
# 7. Save mapping CSV  (LD*_float, LD*_code, Label)
# ==========================================================
LD_all = np.vstack([LD_train, LD_test])
LD_code_tree_all = np.vstack([LD_code_tree_train, LD_code_tree_test])
y_all = np.concatenate([y_train, y_test])

mapping = {}
for j in range(k):
    mapping[f"LD{j+1}_float"] = LD_all[:, j]
for j in range(k):
    mapping[f"LD{j+1}_code"] = LD_code_tree_all[:, j]
mapping["Label"] = y_all

pd.DataFrame(mapping).to_csv(
    os.path.join(TABLES_DIR, "transform_mapping.csv"), index=False)
print("Saved transform_mapping.csv (LD columns)")

# ==========================================================
# 8. Save encoding parameters  (backward-compat keys)
# ==========================================================
encoding_params = {
    "method": "lda",
    "n_components": int(k),
    "bits": int(BITS),
    "max_val": int(MAX_VAL),
    "transform_min": ld_min.tolist(),
    "transform_max": ld_max.tolist(),
    "transform_range": ld_range.tolist(),
    "auto_selected": bool(USER_K is None),
    "transform_components": lda.scalings_[:, :k].T.tolist(),
    "transform_mean": lda.xbar_.tolist(),
    "feature_names": feature_cols,
}
with open(os.path.join(TABLES_DIR, "encoding_params.json"), "w") as f:
    json.dump(encoding_params, f, indent=2)
print("Saved encoding_params.json")

# ==========================================================
# 9. Save universal reduction_config.json
# ==========================================================
feature_columns = [f"LD{j+1}_code" for j in range(k)]
feature_max_values = {col: int(MAX_VAL) for col in feature_columns}

reduction_config = {
    "method": "lda",
    "feature_columns": feature_columns,
    "feature_max_values": feature_max_values,
    "needs_transform_tables": True,
    "n_components": int(k),
    "bits": int(BITS),
}
with open(os.path.join(TABLES_DIR, "reduction_config.json"), "w") as f:
    json.dump(reduction_config, f, indent=2)
print("Saved reduction_config.json")


# ==========================================================
# 10. Generate P4 table_add commands for LDA transform tables
# ==========================================================
feature_max_map = {}
for feat in feature_cols:
    feature_max_map[feat] = P4_FEATURE_MAX.get(feat, MAX_VAL)

def minimize(path):
    domain = {}
    for (feature, sign, threshold) in path:
        domain.setdefault(feature, {"min": None, "max": None})
        if sign == "<=":
            domain[feature]["max"] = threshold
        else:
            domain[feature]["min"] = threshold
    return domain

def format_ranges(domain, feature_names, fmax_map):
    clauses = []
    for fn in feature_names:
        fm = fmax_map.get(fn, MAX_VAL)
        if fn not in domain:
            clauses.append(f"0->{fm}")
            continue
        val = domain[fn]
        lo = (int(val["min"]) + 1) if val["min"] is not None else 0
        hi = int(val["max"]) if val["max"] is not None else fm
        lo = max(0, lo)
        hi = min(hi, fm)
        clauses.append(f"{lo}->{hi}")
    return clauses

def write_p4_commands(dt, fnames, l2c, k, fmax_map, filename):
    t = dt.tree_
    left, right, thresh, feats = t.children_left, t.children_right, t.threshold, t.feature
    prio = [0]

    def dfs(nid, path):
        if left[nid] == right[nid]:
            prio[0] += 1
            new_path = [(fnames[feats[n]], s, thresh[n])
                        for n, s in path if feats[n] != -2]
            dom = minimize(new_path)
            clauses = format_ranges(dom, fnames, fmax_map)
            codes = l2c[nid]
            for j in range(k):
                f.write(f"table_add MyIngress.lda_component{j+1} "
                        f"set_ld{j+1}_code {' '.join(clauses)} => {int(codes[j])} {prio[0]}\n")
            return
        dfs(left[nid], path + [(nid, "<=")])
        dfs(right[nid], path + [(nid, ">")])

    with open(filename, "w") as f:
        dfs(0, [])

write_p4_commands(tree, feature_cols, leaf_to_codes, k, feature_max_map,
                  os.path.join(TABLES_DIR, "s1-commands.txt"))
print("Saved s1-commands.txt")

# Human-readable rules
def write_if_rules(dt, fnames, l2c, filename, k):
    t = dt.tree_
    left, right, thresh, feats = t.children_left, t.children_right, t.threshold, t.feature

    def dfs(nid, path):
        if left[nid] == right[nid]:
            clauses = [f"{fnames[feats[n]]} {s} {thresh[n]:.2f}"
                       for n, s in path if feats[n] != -2]
            codes = l2c[nid]
            assigns = " AND ".join(f"LD{j+1}_code = {int(codes[j])}" for j in range(k))
            f.write(f"IF ({' AND '.join(clauses) or 'TRUE'}) THEN {assigns};\n")
            return
        dfs(left[nid], path + [(nid, "<=")])
        dfs(right[nid], path + [(nid, ">")])

    with open(filename, "w") as f:
        f.write(f"# LDA Transformation Rules\n# {len(fnames)} raw features -> {k} LDA codes\n\n")
        dfs(0, [])

write_if_rules(tree, feature_cols, leaf_to_codes,
               os.path.join(TABLES_DIR, "transform_rules.txt"), k)

print("\n" + "=" * 60)
print("LDA complete.  Run any step 3/4 classifier next:")
print("  python3 3_train_model.py --model-type dt")
print("  python3 3_train_model.py --model-type rf")
print("  python3 3_train_model.py --model-type xgb")
print("  python3 3_train_model.py --model-type gb")
print("  python3 3_train_model.py --model-type knn")
print("  python3 3_train_model.py --model-type svm")
print("  python3 3_train_model.py --model-type cnn")
print("=" * 60)
