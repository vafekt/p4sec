#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import os
import glob
import argparse

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score

# ==========================================================
# 0. CONFIG / CLI
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PCA + DT pipeline with quantization")
    parser.add_argument("--components", "-k", type=int, default=None,
                        help="Number of PCA components (if omitted, auto-select for high explained variance)")
    parser.add_argument("--bits", "-b", type=int, default=16,
                        help="Quantization bits for PCA codes (default: 16). Supports 8, 16, 24, 32 bits.")
    # Hidden option: variance target for auto-selection
    parser.add_argument("--var-target", type=float, default=0.95,
                        help="Explained variance target for auto PCA component selection (default: 0.95)")
    # Removed tree-depth and tree-min-leaf arguments; always use default DecisionTreeRegressor params
    return parser.parse_args()

args = parse_args()

USER_N_COMPONENTS = args.components
BITS = args.bits
VAR_TARGET = args.var_target

# Validate BITS parameter
if BITS not in [8, 16, 24, 32]:
    print(f"WARNING: BITS={BITS}. Recommended values are 8, 16, 24, 32 for P4 compatibility.")
    print(f"P4 range match can handle arbitrary widths, but ensure switch pipeline width supports it.")

# ==========================================================
# No tree hyperparameter tuning; always use default DecisionTreeRegressor params
# ==========================================================
MAX_VAL = 2**BITS - 1     # Dynamic max value based on BITS (e.g., 255 for 8-bit, 65535 for 16-bit, 4294967295 for 32-bit)

# ==========================================================
# 1. Load dataset & cleaning
# ==========================================================
# Resolve paths relative to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PCAPS_DIR = os.path.join(ROOT_DIR, "pcaps")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# Locate dataset CSV inside a `dataset/` directory (filename not fixed)
def find_dataset_csv():
    candidates_dirs = [
        os.path.join(os.path.dirname(__file__), "dataset"),
        os.path.join(ROOT_DIR, "dataset"),
    ]
    for d in candidates_dirs:
        if not os.path.isdir(d):
            continue
        csvs = glob.glob(os.path.join(d, "*.csv"))
        if not csvs:
            continue
        # Prefer a file named dataset.csv if present
        preferred = [p for p in csvs if os.path.basename(p).lower() == "dataset.csv"]
        if preferred:
            return preferred[0]
        # Otherwise pick the most recently modified CSV
        csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return csvs[0]
    raise FileNotFoundError("No CSV dataset found in any dataset/ directory")

csv_path = find_dataset_csv()
print("Using dataset:", csv_path)
df = pd.read_csv(csv_path)

# Assume last column is Label (string attack name)
label_col = df.columns[-1]

# Remove NaN / Inf
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

# ==========================================================
# Select exactly the 8 P4-compatible features in the EXACT order
# that the pca_component* tables declare their key fields.
# 1_data_extraction.py now outputs these columns directly.
# ==========================================================
P4_FEATURE_COLS = [
    "Duration", "MaxIAT", "UrgCount",
    "FwdPktCount", "BwdPktCount",
    "FwdBytes", "BwdBytes",
    "MaxWinSize",
]
X_df = df_clean[P4_FEATURE_COLS].astype(int)

# Features and labels
feature_cols = X_df.columns.tolist()
X = X_df.values                           # shape: (n_samples, 8)
y = df_clean[label_col].values            # string labels

print("Samples after clean:", X.shape[0])
print("Features:", X.shape[1])
print("Example labels:", np.unique(y)[:10])
if USER_N_COMPONENTS is not None:
    print("Using PCA components (specified):", USER_N_COMPONENTS)
else:
    # If not set, use n_features-1 (or 1 if only one feature)
    n_features = X.shape[1]
    k_auto = max(1, n_features - 1)
    print(f"PCA components not specified; using n_features-1: {k_auto}")

# ==========================================================
# 2. Train/test split (NO NORMALIZATION)
# ==========================================================
# CRITICAL: Use RAW features everywhere for consistency
# PCA will be fit on raw features, and DecisionTreeRegressor will map
# raw features -> quantized PCA codes. No normalization needed.
X_train = X
X_test = X
y_train = y
y_test = y

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ==========================================================
# 3. PCA (auto-select components if not provided), fit on RAW train, apply to both
# ===========================================================
if USER_N_COMPONENTS is None:
    # Use n_features-1 (or 1 if only one feature)
    n_features = X_train.shape[1]
    k_auto = max(1, n_features - 1)
    pca = PCA(n_components=k_auto, random_state=42)
else:
    pca = PCA(n_components=USER_N_COMPONENTS, random_state=42)

PC_train = pca.fit_transform(X_train)   # Fit on RAW features
PC_test = pca.transform(X_test)         # Transform RAW features

k = PC_train.shape[1]
print("PC_train shape:", PC_train.shape)
print("PC_test shape:", PC_test.shape)

# ...existing code...

# ==========================================================
# 4. Quantize PCA -> non-negative int (up to BITS bits)
# ==========================================================
pc_min = PC_train.min(axis=0)       # shape (k,)
pc_max = PC_train.max(axis=0)       # shape (k,)
pc_range = pc_max - pc_min
pc_range_safe = np.where(pc_range == 0, 1, pc_range)

def encode_pc(PC_float, pc_min, pc_range_safe, max_val):
    PC_float = np.asarray(PC_float)
    assert PC_float.ndim == 2 and PC_float.shape[1] == len(pc_min), \
        f"encode_pc expects (N,{len(pc_min)}), got {PC_float.shape}"
    pc_min_vec = np.asarray(pc_min).reshape(1, -1)
    pc_range_vec = np.asarray(pc_range_safe).reshape(1, -1)
    PC_norm = (PC_float - pc_min_vec) / pc_range_vec
    PC_code = np.rint(PC_norm * max_val)
    PC_code = np.clip(PC_code, 0, max_val)
    return PC_code.astype(int)

def decode_pc(PC_code, pc_min, pc_range_safe, max_val):
    PC_code = np.asarray(PC_code)
    assert PC_code.ndim == 2 and PC_code.shape[1] == len(pc_min), \
        f"decode_pc expects (N,{len(pc_min)}), got {PC_code.shape}"
    pc_min_vec = np.asarray(pc_min).reshape(1, -1)
    pc_range_vec = np.asarray(pc_range_safe).reshape(1, -1)
    return (PC_code / max_val) * pc_range_vec + pc_min_vec

PC_code_train = encode_pc(PC_train, pc_min, pc_range_safe, MAX_VAL)
PC_code_test = encode_pc(PC_test, pc_min, pc_range_safe, MAX_VAL)

print("PC_code_train shape:", PC_code_train.shape)
for j in range(k):
    print(f"PC{j+1}_code_train range: {PC_code_train[:, j].min()} -> {PC_code_train[:, j].max()}")

# Quantization approximation (sanity check)
PC_train_quant_approx = decode_pc(PC_code_train, pc_min, pc_range_safe, MAX_VAL)
PC_test_quant_approx = decode_pc(PC_code_test, pc_min, pc_range_safe, MAX_VAL)

r2_quant = []
for j in range(k):
    r2 = r2_score(PC_test[:, j], PC_test_quant_approx[:, j])
    r2_quant.append(r2)
print("\nR2 per component (test, original vs quantized):")
for j, r2 in enumerate(r2_quant, 1):
    print(f"PC{j}: {r2}")

# ==========================================================
# 5. Train DecisionTreeRegressor for mapping RAW features -> PCA codes (multi-output)
#    Both PCA and tree use RAW features, so P4 rules will work correctly
#    Use deeper tree for better PCA approximation
# ==========================================================
tree = DecisionTreeRegressor(
    max_depth=12,          # Deeper tree for better resolution
    min_samples_split=2,   # Allow more splits
    min_samples_leaf=1,    # Allow single-sample leaves for precision
    random_state=42
)

# Fit tree on RAW features -> quantized PCA codes
tree.fit(X_train, PC_code_train)

print("\ntree.n_outputs_:", tree.n_outputs_)
print("tree.tree_.value.shape:", tree.tree_.value.shape)  # (n_nodes, 1, k)

# ==========================================================
# 6. Build leaf -> representative PC_code vector (size k) using TRAIN (raw features)
# ==========================================================
leaf_ids_train = tree.apply(X_train)   # leaf node id for each train sample
unique_leaf_ids = np.unique(leaf_ids_train)
print("Number of leaf nodes:", unique_leaf_ids.shape[0])

leaf_to_codes = {}
for leaf_id in unique_leaf_ids:
    idx = np.where(leaf_ids_train == leaf_id)[0]
    codes_in_leaf = PC_code_train[idx]        # (n_leaf_samples, k)
    rep = np.rint(codes_in_leaf.mean(axis=0)).astype(int)  # (k,)
    leaf_to_codes[leaf_id] = rep

# ==========================================================
# 7. Use tree + leaf mapping to approximate PCA codes for TRAIN and TEST (using RAW features)
# ==========================================================
def get_tree_codes(dt, leaf_to_codes, X, k):
    leaf_ids = dt.apply(X)
    codes = np.zeros((X.shape[0], k), dtype=int)
    for i, leaf_id in enumerate(leaf_ids):
        codes[i, :] = leaf_to_codes[leaf_id]
    return codes

PC_code_tree_train = get_tree_codes(tree, leaf_to_codes, X_train, k)
PC_code_tree_test = get_tree_codes(tree, leaf_to_codes, X_test, k)

# Decode to approximate PCA
PC_train_tree_approx = decode_pc(PC_code_tree_train, pc_min, pc_range_safe, MAX_VAL)
PC_test_tree_approx = decode_pc(PC_code_tree_test, pc_min, pc_range_safe, MAX_VAL)

# ==========================================================
# 8. Covariance & correlation checks
# ==========================================================
def corr(x, y):
    return np.corrcoef(x, y)[0, 1]

print("\n=== Correlation (test set) ===")
for j in range(k):
    c_q = corr(PC_test[:, j], PC_test_quant_approx[:, j])
    c_t = corr(PC_test[:, j], PC_test_tree_approx[:, j])
    print(f"PC{j+1} original vs quantized   : {c_q}")
    print(f"PC{j+1} original vs tree-approx : {c_t}")

print("\n=== Covariance (test set, component-wise) ===")
for j in range(k):
    cov_q = np.cov(PC_test[:, j], PC_test_quant_approx[:, j])
    cov_t = np.cov(PC_test[:, j], PC_test_tree_approx[:, j])
    print(f"\nPC{j+1} Cov(original, quantized):\n{cov_q}")
    print(f"PC{j+1} Cov(original, tree-approx):\n{cov_t}")

print("\n=== Full covariance matrices (test) ===")
print("Covariance of original PCA (test):\n", np.cov(PC_test.T))
print("Covariance of tree-approx PCA (test):\n", np.cov(PC_test_tree_approx.T))

# ==========================================================
# 9. Classification with DecisionTreeClassifier
#    Compare 4 feature sets:
#    - original features (no PCA)
#    - original PCA float
#    - quantized PCA codes
#    - tree-approx PCA codes
# ==========================================================
# Direct classification on original features (no PCA)
clf_raw = DecisionTreeClassifier(
    max_depth=100,
    random_state=42
)
clf_raw.fit(X_train, y_train)
y_pred_raw = clf_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

clf_pca = DecisionTreeClassifier(
    max_depth=100,
    random_state=42
)
clf_pca.fit(PC_train, y_train)
y_pred_pca = clf_pca.predict(PC_test)
acc_pca = accuracy_score(y_test, y_pred_pca)

clf_code = DecisionTreeClassifier(
    max_depth=100,
    random_state=42
)
clf_code.fit(PC_code_train, y_train)
y_pred_code = clf_code.predict(PC_code_test)
acc_code = accuracy_score(y_test, y_pred_code)

clf_tree_code = DecisionTreeClassifier(
    max_depth=100,
    random_state=42
)
clf_tree_code.fit(PC_code_tree_train, y_train)
y_pred_tree_code = clf_tree_code.predict(PC_code_tree_test)
acc_tree_code = accuracy_score(y_test, y_pred_tree_code)

print("\n=== DecisionTreeClassifier accuracy (test) ===")
print("Using original raw features   :", acc_raw)
print("Using original PCA (float)    :", acc_pca)
print("Using quantized PCA codes     :", acc_code)
print("Using tree-approx PCA codes   :", acc_tree_code)

# ==========================================================
# 9.1 Detailed metrics: confusion matrix, precision, recall, F1
# ==========================================================
from sklearn.metrics import confusion_matrix, classification_report

labels = sorted(np.unique(y))
metrics = {
    "labels": labels,
    "raw_features": {
        "accuracy": float(acc_raw),
        "classification_report": classification_report(y_test, y_pred_raw, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_raw, labels=labels).tolist(),
    },
    "original_pca": {
        "accuracy": float(acc_pca),
        "classification_report": classification_report(y_test, y_pred_pca, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_pca, labels=labels).tolist(),
    },
    "quantized_codes": {
        "accuracy": float(acc_code),
        "classification_report": classification_report(y_test, y_pred_code, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_code, labels=labels).tolist(),
    },
    "tree_approx_codes": {
        "accuracy": float(acc_tree_code),
        "classification_report": classification_report(y_test, y_pred_tree_code, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_tree_code, labels=labels).tolist(),
    },
}

metrics_path = os.path.join(TABLES_DIR, "pca_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved detailed metrics to:", metrics_path)

# ==========================================================
# 10. Save mapping PCA_float + PCA_code + Label (for all samples)
# ==========================================================
PC_all = np.vstack([PC_train, PC_test])          # (N_all, k)
# CRITICAL: use tree-approximated codes here, NOT the exact quantized PCA codes.
# The P4 switch produces codes via the regressor-tree lookup (PC_code_tree_train),
# NOT by directly quantizing PCA scores (PC_code_train).
# The DT classifier (step 3) must be trained on the same codes the switch will produce,
# otherwise ml_code table ranges will never match the runtime switch output.
PC_code_tree_all = np.vstack([PC_code_tree_train, PC_code_tree_test])
y_all = np.concatenate([y_train, y_test])

mapping_dict = {}
for j in range(k):
    mapping_dict[f"PC{j+1}_float"] = PC_all[:, j]
for j in range(k):
    mapping_dict[f"PC{j+1}_code"] = PC_code_tree_all[:, j]   # tree-approx codes ← matches P4 switch
mapping_dict["Label"] = y_all

mapping_df = pd.DataFrame(mapping_dict)

mapping_csv_path = os.path.join(TABLES_DIR, "pca_integer_mapping.csv")
mapping_df.to_csv(mapping_csv_path, index=False)
print(f"\nMapping PCA -> integer + Label saved to: {mapping_csv_path}")

# ==========================================================
# 11. Save encoding parameters
# ==========================================================
encoding_params = {
    "n_components": int(k),
    "bits": int(BITS),
    "max_val": int(MAX_VAL),
    "pc_min": pc_min.tolist(),
    "pc_max": pc_max.tolist(),
    "pc_range": pc_range.tolist(),
    "auto_selected": bool(USER_N_COMPONENTS is None),
    "variance_target": float(VAR_TARGET) if USER_N_COMPONENTS is None else None
}

params_path = os.path.join(TABLES_DIR, "pca_encoding_params.json")
with open(params_path, "w") as f:
    json.dump(encoding_params, f, indent=2)

print(f"Encoding parameters saved to: {params_path}")

# ==========================================================
# 12. Export P4 table_add commands for PCA transformation
#     Maps raw feature ranges to each PCA component code
#     Generates separate commands for each PCA component (scalable design)
# ==========================================================
def minimize(path):
    """Aggregate min/max constraints for each feature along a path."""
    domain = {}
    for (feature, sign, threshold) in path:
        if feature not in domain:
            domain[feature] = {"min": None, "max": None}
        val = domain[feature]
        if sign == "<=":
            val["max"] = threshold
        else:
            val["min"] = threshold
    return domain

def format_feature_ranges(domain, feature_names, feature_max_map):
    """Format feature constraints as P4 match clauses.
    IMPORTANT: All features MUST be present, in the correct order!
    - All features use RANGE match syntax (lo->hi)
    - For flag fields: 0->0 (only 0), 1->1 (only 1), 0->1 (wildcard/both)
    - For numeric features: standard range matching
    Unconstrained range features get full range (0->FEATURE_MAX)
    Unconstrained flag features get wildcard (0->1)"""
    clauses = []
    
    # Define which features are flags (binary, 0 or 1)
    flag_features = {"FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst"}
    
    for feat_name in feature_names:
        is_flag = feat_name in flag_features
        
        if feat_name not in domain:
            # Feature not constrained
            if is_flag:
                # For flags: unconstrained means wildcard (both 0 and 1)
                clauses.append("0->1")
            else:
                # For numeric: unconstrained means full range
                feature_max = feature_max_map.get(feat_name, MAX_VAL)
                clauses.append(f"0->{feature_max}")
            continue
            
        val = domain[feat_name]
        lo = val["min"]
        hi = val["max"]
        
        if is_flag:
            # For binary flags: thresholds are typically 0.5
            # lo=None, hi=0.5 means feature <= 0.5, so value is 0 (range: 0->0)
            # lo=0.5, hi=None means feature > 0.5, so value is 1 (range: 1->1)
            # Both bounds set shouldn't happen for binary features
            
            if lo is None and hi is not None:
                # Feature <= threshold (value is 0)
                clauses.append("0->0")
            elif lo is not None and hi is None:
                # Feature > threshold (value is 1)
                clauses.append("1->1")
            else:
                # Unconstrained or both bounds (shouldn't happen)
                clauses.append("0->1")
        else:
            # For numeric features, standard range conversion
            feature_max = feature_max_map.get(feat_name)
            if feature_max is None:
                feature_max = int(np.nanmax(X_df[feat_name])) if feat_name in X_df.columns else MAX_VAL

            if lo is None:
                lo = -1  # will become 0 after +1
            if hi is None:
                hi = feature_max

            # Decision tree uses (lo, hi]; convert to [lo+1, hi]
            lo = int(lo) + 1
            hi = int(hi)

            # Clamp to feature range
            if lo < 0:
                lo = 0
            if hi > feature_max:
                hi = feature_max

            clauses.append(f"{lo}->{hi}")
    
    return clauses

def write_pca_p4_commands(dt, feature_names, leaf_to_codes, k, max_val, feature_max_map, filename):
    """
    Generate P4 table_add commands for PCA transformation tables.
    Creates one command per PCA component per leaf node.
    Scalable: works with any number of PCA components and bit widths.
    """
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    features = tree_.feature
    
    priority = [0]  # mutable container for priority counter

    def dfs(node_id, path):
        is_leaf = (left[node_id] == right[node_id])

        if is_leaf:
            priority[0] += 1
            new_path = []
            for (n_id, sign) in path:
                feat_idx = features[n_id]
                if feat_idx == -2:
                    continue
                feat_name = feature_names[feat_idx]
                thr = threshold[n_id]
                new_path.append((feat_name, sign, thr))

            clause = minimize(new_path)
            clauses = format_feature_ranges(clause, feature_names, feature_max_map)
            codes = leaf_to_codes[node_id]  # shape (k,)
            
            # Generate one table_add per PCA component
            for j in range(k):
                code_val = int(codes[j])
                clause_str = ' '.join(clauses) if clauses else ''
                f.write(f"table_add MyIngress.pca_component{j+1} set_pc{j+1}_code {clause_str} => {code_val} {priority[0]}\n")

            
            return

        # Left (<=)
        new_path_left = path.copy()
        new_path_left.append((node_id, "<="))
        dfs(left[node_id], new_path_left)

        # Right (>)
        new_path_right = path.copy()
        new_path_right.append((node_id, ">"))
        dfs(right[node_id], new_path_right)

    with open(filename, "w") as f:
        dfs(0, [])

# Build feature max values based on P4 field widths (fallback to dataset max)
FEATURE_MAX_DEFAULTS = {
    "Duration":    (2**48 - 1),
    "MaxIAT":      (2**48 - 1),
    "UrgCount":    (2**32 - 1),
    "FwdPktCount": (2**32 - 1),
    "BwdPktCount": (2**32 - 1),
    "FwdBytes":    (2**32 - 1),
    "BwdBytes":    (2**32 - 1),
    "MaxWinSize":  (2**16 - 1),
}

feature_max_map = {}
for feat in feature_cols:
    if feat in FEATURE_MAX_DEFAULTS:
        feature_max_map[feat] = FEATURE_MAX_DEFAULTS[feat]
    else:
        feature_max_map[feat] = int(np.nanmax(X_df[feat])) if feat in X_df.columns else MAX_VAL

# Generate P4 commands for PCA transformation
pca_commands_path = os.path.join(TABLES_DIR, "s1-commands.txt")
write_pca_p4_commands(tree, feature_cols, leaf_to_codes, k, MAX_VAL, feature_max_map, pca_commands_path)
print(f"P4 table_add commands for PCA components saved to: {pca_commands_path}")

# Also keep IF rules for human readability
def write_if_rules_for_pca(dt, feature_names, leaf_to_codes, filename):
    """
    Write IF-THEN rules in human-readable format.
    Rules show: IF (feature ranges) THEN (PCA component codes)
    """
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    features = tree_.feature

    def dfs(node_id, path):
        is_leaf = (left[node_id] == right[node_id])

        if is_leaf:
            clauses = []
            for (n_id, sign) in path:
                feat_idx = features[n_id]
                if feat_idx == -2:
                    continue
                feat_name = feature_names[feat_idx]
                thr = threshold[n_id]
                clauses.append(f"{feat_name} {sign} {thr:.2f}")

            condition_str = " AND ".join(clauses) if clauses else "TRUE"

            codes = leaf_to_codes[node_id]  # shape (k,)
            assigns = " AND ".join(
                [f"PC{j+1}_code = {int(codes[j])}" for j in range(len(codes))]
            )

            rule = f"IF ({condition_str}) THEN {assigns};\n"
            f.write(rule)
            return

        # Left (<=)
        new_path_left = path.copy()
        new_path_left.append((node_id, "<="))
        dfs(left[node_id], new_path_left)

        # Right (>)
        new_path_right = path.copy()
        new_path_right.append((node_id, ">"))
        dfs(right[node_id], new_path_right)

    with open(filename, "w") as f:
        f.write("# PCA Transformation Rules (IF-THEN)\n")
        f.write("# Maps 3 raw features to PCA component codes\n\n")
        dfs(0, [])

rules_if_path = os.path.join(TABLES_DIR, "pca_tree_if_rules.txt")
write_if_rules_for_pca(tree, feature_cols, leaf_to_codes, rules_if_path)
print(f"IF rules (feature ranges -> PC*_code) saved to: {rules_if_path}")