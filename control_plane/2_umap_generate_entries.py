#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import os
import glob
import argparse
import sys
import pickle

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def _load_umap_class():
    try:
        import importlib.util
        import os
        import types
        import sys as _sys
        spec = importlib.util.find_spec("umap")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("umap package not found")
        umap_pkg_dir = spec.submodule_search_locations[0]
        umap_path = os.path.join(umap_pkg_dir, "umap_.py")
        if not os.path.exists(umap_path):
            raise ImportError("umap_.py not found in umap package")
        # Create a stub 'umap' package to avoid executing umap/__init__.py
        if "umap" not in _sys.modules:
            umap_pkg = types.ModuleType("umap")
            umap_pkg.__path__ = [umap_pkg_dir]
            _sys.modules["umap"] = umap_pkg
        mod_spec = importlib.util.spec_from_file_location("umap_umap_", umap_path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        return module.UMAP
    except Exception as e:
        print("ERROR: Failed to import core UMAP without TensorFlow.")
        print("  This can happen if 'umap' tries to import TensorFlow parametric UMAP.")
        print("  Ensure umap-learn is installed and TensorFlow is not required.")
        print(f"  Details: {e}")
        raise


UMAP = _load_umap_class()


# ==========================================================
# 0. CONFIG / CLI
# ==========================================================
def parse_args():
    parser = P4secArgumentParser(
        description="UMAP pipeline with quantization (works with any classifier in step 3)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Outputs:\n"
            "  tables/s1-commands.txt           (P4 transform entries)\n"
            "  tables/pca_encoding_params.json  (transform params; legacy filename used by all methods)\n"
            "  tables/pca_integer_mapping.csv   (codes + labels; legacy filename used by all methods)\n"
            "  tables/reduction_config.json     (universal config)\n"
            "Code prefix: UM*_code\n"
        )
    )
    parser.add_argument("--components", "-k", type=int, default=2,
                        help="Number of UMAP components (default: 2)")
    parser.add_argument("--bits", "-b", type=int, default=16,
                        help="Quantization bits for UMAP codes (default: 16). Supports 8, 16, 24, 32 bits.")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric", type=str, default="euclidean",
                        help="UMAP metric (default: euclidean)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for UMAP (default: 42)")
    return parser.parse_args()

args = parse_args()

USER_N_COMPONENTS = args.components
BITS = args.bits

# Validate BITS parameter
if BITS not in [8, 16, 24, 32]:
    print(f"WARNING: BITS={BITS}. Recommended values are 8, 16, 24, 32 for P4 compatibility.")
    print("P4 range match can handle arbitrary widths, but ensure switch pipeline width supports it.")

MAX_VAL = 2**BITS - 1

# ==========================================================
# 1. Load dataset & cleaning
# ==========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

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
        preferred = [p for p in csvs if os.path.basename(p).lower() == "dataset.csv"]
        if preferred:
            return preferred[0]
        csvs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return csvs[0]
    raise FileNotFoundError("No CSV dataset found in any dataset/ directory")

csv_path = find_dataset_csv()
print("Using dataset:", csv_path)
df = pd.read_csv(csv_path)

label_col = df.columns[-1]
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

# P4 feature order
P4_FEATURE_COLS = [
    "Protocol",
    "Duration", "MaxIAT", "UrgCount",
    "FwdPktCount", "BwdPktCount",
    "FwdBytes", "BwdBytes",
    "MaxWinSize",
    "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst",
]
X_df = df_clean[P4_FEATURE_COLS].astype(int)
feature_cols = X_df.columns.tolist()
X = X_df.values
y = df_clean[label_col].values

print("Samples after clean:", X.shape[0])
print("Features:", X.shape[1])
print("Example labels:", np.unique(y)[:10])
print("Using UMAP components:", USER_N_COMPONENTS)

# ==========================================================
# 2. Train/test split (no normalization; consistent with pipeline)
# ==========================================================
X_train = X
X_test = X
y_train = y
y_test = y

# ==========================================================
# 3. UMAP fit + transform (raw features)
# ==========================================================
umap_model = UMAP(
    n_components=USER_N_COMPONENTS,
    n_neighbors=args.n_neighbors,
    min_dist=args.min_dist,
    metric=args.metric,
    random_state=args.random_state,
    transform_seed=args.random_state,
)

UM_train = umap_model.fit_transform(X_train)
UM_test = umap_model.transform(X_test)

k = UM_train.shape[1]
print("UM_train shape:", UM_train.shape)
print("UM_test shape:", UM_test.shape)

# ==========================================================
# 4. Quantize UMAP -> non-negative int (up to BITS bits)
# ==========================================================
um_min = UM_train.min(axis=0)
um_max = UM_train.max(axis=0)
um_range = um_max - um_min
um_range_safe = np.where(um_range == 0, 1, um_range)

def encode_um(UM_float, um_min, um_range_safe, max_val):
    UM_float = np.asarray(UM_float)
    assert UM_float.ndim == 2 and UM_float.shape[1] == len(um_min), \
        f"encode_um expects (N,{len(um_min)}), got {UM_float.shape}"
    um_min_vec = np.asarray(um_min).reshape(1, -1)
    um_range_vec = np.asarray(um_range_safe).reshape(1, -1)
    UM_norm = (UM_float - um_min_vec) / um_range_vec
    UM_code = np.rint(UM_norm * max_val)
    UM_code = np.clip(UM_code, 0, max_val)
    return UM_code.astype(int)

def decode_um(UM_code, um_min, um_range_safe, max_val):
    UM_code = np.asarray(UM_code)
    assert UM_code.ndim == 2 and UM_code.shape[1] == len(um_min), \
        f"decode_um expects (N,{len(um_min)}), got {UM_code.shape}"
    um_min_vec = np.asarray(um_min).reshape(1, -1)
    um_range_vec = np.asarray(um_range_safe).reshape(1, -1)
    return (UM_code / max_val) * um_range_vec + um_min_vec

UM_code_train = encode_um(UM_train, um_min, um_range_safe, MAX_VAL)
UM_code_test = encode_um(UM_test, um_min, um_range_safe, MAX_VAL)

print("UM_code_train shape:", UM_code_train.shape)
for j in range(k):
    print(f"UM{j+1}_code_train range: {UM_code_train[:, j].min()} -> {UM_code_train[:, j].max()}")

# ==========================================================
# 5. Train DecisionTreeRegressor for mapping RAW -> UMAP codes (multi-output)
# ==========================================================
tree = DecisionTreeRegressor(
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
tree.fit(X_train, UM_code_train)

# ==========================================================
# 6. Build leaf -> representative UM_code vector (size k)
# ==========================================================
leaf_ids_train = tree.apply(X_train)
unique_leaf_ids = np.unique(leaf_ids_train)
print("Number of leaf nodes:", unique_leaf_ids.shape[0])

leaf_to_codes = {}
for leaf_id in unique_leaf_ids:
    idx = np.where(leaf_ids_train == leaf_id)[0]
    codes_in_leaf = UM_code_train[idx]
    rep = np.rint(codes_in_leaf.mean(axis=0)).astype(int)
    leaf_to_codes[leaf_id] = rep

def get_tree_codes(dt, leaf_to_codes, X, k):
    leaf_ids = dt.apply(X)
    codes = np.zeros((X.shape[0], k), dtype=int)
    for i, leaf_id in enumerate(leaf_ids):
        codes[i, :] = leaf_to_codes[leaf_id]
    return codes

UM_code_tree_train = get_tree_codes(tree, leaf_to_codes, X_train, k)
UM_code_tree_test = get_tree_codes(tree, leaf_to_codes, X_test, k)

# ==========================================================
# 7. Classification comparison
# ==========================================================
clf_raw = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_raw.fit(X_train, y_train)
y_pred_raw = clf_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

clf_um = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_um.fit(UM_train, y_train)
y_pred_um = clf_um.predict(UM_test)
acc_um = accuracy_score(y_test, y_pred_um)

clf_code = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_code.fit(UM_code_train, y_train)
y_pred_code = clf_code.predict(UM_code_test)
acc_code = accuracy_score(y_test, y_pred_code)

clf_tree_code = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_tree_code.fit(UM_code_tree_train, y_train)
y_pred_tree_code = clf_tree_code.predict(UM_code_tree_test)
acc_tree_code = accuracy_score(y_test, y_pred_tree_code)

print("\n=== DecisionTreeClassifier accuracy (test) ===")
print("Using original raw features   :", acc_raw)
print("Using UMAP (float)            :", acc_um)
print("Using quantized UMAP codes    :", acc_code)
print("Using tree-approx UMAP codes  :", acc_tree_code)

labels = sorted(np.unique(y))
metrics = {
    "labels": labels,
    "raw_features": {
        "accuracy": float(acc_raw),
        "classification_report": classification_report(y_test, y_pred_raw, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_raw, labels=labels).tolist(),
    },
    "umap_float": {
        "accuracy": float(acc_um),
        "classification_report": classification_report(y_test, y_pred_um, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_um, labels=labels).tolist(),
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

metrics_path = os.path.join(TABLES_DIR, "umap_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved detailed metrics to:", metrics_path)

# ==========================================================
# 8. Save mapping UMAP_float + UMAP_code + Label (for all samples)
# ==========================================================
UM_all = np.vstack([UM_train, UM_test])
UM_code_tree_all = np.vstack([UM_code_tree_train, UM_code_tree_test])
y_all = np.concatenate([y_train, y_test])

mapping_dict = {}
for j in range(k):
    mapping_dict[f"UM{j+1}_float"] = UM_all[:, j]
for j in range(k):
    mapping_dict[f"UM{j+1}_code"] = UM_code_tree_all[:, j]
mapping_dict["Label"] = y_all

mapping_df = pd.DataFrame(mapping_dict)
mapping_csv_path = os.path.join(TABLES_DIR, "pca_integer_mapping.csv")
mapping_df.to_csv(mapping_csv_path, index=False)
print(f"\nMapping UMAP -> integer + Label saved to: {mapping_csv_path}")

# ==========================================================
# 9. Save encoding parameters + models
# ==========================================================
umap_model_path = None
umap_tree_path = os.path.join(TABLES_DIR, "umap_tree.pkl")
try:
    tmp_model_path = os.path.join(TABLES_DIR, "umap_model.pkl")
    with open(tmp_model_path, "wb") as f:
        pickle.dump(umap_model, f)
    umap_model_path = tmp_model_path
except Exception as e:
    print(f"WARNING: Could not pickle UMAP model (will rely on tree only): {e}")

with open(umap_tree_path, "wb") as f:
    pickle.dump(tree, f)

encoding_params = {
    "method": "umap",
    "n_components": int(k),
    "bits": int(BITS),
    "max_val": int(MAX_VAL),
    "pc_min": um_min.tolist(),
    "pc_max": um_max.tolist(),
    "pc_range": um_range.tolist(),
    "umap_params": {
        "n_neighbors": int(args.n_neighbors),
        "min_dist": float(args.min_dist),
        "metric": args.metric,
        "random_state": int(args.random_state),
    },
    "umap_tree_path": os.path.relpath(umap_tree_path, os.path.dirname(__file__)),
}
if umap_model_path:
    encoding_params["umap_model_path"] = os.path.relpath(umap_model_path, os.path.dirname(__file__))

params_path = os.path.join(TABLES_DIR, "pca_encoding_params.json")
with open(params_path, "w") as f:
    json.dump(encoding_params, f, indent=2)
print(f"Encoding parameters saved to: {params_path}")

# ==========================================================
# 9b. Save universal reduction_config.json
# ==========================================================
feature_columns = [f"UM{j+1}_code" for j in range(k)]
feature_max_values = {col: int(MAX_VAL) for col in feature_columns}

reduction_config = {
    "method": "umap",
    "feature_columns": feature_columns,
    "feature_max_values": feature_max_values,
    "needs_transform_tables": True,
    "n_components": int(k),
    "bits": int(BITS),
}
reduction_config_path = os.path.join(TABLES_DIR, "reduction_config.json")
with open(reduction_config_path, "w") as f:
    json.dump(reduction_config, f, indent=2)
print(f"Reduction config saved to: {reduction_config_path}")

# ==========================================================
# 10. Export P4 table_add commands for UMAP transformation
# ==========================================================
def minimize(path):
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
    clauses = []
    for feat_name in feature_names:
        feature_max = feature_max_map.get(feat_name)
        if feature_max is None:
            feature_max = int(np.nanmax(X_df[feat_name])) if feat_name in X_df.columns else MAX_VAL
        if feat_name not in domain:
            clauses.append(f"0->{feature_max}")
            continue
        val = domain[feat_name]
        lo = val["min"]
        hi = val["max"]
        if lo is None:
            lo = -1
        if hi is None:
            hi = feature_max
        lo = int(lo) + 1
        hi = int(hi)
        if lo < 0:
            lo = 0
        if hi > feature_max:
            hi = feature_max
        clauses.append(f"{lo}->{hi}")
    return clauses

def write_umap_p4_commands(dt, feature_names, leaf_to_codes, k, max_val, feature_max_map, filename):
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    features = tree_.feature

    priority = [0]

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
            codes = leaf_to_codes[node_id]

            for j in range(k):
                code_val = int(codes[j])
                clause_str = ' '.join(clauses) if clauses else ''
                f.write(f"table_add MyIngress.umap_component{j+1} set_um{j+1}_code {clause_str} => {code_val} {priority[0]}\n")
            return

        new_path_left = path.copy()
        new_path_left.append((node_id, "<="))
        dfs(left[node_id], new_path_left)

        new_path_right = path.copy()
        new_path_right.append((node_id, ">"))
        dfs(right[node_id], new_path_right)

    with open(filename, "w") as f:
        dfs(0, [])

FEATURE_MAX_DEFAULTS = {
    "Protocol":    (2**8  - 1),
    "Duration":    (2**48 - 1),
    "MaxIAT":      (2**48 - 1),
    "UrgCount":    (2**32 - 1),
    "FwdPktCount": (2**32 - 1),
    "BwdPktCount": (2**32 - 1),
    "FwdBytes":    (2**32 - 1),
    "BwdBytes":    (2**32 - 1),
    "MaxWinSize":  (2**16 - 1),
    "FlagsSyn":    (2**32 - 1),
    "FlagsAck":    (2**32 - 1),
    "FlagsFin":    (2**32 - 1),
    "FlagsRst":    (2**32 - 1),
}

feature_max_map = {}
for feat in feature_cols:
    if feat in FEATURE_MAX_DEFAULTS:
        feature_max_map[feat] = FEATURE_MAX_DEFAULTS[feat]
    else:
        feature_max_map[feat] = int(np.nanmax(X_df[feat])) if feat in X_df.columns else MAX_VAL

umap_commands_path = os.path.join(TABLES_DIR, "s1-commands.txt")
write_umap_p4_commands(tree, feature_cols, leaf_to_codes, k, MAX_VAL, feature_max_map, umap_commands_path)
print(f"P4 table_add commands for UMAP components saved to: {umap_commands_path}")

def write_if_rules_for_umap(dt, feature_names, leaf_to_codes, filename):
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
            codes = leaf_to_codes[node_id]
            assigns = " AND ".join(
                [f"UM{j+1}_code = {int(codes[j])}" for j in range(len(codes))]
            )
            rule = f"IF ({condition_str}) THEN {assigns};\n"
            f.write(rule)
            return

        new_path_left = path.copy()
        new_path_left.append((node_id, "<="))
        dfs(left[node_id], new_path_left)

        new_path_right = path.copy()
        new_path_right.append((node_id, ">"))
        dfs(right[node_id], new_path_right)

    with open(filename, "w") as f:
        f.write("# UMAP Transformation Rules (IF-THEN)\n")
        f.write("# Maps raw features to UMAP component codes\n\n")
        dfs(0, [])

rules_if_path = os.path.join(TABLES_DIR, "umap_tree_if_rules.txt")
write_if_rules_for_umap(tree, feature_cols, leaf_to_codes, rules_if_path)
print(f"IF rules (feature ranges -> UM*_code) saved to: {rules_if_path}")

print()
print("=" * 60)
print("IMPORTANT: s1-commands.txt now contains ONLY UMAP transform entries.")
print("You MUST re-run the unified classifier pipeline next:")
print("  python3 3_train_model.py --model-type dt")
print("  python3 3_train_model.py --model-type rf")
print("  python3 3_train_model.py --model-type xgb")
print("  python3 3_train_model.py --model-type gb")
print("  python3 3_train_model.py --model-type knn")
print("  python3 3_train_model.py --model-type svm")
print("  python3 3_train_model.py --model-type cnn")
print("Then run the matching 4_generate_model_entries.py and 5_generating_p4_code.py steps.")
print("Skipping the classifier-entry regeneration step will result in all flows")
print("being classified as class 0 (Benign) because model table entries are absent.")
print("=" * 60)
