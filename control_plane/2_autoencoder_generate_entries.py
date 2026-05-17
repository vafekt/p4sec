#!/usr/bin/env python3
"""
Autoencoder + surrogate DT pipeline with quantization.

Inspired by Planter's bottleneck autoencoder flow: learn a compact latent
representation from the raw flow features, quantize that latent space, and
deploy a surrogate DecisionTreeRegressor so the P4 program can map raw flow
features to integer latent codes using range-match tables.

Outputs (same locations, compatible with steps 3/4/5/6):
  tables/transform_mapping.csv       Columns: AE1_float...AEk_float, AE1_code...AEk_code, Label
  tables/encoding_params.json        Encoding metadata (scaler, encoder weights, min/max/range)
  tables/reduction_config.json       Universal config for steps 3/4/5/6
  tables/s1-commands.txt             P4 table_add commands for transform tables
  tables/transform_rules.txt         Human-readable surrogate tree rules
  tables/transform_metrics.json      Reconstruction + classification metrics
"""

import argparse
import sys
import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline_utils import P4_FEATURE_MAX, find_dataset_csv

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


def parse_args():
    parser = P4secArgumentParser(
        description="Autoencoder pipeline with quantization",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Outputs:\n"
            "  tables/s1-commands.txt           (P4 transform entries)\n"
            "  tables/encoding_params.json      (transform params; shared filename used by all methods)\n"
            "  tables/transform_mapping.csv     (codes + labels; shared filename used by all methods)\n"
            "  tables/reduction_config.json     (universal config)\n"
            "Code prefix: AE*_code\n"
            "Note: Autoencoder codes can be used with any classifier in step 3.\n"
        )
    )
    parser.add_argument("--components", "-k", type=int, default=None,
                        help="Latent dimensions (default: min(6, n_features//2))")
    parser.add_argument("--bits", "-b", type=int, default=16,
                        help="Quantization bits for autoencoder codes (default: 16)")
    parser.add_argument("--activation", choices=["identity", "relu", "tanh", "logistic"],
                        default="tanh", help="Encoder hidden activation (default: tanh)")
    parser.add_argument("--solver", choices=["adam", "lbfgs"], default="adam",
                        help="MLPRegressor solver (default: adam)")
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help="L2 regularization strength (default: 1e-4)")
    parser.add_argument("--max-iter", type=int, default=800,
                        help="Autoencoder max training iterations (default: 800)")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


ARGS = parse_args()
USER_K = ARGS.components
BITS = ARGS.bits
ACTIVATION = ARGS.activation
SOLVER = ARGS.solver
ALPHA = ARGS.alpha
MAX_ITER = ARGS.max_iter
RANDOM_STATE = ARGS.random_state

if BITS not in [8, 16, 24, 32]:
    print(f"WARNING: BITS={BITS}. Recommended values are 8, 16, 24, 32.")

MAX_VAL = 2 ** BITS - 1

TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

P4_FEATURE_COLS = [
    "Protocol",
    "SrcPort", "DstPort",
    "Duration", "MaxIAT",
    "FwdPktCount", "BwdPktCount", "FwdBytes", "BwdBytes",
    "FwdMaxPktLen", "BwdMaxPktLen",
    "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst", "FlagsPsh",
    "MaxWinSize", "InitFwdWinBytes",
]




def hidden_activation(values, activation_name):
    values = np.asarray(values, dtype=np.float64)
    if activation_name == "identity":
        return values
    if activation_name == "relu":
        return np.maximum(values, 0.0)
    if activation_name == "tanh":
        return np.tanh(values)
    if activation_name == "logistic":
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    raise ValueError(f"Unsupported activation: {activation_name}")


def encode_latent(latent_float, latent_min, latent_range_safe, max_val):
    latent_float = np.asarray(latent_float)
    norm = (latent_float - latent_min.reshape(1, -1)) / latent_range_safe.reshape(1, -1)
    codes = np.clip(np.rint(norm * max_val), 0, max_val)
    return codes.astype(int)


def decode_latent(latent_code, latent_min, latent_range_safe, max_val):
    latent_code = np.asarray(latent_code)
    return (latent_code / max_val) * latent_range_safe.reshape(1, -1) + latent_min.reshape(1, -1)


csv_path = find_dataset_csv(__file__)
print("Using dataset:", csv_path)
df = pd.read_csv(csv_path)
label_col = df.columns[-1]
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

for _col in P4_FEATURE_COLS:
    _bad = pd.to_numeric(df_clean[_col], errors='coerce').isna()
    if _bad.any():
        print(f"WARNING: dropping {_bad.sum()} row(s) with non-numeric '{_col}': "
              f"{df_clean.loc[_bad, _col].values[:3]}")
        df_clean = df_clean[~_bad]

X_df = df_clean[P4_FEATURE_COLS].astype(int)
feature_cols = X_df.columns.tolist()
X = X_df.values.astype(np.float64)
y = df_clean[label_col].values

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print("Labels:", np.unique(y)[:10])

if USER_K is None:
    k = max(1, min(6, X.shape[1] // 2))
else:
    k = max(1, min(USER_K, X.shape[1] - 1 if X.shape[1] > 1 else 1))
    if USER_K != k:
        print(f"WARNING: requested {USER_K} components but using {k}.")

print(f"Autoencoder latent dimensions: {k}")

X_train = X_test = X
y_train = y_test = y

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

autoencoder = MLPRegressor(
    hidden_layer_sizes=(k,),
    activation=ACTIVATION,
    solver=SOLVER,
    alpha=ALPHA,
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE,
)
autoencoder.fit(X_scaled_train, X_scaled_train)

encoder_weights = np.asarray(autoencoder.coefs_[0], dtype=np.float64)
encoder_bias = np.asarray(autoencoder.intercepts_[0], dtype=np.float64)
decoder_weights = np.asarray(autoencoder.coefs_[1], dtype=np.float64)
decoder_bias = np.asarray(autoencoder.intercepts_[1], dtype=np.float64)

AE_train_linear = X_scaled_train @ encoder_weights + encoder_bias
AE_test_linear = X_scaled_test @ encoder_weights + encoder_bias
AE_train = hidden_activation(AE_train_linear, ACTIVATION)
AE_test = hidden_activation(AE_test_linear, ACTIVATION)

X_recon_train = AE_train @ decoder_weights + decoder_bias
X_recon_test = AE_test @ decoder_weights + decoder_bias
train_mse = mean_squared_error(X_scaled_train, X_recon_train)
test_mse = mean_squared_error(X_scaled_test, X_recon_test)

print(f"Latent train shape: {AE_train.shape}")
print(f"Reconstruction MSE (train): {train_mse:.6f}")
print(f"Reconstruction MSE (test) : {test_mse:.6f}")

ae_min = AE_train.min(axis=0)
ae_max = AE_train.max(axis=0)
ae_range = ae_max - ae_min
ae_range_safe = np.where(ae_range == 0, 1, ae_range)

AE_code_train = encode_latent(AE_train, ae_min, ae_range_safe, MAX_VAL)
AE_code_test = encode_latent(AE_test, ae_min, ae_range_safe, MAX_VAL)

for j in range(k):
    print(f"AE{j+1}_code range: {AE_code_train[:, j].min()} -> {AE_code_train[:, j].max()}")

AE_train_quant_approx = decode_latent(AE_code_train, ae_min, ae_range_safe, MAX_VAL)
for j in range(k):
    r2 = r2_score(AE_train[:, j], AE_train_quant_approx[:, j])
    print(f"AE{j+1} quantization R2: {r2:.6f}")

from sklearn.tree import DecisionTreeRegressor as _DTReg
tree = _DTReg(max_depth=None, min_samples_leaf=1, max_leaf_nodes=300000, random_state=RANDOM_STATE)
tree.fit(X_train, AE_code_train)

leaf_ids_train = tree.apply(X_train)
unique_leaf_ids = np.unique(leaf_ids_train)
print(f"Surrogate tree leaves: {len(unique_leaf_ids)}")

leaf_to_codes = {}
for leaf_id in unique_leaf_ids:
    idx = np.where(leaf_ids_train == leaf_id)[0]
    labels_in_leaf = y_train[idx]
    values, counts = np.unique(labels_in_leaf, return_counts=True)
    majority_class = values[np.argmax(counts)]
    mask = labels_in_leaf == majority_class
    leaf_to_codes[leaf_id] = np.rint(AE_code_train[idx][mask].mean(axis=0)).astype(int)


def get_tree_codes(dt, leaf_mapping, X_values, n_codes):
    leaf_ids = dt.apply(X_values)
    codes = np.zeros((X_values.shape[0], n_codes), dtype=int)
    for i, leaf_id in enumerate(leaf_ids):
        codes[i, :] = leaf_mapping[leaf_id]
    return codes


AE_code_tree_train = get_tree_codes(tree, leaf_to_codes, X_train, k)
AE_code_tree_test = get_tree_codes(tree, leaf_to_codes, X_test, k)

clf_raw = DecisionTreeClassifier(max_depth=100, random_state=RANDOM_STATE)
clf_raw.fit(X_train, y_train)
y_pred_raw = clf_raw.predict(X_test)
acc_raw = accuracy_score(y_test, y_pred_raw)

clf_ae = DecisionTreeClassifier(max_depth=100, random_state=RANDOM_STATE)
clf_ae.fit(AE_train, y_train)
y_pred_ae = clf_ae.predict(AE_test)
acc_ae = accuracy_score(y_test, y_pred_ae)

clf_code = DecisionTreeClassifier(max_depth=100, random_state=RANDOM_STATE)
clf_code.fit(AE_code_train, y_train)
y_pred_code = clf_code.predict(AE_code_test)
acc_code = accuracy_score(y_test, y_pred_code)

clf_tree = DecisionTreeClassifier(max_depth=100, random_state=RANDOM_STATE)
clf_tree.fit(AE_code_tree_train, y_train)
y_pred_tree = clf_tree.predict(AE_code_tree_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

print("\n=== Classification Accuracy ===")
print(f"Raw features         : {acc_raw}")
print(f"AE latent float      : {acc_ae}")
print(f"Quantized AE codes   : {acc_code}")
print(f"Tree-approx AE codes : {acc_tree}")

labels = sorted(np.unique(y))
metrics = {
    "method": "Autoencoder",
    "n_components": int(k),
    "activation": ACTIVATION,
    "solver": SOLVER,
    "alpha": float(ALPHA),
    "max_iter": int(MAX_ITER),
    "reconstruction_mse": {
        "train": float(train_mse),
        "test": float(test_mse),
    },
    "labels": labels,
    "raw_features": {
        "accuracy": float(acc_raw),
        "classification_report": classification_report(y_test, y_pred_raw, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_raw, labels=labels).tolist(),
    },
    "original_autoencoder": {
        "accuracy": float(acc_ae),
        "classification_report": classification_report(y_test, y_pred_ae, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_ae, labels=labels).tolist(),
    },
    "quantized_codes": {
        "accuracy": float(acc_code),
        "classification_report": classification_report(y_test, y_pred_code, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_code, labels=labels).tolist(),
    },
    "tree_approx_codes": {
        "accuracy": float(acc_tree),
        "classification_report": classification_report(y_test, y_pred_tree, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_tree, labels=labels).tolist(),
    },
}
with open(os.path.join(TABLES_DIR, "transform_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

AE_all = np.vstack([AE_train, AE_test])
AE_code_tree_all = np.vstack([AE_code_tree_train, AE_code_tree_test])
y_all = np.concatenate([y_train, y_test])

mapping = {}
for j in range(k):
    mapping[f"AE{j+1}_float"] = AE_all[:, j]
for j in range(k):
    mapping[f"AE{j+1}_code"] = AE_code_tree_all[:, j]
mapping["Label"] = y_all

pd.DataFrame(mapping).to_csv(
    os.path.join(TABLES_DIR, "transform_mapping.csv"), index=False)
print("Saved transform_mapping.csv (AE columns)")

encoding_params = {
    "method": "autoencoder",
    "n_components": int(k),
    "bits": int(BITS),
    "max_val": int(MAX_VAL),
    "transform_min": ae_min.tolist(),
    "transform_max": ae_max.tolist(),
    "transform_range": ae_range.tolist(),
    "auto_selected": bool(USER_K is None),
    "feature_names": feature_cols,
    "encoder_activation": ACTIVATION,
    "solver": SOLVER,
    "alpha": float(ALPHA),
    "max_iter": int(MAX_ITER),
    "training_loss": float(getattr(autoencoder, "loss_", 0.0)),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "encoder_weights": encoder_weights.tolist(),
    "encoder_bias": encoder_bias.tolist(),
    "decoder_weights": decoder_weights.tolist(),
    "decoder_bias": decoder_bias.tolist(),
    "planter_inspired": True,
}
with open(os.path.join(TABLES_DIR, "encoding_params.json"), "w") as f:
    json.dump(encoding_params, f, indent=2)

feature_columns = [f"AE{j+1}_code" for j in range(k)]
feature_max_values = {col: int(MAX_VAL) for col in feature_columns}
reduction_config = {
    "method": "autoencoder",
    "feature_columns": feature_columns,
    "feature_max_values": feature_max_values,
    "needs_transform_tables": True,
    "n_components": int(k),
    "bits": int(BITS),
    "activation": ACTIVATION,
    "planter_inspired": True,
}
with open(os.path.join(TABLES_DIR, "reduction_config.json"), "w") as f:
    json.dump(reduction_config, f, indent=2)
print("Saved reduction_config.json")


feature_max_map = {feat: P4_FEATURE_MAX.get(feat, MAX_VAL) for feat in feature_cols}


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
    for feature_name in feature_names:
        feature_max = fmax_map.get(feature_name, MAX_VAL)
        if feature_name not in domain:
            clauses.append(f"0->{feature_max}")
            continue
        val = domain[feature_name]
        lo = (int(val["min"]) + 1) if val["min"] is not None else 0
        hi = int(val["max"]) if val["max"] is not None else feature_max
        lo = max(0, lo)
        hi = min(hi, feature_max)
        clauses.append(f"{lo}->{hi}")
    return clauses


def write_p4_commands(dt, feature_names, leaf_mapping, n_codes, fmax_map, filename):
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    features = tree_.feature
    priority = [0]

    def dfs(node_id, path):
        if left[node_id] == right[node_id]:
            priority[0] += 1
            new_path = [(feature_names[features[n]], sign, threshold[n])
                        for n, sign in path if features[n] != -2]
            domain = minimize(new_path)
            clauses = format_ranges(domain, feature_names, fmax_map)
            codes = leaf_mapping[node_id]
            for j in range(n_codes):
                f.write(
                    f"table_add MyIngress.ae_component{j+1} "
                    f"set_ae{j+1}_code {' '.join(clauses)} => {int(codes[j])} {priority[0]}\n"
                )
            return
        dfs(left[node_id], path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    with open(filename, "w") as f:
        dfs(0, [])


write_p4_commands(tree, feature_cols, leaf_to_codes, k, feature_max_map,
                  os.path.join(TABLES_DIR, "s1-commands.txt"))
print("Saved s1-commands.txt")


def write_if_rules(dt, feature_names, leaf_mapping, filename, n_codes):
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right
    threshold = tree_.threshold
    features = tree_.feature

    def dfs(node_id, path):
        if left[node_id] == right[node_id]:
            clauses = [f"{feature_names[features[n]]} {sign} {threshold[n]:.2f}"
                       for n, sign in path if features[n] != -2]
            codes = leaf_mapping[node_id]
            assigns = " AND ".join(f"AE{j+1}_code = {int(codes[j])}" for j in range(n_codes))
            f.write(f"IF ({' AND '.join(clauses) or 'TRUE'}) THEN {assigns};\n")
            return
        dfs(left[node_id], path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    with open(filename, "w") as f:
        f.write(f"# Autoencoder Transformation Rules\n# {len(feature_names)} raw features -> {n_codes} AE codes\n\n")
        dfs(0, [])


write_if_rules(tree, feature_cols, leaf_to_codes,
               os.path.join(TABLES_DIR, "transform_rules.txt"), k)

print("\n" + "=" * 60)
print("Autoencoder complete. Run any step 3/4 classifier next:")
print("  python3 3_train_model.py --model-type dt")
print("  python3 3_train_model.py --model-type rf")
print("=" * 60)
