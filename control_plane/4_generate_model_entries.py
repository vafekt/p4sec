#!/usr/bin/env python3
"""
Universal P4 table_add entry generator for trained ML classifiers.

Supports seven classifier back-ends via --model-type:
  dt    DecisionTree       → single ml_code range-match table
  rf    RandomForest       → N rf_tree_i tables + rf_vote_classify
  xgb   XGBoost            → N*K xgb_tree_i tables + xgb_classify proxy DT
  gb    GradientBoosting   → same P4 layout as XGB (sklearn trees, no xgboost dep)
  knn   K-Nearest Neighbors → kd-tree range-match in P4 (ml_code)
  svm   Support Vector Machine → kd-tree range-match in P4 (ml_code)
  cnn   1D CNN             → P4 neural lookup tables (no DT/RF surrogate)

Works with any step-2 reduction method (PCA / LDA / Autoencoder / UMAP / Feature Selection).
Per-feature max values are auto-detected from tables/reduction_config.json.

Input:   model/<model_type>.model
Output:  tables/s1-commands.txt  (appended with classifier entries)
         tables/<model_type>_tree(s).txt  (human-readable rules)
"""

import os
import sys
import json
import math
import re
import argparse
import sys
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from pipeline_utils import detect_feature_max_values, load_reduction_config

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


# ─── CLI ─────────────────────────────────────────────────────────────────
parser = P4secArgumentParser(
    description="Generate P4 classifier entries (universal)",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Model-specific options:\n"
        "  xgb/gb : --csv (training CSV), --proxy-max-depth (proxy DT), --params (xgb/gb params)\n"
        "  rf     : --params (rf_params.json)\n"
        "  knn   : --csv (training CSV), --params (knn_params.json)\n"
        "  svm   : --csv (training CSV), --params (svm_params.json)\n"
        "  cnn    : --params (cnn_params.json)\n"
    )
)
parser.add_argument('--model-type', '-m', required=True,
                    choices=['dt', 'rf', 'xgb', 'gb', 'knn', 'svm', 'cnn'],
                    help='Classifier: dt | rf | xgb | gb | knn | svm | cnn')
parser.add_argument('-i', default=None,
                    help='Model path (default: model/<model_type>.model)')
parser.add_argument('-o', default="tables/s1-commands.txt",
                    help='Output P4 commands file')
parser.add_argument('--tree-out', default=None,
                    help='Human-readable tree rules (default: tables/<type>_tree(s).txt)')
parser.add_argument('--params', default=None,
                    help='Params JSON (for XGB/GB/RF/CNN; auto-detected if omitted)')
parser.add_argument('--csv', default="tables/transform_mapping.csv",
                    help='Training CSV for proxy DT (XGB/GB only)')
parser.add_argument('--proxy-max-depth', type=int, default=None,
                    help='Proxy DT max_depth for XGB/GB (default: None)')

args = parser.parse_args()

model_type = args.model_type
inputfile  = args.i or f"model/{model_type}.model"
outputfile = args.o
tree_output = args.tree_out or f"tables/{model_type}_tree{'s' if model_type != 'dt' else ''}.txt"

tables_dir = os.path.join(os.path.dirname(__file__), 'tables')
for d in [os.path.dirname(outputfile), os.path.dirname(tree_output)]:
    if d:
        os.makedirs(d, exist_ok=True)

FEAT_MAX = detect_feature_max_values(tables_dir)


# ─── Shared utilities ────────────────────────────────────────────────────

# All model-specific table prefixes across every backend.
# load_base_lines() always strips ALL of these so that switching from one
# model type to another never leaves stale entries in s1-commands.txt.
ALL_MODEL_PREFIXES = (
    "table_add MyIngress.ml_code",           # DT / KNN / SVM
    "table_add MyIngress.rf_tree_",          # RF
    "table_add MyIngress.rf_vote_classify",  # RF
    "table_add MyIngress.xgb_tree_",         # XGB / GB
    "table_add MyIngress.xgb_classify",      # XGB / GB
    "table_add MyIngress.cnn",               # CNN
)


def load_base_lines(path, drop_prefixes=None):
    """Load existing s1-commands.txt, stripping all model-specific entries.

    Always strips ALL_MODEL_PREFIXES regardless of drop_prefixes, so switching
    model types never leaves stale table entries behind in the file.
    """
    if not os.path.exists(path):
        return []
    return [l for l in open(path) if not l.lstrip().startswith(ALL_MODEL_PREFIXES)]


def generate_cnn_entries(cnn_params, output_path):
    feature_names = cnn_params.get("feature_names", [])
    classes = cnn_params.get("classes", [])
    hidden1_units = int(cnn_params.get("hidden1_units", 0))
    hidden2_units = int(cnn_params.get("hidden2_units", 0))
    p4_layers = int(cnn_params.get("p4_layers", 2 if hidden2_units > 0 else 1))
    pool = int(cnn_params.get("pool", 2))
    input_bits = int(cnn_params.get("input_bits", 8))
    hidden_bits = int(cnn_params.get("hidden_bits", 8))
    w1_int = np.array(cnn_params.get("w1_int", []), dtype=np.int64)
    w3_int = np.array(cnn_params.get("w3_int", []), dtype=np.int64)
    use_quanti = bool(cnn_params.get("use_quanti", False))
    # Per-unit quantization steps (fallback to legacy scalar for backward compat)
    q1_steps = cnn_params.get("q1_steps", None)
    q1_maxes = cnn_params.get("q1_maxes", None)
    if q1_steps is None:
        legacy_step = int(cnn_params.get("q1_step", 1))
        legacy_max = int(cnn_params.get("q1_max_pos", 0))
        q1_steps = [legacy_step] * hidden1_units
        q1_maxes = [legacy_max] * hidden1_units

    if not feature_names or hidden1_units <= 0 or w1_int.size == 0 or w3_int.size == 0:
        print("ERROR: cnn_params.json missing required fields.")
        sys.exit(1)

    max_q = (1 << input_bits) - 1
    def to_u32(val):
        return int(val) % (1 << 32)
    base_lines = load_base_lines(output_path, ("table_add MyIngress.cnn",))
    with open(output_path, 'w') as f:
        f.writelines(base_lines)

        # Hidden layer 1 weight tables
        for h in range(hidden1_units):
            for fi, _ in enumerate(feature_names):
                table_name = f"cnn1_h{h}_f{fi}"
                action_name = f"cnn1_add_h{h}"
                for q in range(max_q + 1):
                    delta = int(w1_int[h][fi] * q)
                    f.write(f"table_add MyIngress.{table_name} {action_name} {q} => {to_u32(delta)}\n")

        if use_quanti:
            levels = 1 << hidden_bits
            for h in range(hidden1_units):
                table_name = f"cnn1_quant_h{h}"
                action_name = f"set_cnn1_h{h}"
                h_step = int(q1_steps[h])
                h_max = int(q1_maxes[h])
                for q in range(levels):
                    lo = q * h_step
                    if lo > h_max:
                        break
                    hi = min(h_max, (q + 1) * h_step - 1)
                    f.write(f"table_add MyIngress.{table_name} {action_name} {lo}->{hi} => {q}\n")

        if p4_layers == 1:
            # P4CNN1: output layer reads directly from quantized hidden1
            for c in range(len(classes)):
                for h in range(hidden1_units):
                    table_name = f"cnn_out_c{c}_h{h}"
                    action_name = f"cnn_out_add_c{c}"
                    for a in range(0, (1 << hidden_bits)):
                        delta = int(w3_int[c][h] * a)
                        f.write(f"table_add MyIngress.{table_name} {action_name} {a} => {to_u32(delta)}\n")
        else:
            # P4CNN2: hidden layer 2 tables (after pooling)
            w2_int = np.array(cnn_params.get("w2_int", []), dtype=np.int64)
            q2_steps = cnn_params.get("q2_steps", None)
            q2_maxes = cnn_params.get("q2_maxes", None)
            if q2_steps is None:
                legacy_step = int(cnn_params.get("q2_step", 1))
                legacy_max = int(cnn_params.get("q2_max_pos", 0))
                q2_steps = [legacy_step] * hidden2_units
                q2_maxes = [legacy_max] * hidden2_units

            pooled = hidden1_units // max(1, pool)
            for h in range(hidden2_units):
                for pi in range(pooled):
                    table_name = f"cnn2_h{h}_p{pi}"
                    action_name = f"cnn2_add_h{h}"
                    for a in range(0, (1 << hidden_bits)):
                        delta = int(w2_int[h][pi] * a)
                        f.write(f"table_add MyIngress.{table_name} {action_name} {a} => {to_u32(delta)}\n")

            if use_quanti:
                levels = 1 << hidden_bits
                for h in range(hidden2_units):
                    table_name = f"cnn2_quant_h{h}"
                    action_name = f"set_cnn2_h{h}"
                    h_step = int(q2_steps[h])
                    h_max_val = int(q2_maxes[h])
                    for q in range(levels):
                        lo = q * h_step
                        if lo > h_max_val:
                            break
                        hi = min(h_max_val, (q + 1) * h_step - 1)
                        f.write(f"table_add MyIngress.{table_name} {action_name} {lo}->{hi} => {q}\n")

            # Output layer tables
            for c in range(len(classes)):
                for h in range(hidden2_units):
                    table_name = f"cnn_out_c{c}_h{h}"
                    action_name = f"cnn_out_add_c{c}"
                    for a in range(0, (1 << hidden_bits)):
                        delta = int(w3_int[c][h] * a)
                        f.write(f"table_add MyIngress.{table_name} {action_name} {a} => {to_u32(delta)}\n")

    print(f"Wrote CNN table entries (P4CNN{p4_layers}) to {output_path}")


if model_type == 'cnn':
    params_path = args.params or os.path.join(tables_dir, "cnn_params.json")
    if not os.path.exists(params_path):
        print(f"ERROR: CNN params not found: {params_path}")
        sys.exit(1)
    with open(params_path, 'r') as f:
        cnn_params = json.load(f)
    generate_cnn_entries(cnn_params, outputfile)
    sys.exit(0)


def minimize(path):
    """Collapse root-to-leaf constraints to {feature: {min, max}}."""
    domain = {}
    for (feat, sign, threshold) in path:
        domain.setdefault(feat, {"min": None, "max": None})
        if sign == "<=":
            domain[feat]["max"] = threshold
        else:
            domain[feat]["min"] = threshold
    return domain


def build_clause(domain, feature_names, feat_max):
    """Convert constraints to P4 range-match strings (lo->hi)."""
    clause, total_width = [], 0
    for fn in feature_names:
        val = domain.get(fn, {"min": None, "max": None})
        fm = feat_max.get(fn, 65535)
        lo = (int(val["min"]) + 1) if val["min"] is not None else 0
        hi = int(val["max"]) if val["max"] is not None else fm
        lo, hi = max(0, lo), min(hi, fm)
        total_width += hi - lo + 1
        clause.append(f"{lo}->{hi}")
    return clause, total_width


# ─── sklearn tree walking (DT, RF, GB) ──────────────────────────────────

def walk_sklearn_tree_rules(dt, feature_names, feat_max, table_name, action_name,
                            class_mapper=None, quant_fn=None):
    """
    Walk a sklearn DecisionTree and produce P4 table_add rules.

    class_mapper: for classifiers — maps dt.classes_[idx] → integer vote/class_id.
    quant_fn:     for regressors  — maps float leaf value → integer delta.
    """
    tree_ = dt.tree_
    left  = tree_.children_left
    right = tree_.children_right
    feat_at = [feature_names[i] if i >= 0 else None for i in tree_.feature]
    rules = []

    def dfs(node_id, path):
        if left[node_id] == right[node_id]:  # leaf
            new_path = [(feat_at[n], s, tree_.threshold[n]) for n, s in path]
            domain = minimize(new_path)
            clause, width = build_clause(domain, feature_names, feat_max)

            if quant_fn is not None:
                # Regressor leaf (GB/XGB per-tree)
                leaf_val = float(tree_.value[node_id, 0, 0])
                action_val = quant_fn(leaf_val)
            else:
                # Classifier leaf (DT, RF)
                a = list(tree_.value[node_id][0])
                local_idx = a.index(max(a))
                local_label = dt.classes_[local_idx]
                action_val = class_mapper(local_label) if class_mapper else local_idx

            rule_str = (f"table_add MyIngress.{table_name} {action_name} "
                        f"{' '.join(clause)} => {action_val}")
            rules.append(((len(clause), -width), rule_str))
            return
        dfs(left[node_id], path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    dfs(0, [])
    return rules


def write_sklearn_tree_text(dt, feature_names, label, fh, is_regressor=False):
    """Write human-readable IF/THEN rules for one sklearn tree."""
    tree_ = dt.tree_
    left  = tree_.children_left
    right = tree_.children_right
    feat_at = [feature_names[i] if i >= 0 else None for i in tree_.feature]
    fh.write(f"\n# --- {label} ---\n")

    def dfs(node_id, path):
        if left[node_id] == right[node_id]:
            clauses = [f"{feat_at[n]} {s} {tree_.threshold[n]:.4f}" for n, s in path]
            cond = ' AND '.join(clauses) or 'TRUE'
            if is_regressor:
                fh.write(f"\tIF {cond} THEN leaf={float(tree_.value[node_id, 0, 0]):.6f};\n")
            else:
                a = list(tree_.value[node_id][0])
                fh.write(f"\tIF {cond} THEN {dt.classes_[a.index(max(a))]};\n")
            return
        dfs(left[node_id], path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    dfs(0, [])


# ─── Quantiser for boosting leaf values ──────────────────────────────────

def make_quantiser(leaf_vals, delta_bits=8):
    lo, hi = min(leaf_vals), max(leaf_vals)
    rng, top = hi - lo, 2**delta_bits - 1
    return lambda v: 0 if rng == 0 else int(round((v - lo) / rng * top))


def collect_leaf_values_sklearn(tree_obj):
    t = tree_obj.tree_
    return [float(t.value[nid, 0, 0])
            for nid in range(t.node_count)
            if t.children_left[nid] == t.children_right[nid]]


# ─── XGBoost tree parsing ────────────────────────────────────────────────

def parse_xgb_trees(booster, feature_names):
    """Parse XGBoost booster into list of per-tree node dicts."""
    df = booster.trees_to_dataframe()
    f2name = {f"f{i}": fn for i, fn in enumerate(feature_names)}
    all_trees = []
    for t_idx in range(int(df['Tree'].max()) + 1):
        tdf = df[df['Tree'] == t_idx]
        nodes = {}
        for _, row in tdf.iterrows():
            nid = int(row['Node'])
            feat, gain = row['Feature'], float(row['Gain'])
            if feat == 'Leaf':
                nodes[nid] = {'feature': 'Leaf', 'gain': gain}
            else:
                nodes[nid] = {
                    'feature': f2name.get(feat, feat),
                    'split': float(row['Split']),
                    'yes': int(str(row['Yes']).split('-')[1]),
                    'no':  int(str(row['No']).split('-')[1]),
                }
        all_trees.append(nodes)
    return all_trees


def walk_xgb_tree_rules(nodes, feature_names, feat_max, table_name, action_name, quant_fn):
    """Walk a parsed XGB tree (dict of nodes) and produce P4 rules."""
    rules = []

    def dfs(nid, path):
        node = nodes[nid]
        if node['feature'] == 'Leaf':
            domain = minimize(path)
            clause, width = build_clause(domain, feature_names, feat_max)
            delta = quant_fn(node['gain'])
            rules.append(((len(clause), -width),
                f"table_add MyIngress.{table_name} {action_name} {' '.join(clause)} => {delta}"))
            return
        dfs(node['yes'], path + [(node['feature'], "<=", node['split'])])
        dfs(node['no'],  path + [(node['feature'], ">",  node['split'])])

    dfs(0, [])
    return rules


def collect_leaf_values_xgb(nodes):
    return [n['gain'] for n in nodes.values() if n['feature'] == 'Leaf']


def write_xgb_tree_text(nodes, feature_names, label, fh):
    fh.write(f"\n# --- {label} ---\n")
    def dfs(nid, path):
        node = nodes[nid]
        if node['feature'] == 'Leaf':
            clauses = [f"{f} {s} {t:.4f}" for f, s, t in path]
            fh.write(f"\tIF {' AND '.join(clauses) or 'TRUE'} THEN leaf={node['gain']:.6f};\n")
            return
        dfs(node['yes'], path + [(node['feature'], "<=", node['split'])])
        dfs(node['no'],  path + [(node['feature'], ">",  node['split'])])
    dfs(0, [])


# ─── Proxy DT builder (shared by XGB and GB) ────────────────────────────

def build_proxy_dt_rules(proxy_dt, n_classes, max_acc_val):
    """Walk proxy DT to produce xgb_classify entries."""
    score_cols = [f"score_c{c}" for c in range(n_classes)]
    rules = []

    def walk(dt, nid, fnames, smax, path=None):
        if path is None:
            path = []
        t = dt.tree_
        left, right = t.children_left, t.children_right
        if left[nid] == right[nid]:
            new_path = [(fnames[t.feature[n]], s, t.threshold[n]) for n, s in path]
            domain = minimize(new_path)
            clause, tw = [], 0
            for fn in fnames:
                val = domain.get(fn, {"min": None, "max": None})
                lo = (int(val["min"]) + 1) if val["min"] is not None else 0
                hi = int(val["max"]) if val["max"] is not None else smax
                lo, hi = max(0, lo), min(hi, smax)
                tw += hi - lo + 1
                clause.append(f"{lo}->{hi}")
            a = list(t.value[nid][0])
            rules.append(((len(clause), -tw),
                f"table_add MyIngress.xgb_classify set_result {' '.join(clause)} => {a.index(max(a))}"))
            return
        walk(dt, left[nid], fnames, smax, path + [(nid, "<=")])
        walk(dt, right[nid], fnames, smax, path + [(nid, ">")])

    walk(proxy_dt, 0, score_cols, max_acc_val)
    rules.sort(key=lambda x: x[0], reverse=True)
    return rules


def write_dt_params(tables_dir, model_type_label, classes, proxy_info=None):
    params = {
        "model_type": model_type_label,
        "classes": list(classes),
    }
    if proxy_info:
        params.update(proxy_info)
    out_path = os.path.join(tables_dir, "dt_params.json")
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote DT params: {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ═════════════════════════════════════════════════════════════════════════

model = pd.read_pickle(inputfile)
print(f"Loaded model: {inputfile}")

# Get feature names from model, fallback to reduction_config, then params
if hasattr(model, 'feature_names_in_'):
    FNAMES = model.feature_names_in_.tolist()
else:
    # Try reduction_config.json (universal)
    _rcfg = load_reduction_config(tables_dir)
    if _rcfg and 'feature_columns' in _rcfg:
        FNAMES = _rcfg['feature_columns']
    elif model_type in ('xgb', 'gb'):
        params_path = args.params or os.path.join(tables_dir, f'{model_type}_params.json')
        if not os.path.exists(params_path):
            params_path = os.path.join(tables_dir, 'xgb_params.json')
        with open(params_path) as f:
            FNAMES = json.load(f)['feature_names']
    elif model_type in ('rf',):
        params_path = args.params or os.path.join(tables_dir, f'{model_type}_params.json')
        if not os.path.exists(params_path):
            params_path = os.path.join(tables_dir, 'rf_params.json')
        with open(params_path) as f:
            FNAMES = json.load(f)['feature_names']
    else:
        n = model.n_features_in_ if hasattr(model, 'n_features_in_') else model.tree_.n_features
        FNAMES = [f"feature_{i}" for i in range(n)]
        print(f"WARNING: Could not detect feature names. Using {FNAMES}")

fmax = {fn: FEAT_MAX.get(fn, 65535) for fn in FNAMES}
print(f"Features: {FNAMES}")
print(f"Feature max values: {fmax}")


# ═════════════════════════════════════════════════════════════════════════
# DT
# ═════════════════════════════════════════════════════════════════════════
if model_type == 'dt':
    RULE_PREFIXES = ("table_add MyIngress.ml_code",)

    label_enc = {label: idx for idx, label in enumerate(model.classes_)}
    print(f"Labels: {label_enc}")

    rules = walk_sklearn_tree_rules(
        model, FNAMES, fmax, "ml_code", "set_result",
        class_mapper=lambda lbl: label_enc.get(lbl, label_enc.get(str(lbl), 0)))
    rules.sort(key=lambda x: x[0], reverse=True)

    base = load_base_lines(outputfile, RULE_PREFIXES)

    # Write ml_code entries FIRST so they load even if pca loading is interrupted
    # BMv2: higher priority number = matched first → most-specific rules get highest numbers
    with open(outputfile, "w") as f:
        for prio, (_, rs) in zip(range(len(rules), 0, -1), rules):
            f.write(f"{rs} {prio}\n")
        for line in base:
            f.write(line)
    print(f"Wrote {len(rules)} DT entries to {outputfile} (classifier entries placed first)")


    with open(tree_output, "w") as fh:
        write_sklearn_tree_text(model, FNAMES, "DecisionTree", fh)
    print(f"Tree structure: {tree_output}")
    write_dt_params(tables_dir, "dt", model.classes_)


# ═════════════════════════════════════════════════════════════════════════
# KNN / SVM  (kd-tree space partitioning → P4 range-match rules)
# ═════════════════════════════════════════════════════════════════════════
elif model_type in ('knn', 'svm'):

    RULE_PREFIXES = ("table_add MyIngress.ml_code",)

    # Load params
    params_path = args.params or os.path.join(tables_dir, f'{model_type}_params.json')
    assert os.path.exists(params_path), f"{model_type.upper()} params not found: {params_path}"
    with open(params_path) as f:
        kdtree_params = json.load(f)

    depth_key = 'knn_depth' if model_type == 'knn' else 'svm_depth'
    max_depth = kdtree_params.get(depth_key, 16)

    label_enc = {label: idx for idx, label in enumerate(model.classes_)}
    print(f"Labels: {label_enc}")
    print(f"KD-tree max depth: {max_depth}")

    num_features = len(FNAMES)
    rules = []
    MAX_RULES = 50000  # safety cap

    # Use training data to guide splitting
    assert os.path.exists(args.csv), f"CSV not found: {args.csv}"
    df_train = pd.read_csv(args.csv)
    df_train.columns = df_train.columns.str.lower()
    fnames_lower = [f.lower() for f in FNAMES]
    X_all = df_train[fnames_lower].values.astype(np.float64)
    y_model_all = model.predict(X_all)

    # Per-feature bounds for P4 range matching
    feat_max_arr = np.array([float(fmax.get(fn, 65535)) for fn in FNAMES])

    def kdtree_partition(lo, hi, depth, data_indices):
        """Binary space partition: split one feature at a time.
        Uses training data to determine the split point (median) and to check
        if the cell is pure (all same class).
        """
        if len(rules) >= MAX_RULES:
            return

        if len(data_indices) == 0:
            center = ((lo + hi) / 2.0).reshape(1, -1)
            cls = model.predict(center)[0]
            cls_id = label_enc.get(cls, label_enc.get(str(cls), 0))
            clause, tw = _make_clause(lo, hi)
            rules.append(((num_features, -tw),
                f"table_add MyIngress.ml_code set_result {' '.join(clause)} => {cls_id}"))
            return

        preds = y_model_all[data_indices]
        unique_preds = np.unique(preds)

        if len(unique_preds) == 1 or depth >= max_depth:
            cls = Counter(preds).most_common(1)[0][0]
            cls_id = label_enc.get(cls, label_enc.get(str(cls), 0))
            clause, tw = _make_clause(lo, hi)
            rules.append(((num_features, -tw),
                f"table_add MyIngress.ml_code set_result {' '.join(clause)} => {cls_id}"))
            return

        X_cell = X_all[data_indices]
        best_dim = None
        best_impurity_reduction = -1

        for attempt in range(num_features):
            dim = (depth + attempt) % num_features
            if hi[dim] - lo[dim] < 1:
                continue
            median_val = np.floor(np.median(X_cell[:, dim]))
            median_val = max(lo[dim], min(median_val, hi[dim] - 1))

            left_mask = X_cell[:, dim] <= median_val
            right_mask = ~left_mask
            n_left, n_right = left_mask.sum(), right_mask.sum()

            if n_left == 0 or n_right == 0:
                continue

            def gini(indices):
                if len(indices) == 0:
                    return 0
                counts = np.bincount([label_enc.get(p, 0) for p in y_model_all[indices]])
                probs = counts / len(indices)
                return 1.0 - np.sum(probs ** 2)

            left_idx = data_indices[left_mask]
            right_idx = data_indices[right_mask]
            parent_gini = gini(data_indices)
            child_gini = (n_left * gini(left_idx) + n_right * gini(right_idx)) / len(data_indices)
            reduction = parent_gini - child_gini

            if reduction > best_impurity_reduction:
                best_impurity_reduction = reduction
                best_dim = dim
                best_median = median_val
                best_left_idx = left_idx
                best_right_idx = right_idx

        if best_dim is None:
            cls = Counter(preds).most_common(1)[0][0]
            cls_id = label_enc.get(cls, label_enc.get(str(cls), 0))
            clause, tw = _make_clause(lo, hi)
            rules.append(((num_features, -tw),
                f"table_add MyIngress.ml_code set_result {' '.join(clause)} => {cls_id}"))
            return

        lo_left, hi_left = lo.copy(), hi.copy()
        hi_left[best_dim] = best_median

        lo_right, hi_right = lo.copy(), hi.copy()
        lo_right[best_dim] = best_median + 1

        kdtree_partition(lo_left, hi_left, depth + 1, best_left_idx)
        kdtree_partition(lo_right, hi_right, depth + 1, best_right_idx)

    def _make_clause(lo, hi):
        clause = []
        total_width = 1
        for f_idx, fn in enumerate(FNAMES):
            r_lo = max(0, int(lo[f_idx]))
            r_hi = min(int(hi[f_idx]), int(feat_max_arr[f_idx]))
            clause.append(f"{r_lo}->{r_hi}")
            total_width *= max(1, r_hi - r_lo + 1)
        return clause, total_width

    print(f"Building kd-tree partitioning ({num_features} features, depth {max_depth})...")
    bounds_lo = np.zeros(num_features)
    bounds_hi = feat_max_arr.copy()
    all_indices = np.arange(len(X_all))
    kdtree_partition(bounds_lo, bounds_hi, 0, all_indices)
    rules.sort(key=lambda x: x[0], reverse=True)
    print(f"Generated {len(rules)} range-match rules")

    # Verify accuracy on training data
    n_rules = len(rules)
    rule_lo = np.zeros((n_rules, num_features), dtype=np.int64)
    rule_hi = np.zeros((n_rules, num_features), dtype=np.int64)
    rule_cls = np.zeros(n_rules, dtype=np.int64)
    for ri, (_, rule_str) in enumerate(rules):
        parts = rule_str.split(' => ')
        rule_cls[ri] = int(parts[1])
        range_strs = parts[0].split('set_result ')[1].split()
        for f_idx, rs in enumerate(range_strs):
            lo_s, hi_s = rs.split('->')
            rule_lo[ri, f_idx] = int(lo_s)
            rule_hi[ri, f_idx] = int(hi_s)

    y_model_enc = np.array([label_enc.get(y, 0) for y in y_model_all])
    y_table = np.full(len(X_all), -1, dtype=np.int64)
    for si in range(len(X_all)):
        x = X_all[si]
        match = np.all((x >= rule_lo) & (x <= rule_hi), axis=1)
        matched = np.where(match)[0]
        if len(matched) > 0:
            y_table[si] = rule_cls[matched[0]]

    table_acc = np.mean(y_table == y_model_enc)
    mt_upper = model_type.upper()
    print(f"KD-tree table accuracy (vs {mt_upper}): {table_acc:.4f}")

    base = load_base_lines(outputfile, RULE_PREFIXES)
    with open(outputfile, "w") as f:
        for prio, (_, rs) in zip(range(len(rules), 0, -1), rules):
            f.write(f"{rs} {prio}\n")
        for line in base:
            f.write(line)
    print(f"Wrote {len(rules)} {mt_upper} entries to {outputfile} (classifier entries placed first)")

    with open(tree_output, "w") as fh:
        fh.write(f"# {mt_upper} kd-tree partitioning (depth={max_depth}, acc={table_acc:.4f})\n")
        fh.write(f"# {len(rules)} range-match rules, {num_features} features\n")
        fh.write(f"# Labels: {label_enc}\n\n")
        for _, rs in rules[:50]:
            fh.write(f"{rs}\n")
        if len(rules) > 50:
            fh.write(f"\n# ... ({len(rules) - 50} more rules)\n")
    print(f"Tree structure: {tree_output}")

    write_dt_params(tables_dir, model_type, model.classes_,
                    proxy_info={"kdtree_depth": max_depth,
                                "kdtree_rules": len(rules),
                                "kdtree_accuracy": float(table_acc)})


# ═════════════════════════════════════════════════════════════════════════
# RF  (P4 vote deployment)
# ═════════════════════════════════════════════════════════════════════════
elif model_type in ('rf',):
    RULE_PREFIXES = ("table_add MyIngress.rf_tree_", "table_add MyIngress.rf_vote_classify")

    n_classes = len(model.classes_)
    n_est     = len(model.estimators_)
    vote_bits = max(1, math.ceil(math.log2(n_classes))) if n_classes > 1 else 1
    label_enc = {label: idx for idx, label in enumerate(model.classes_)}
    print(f"Labels: {label_enc}")
    print(f"n_estimators={n_est}, vote_bits={vote_bits}, vote_table={2**(n_est*vote_bits):,}")

    all_tree_rules = []
    with open(tree_output, "w") as tf:
        tf.write(f"# {model_type.upper()} — {n_est} trees, {n_classes} classes\n")
        for i, est in enumerate(model.estimators_):
            rules = walk_sklearn_tree_rules(
                est, FNAMES, fmax,
                f"rf_tree_{i}", f"set_rf_tree_{i}_vote",
                class_mapper=lambda lbl: label_enc.get(lbl, int(lbl) if not isinstance(lbl, str) else 0))
            rules.sort(key=lambda x: x[0], reverse=True)
            all_tree_rules.append(rules)
            print(f"  tree {i}: {len(rules)} entries")
            write_sklearn_tree_text(est, FNAMES, f"Tree {i}", tf)

    # Vote aggregation table
    print(f"\nGenerating vote-aggregation ({2**(n_est*vote_bits):,} entries)...")
    agg_rules = []
    for votes in product(range(n_classes), repeat=n_est):
        majority = Counter(votes).most_common(1)[0][0]
        packed = sum(v << (i * vote_bits) for i, v in enumerate(votes))
        agg_rules.append(f"table_add MyIngress.rf_vote_classify set_result {packed} => {majority}")

    base = load_base_lines(outputfile, RULE_PREFIXES)
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        # BMv2: higher priority number = matched first → most-specific rules get highest numbers
        for rules in all_tree_rules:
            for prio, (_, rs) in zip(range(len(rules), 0, -1), rules):
                f.write(f"{rs} {prio}\n")
        for rs in agg_rules:
            f.write(f"{rs}\n")

    total = sum(len(r) for r in all_tree_rules)
    print(f"Total: {total} tree + {len(agg_rules)} vote entries -> {outputfile}")


# ═════════════════════════════════════════════════════════════════════════
# XGB
# ═════════════════════════════════════════════════════════════════════════
elif model_type == 'xgb':
    try:
        import xgboost as xgb_lib
    except ImportError:
        print("ERROR: xgboost required. Install with: pip install xgboost")
        sys.exit(1)

    RULE_PREFIXES = ("table_add MyIngress.xgb_tree_", "table_add MyIngress.xgb_classify")

    params_path = args.params or os.path.join(tables_dir, 'xgb_params.json')
    with open(params_path) as f:
        params = json.load(f)
    n_classes   = params['n_classes']
    total_trees = params['total_trees']
    classes     = params['classes']
    label_enc   = {l: i for i, l in enumerate(classes)}

    booster = model.get_booster()
    all_nodes = parse_xgb_trees(booster, FNAMES)

    all_tree_rules = []
    with open(tree_output, "w") as tf:
        tf.write(f"# XGBoost — {total_trees} trees, {n_classes} classes\n")
        for tidx in range(total_trees):
            cidx = tidx % n_classes if n_classes > 2 else 1
            nodes = all_nodes[tidx]
            lvals = collect_leaf_values_xgb(nodes)
            qfn = make_quantiser(lvals)
            rules = walk_xgb_tree_rules(nodes, FNAMES, fmax,
                f"xgb_tree_{tidx}", f"add_xgb_score_c{cidx}", qfn)
            rules.sort(key=lambda x: x[0], reverse=True)
            all_tree_rules.append(rules)
            print(f"  tree {tidx:3d} (class {cidx}): {len(rules)} entries")
            write_xgb_tree_text(nodes, FNAMES, f"XGB Tree {tidx} (class {cidx})", tf)

    # Proxy DT
    print(f"\nBuilding proxy DT...")
    assert os.path.exists(args.csv), f"CSV not found: {args.csv}"
    df_train = pd.read_csv(args.csv)
    df_train.columns = df_train.columns.str.lower()
    fnames_lower = [f.lower() for f in FNAMES]
    X_raw = df_train[fnames_lower].values.astype(np.float64)
    Y_raw = df_train['label'].values
    mask = np.isfinite(X_raw).all(axis=1)
    X_raw, Y_raw = X_raw[mask], Y_raw[mask]

    dmatrix = xgb_lib.DMatrix(X_raw, feature_names=FNAMES)
    leaf_indices = booster.predict(dmatrix, pred_leaf=True)
    n_samples = X_raw.shape[0]
    accum = np.zeros((n_samples, n_classes), dtype=np.int32)
    for tidx in range(total_trees):
        cidx = tidx % n_classes if n_classes > 2 else 1
        nodes = all_nodes[tidx]
        lvals = collect_leaf_values_xgb(nodes)
        qfn = make_quantiser(lvals)
        l2d = {nid: qfn(n['gain']) for nid, n in nodes.items() if n['feature'] == 'Leaf'}
        for si in range(n_samples):
            accum[si, cidx] += l2d.get(int(leaf_indices[si, tidx]), 0)

    from sklearn import tree as sklearn_tree
    y_enc = np.array([label_enc.get(y, 0) for y in Y_raw])
    proxy_dt = sklearn_tree.DecisionTreeClassifier(
        max_depth=args.proxy_max_depth, random_state=42)
    proxy_dt.fit(accum, y_enc)
    proxy_acc = np.mean(proxy_dt.predict(accum) == y_enc)
    print(f"Proxy DT accuracy: {proxy_acc:.4f}")

    proxy_rules = build_proxy_dt_rules(proxy_dt, n_classes, int(accum.max()) + 1)
    print(f"xgb_classify entries: {len(proxy_rules)}")

    with open(tree_output, "a") as tf:
        tf.write(f"\n\n# === Proxy DT (accuracy={proxy_acc:.4f}) ===\n")
        for line in sklearn_tree.export_text(
                proxy_dt, feature_names=[f"score_c{c}" for c in range(n_classes)]).splitlines():
            tf.write(f"# {line}\n")

    base = load_base_lines(outputfile, RULE_PREFIXES)
    # BMv2: higher priority number = matched first → most-specific rules get highest numbers
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        for rules in all_tree_rules:
            for prio, (_, rs) in zip(range(len(rules), 0, -1), rules):
                f.write(f"{rs} {prio}\n")
        for prio, (_, rs) in zip(range(len(proxy_rules), 0, -1), proxy_rules):
            f.write(f"{rs} {prio}\n")

    total = sum(len(r) for r in all_tree_rules)
    print(f"Total: {total} tree + {len(proxy_rules)} classify entries -> {outputfile}")


# ═════════════════════════════════════════════════════════════════════════
# GB (same P4 layout as XGB, but sklearn trees — no xgboost dependency)
# ═════════════════════════════════════════════════════════════════════════
elif model_type == 'gb':
    RULE_PREFIXES = ("table_add MyIngress.xgb_tree_", "table_add MyIngress.xgb_classify")

    params_path = args.params or os.path.join(tables_dir, 'gb_params.json')
    if not os.path.exists(params_path):
        params_path = os.path.join(tables_dir, 'xgb_params.json')
    with open(params_path) as f:
        params = json.load(f)
    n_classes   = params['n_classes']
    total_trees = params['total_trees']
    n_est       = params['n_estimators']
    classes     = params['classes']
    label_enc   = {l: i for i, l in enumerate(classes)}

    # Extract all trees from gb.estimators_ (shape: n_estimators x n_tree_outputs)
    all_trees = []  # (tree_obj, class_idx, global_idx)
    gidx = 0
    for r in range(n_est):
        for oi in range(model.estimators_.shape[1]):
            cidx = oi if n_classes > 2 else 1
            all_trees.append((model.estimators_[r, oi], cidx, gidx))
            gidx += 1
    assert len(all_trees) == total_trees

    all_tree_rules = []
    with open(tree_output, "w") as tf:
        tf.write(f"# GradientBoosting — {total_trees} trees, {n_classes} classes\n")
        for tree_obj, cidx, gidx in all_trees:
            lvals = collect_leaf_values_sklearn(tree_obj)
            qfn = make_quantiser(lvals)
            rules = walk_sklearn_tree_rules(
                tree_obj, FNAMES, fmax,
                f"xgb_tree_{gidx}", f"add_xgb_score_c{cidx}",
                quant_fn=qfn)
            rules.sort(key=lambda x: x[0], reverse=True)
            all_tree_rules.append(rules)
            print(f"  tree {gidx:3d} (class {cidx}): {len(rules)} entries")
            write_sklearn_tree_text(tree_obj, FNAMES, f"GB Tree {gidx} (class {cidx})", tf,
                                    is_regressor=True)

    # Proxy DT on accumulated scores
    print(f"\nBuilding proxy DT...")
    assert os.path.exists(args.csv), f"CSV not found: {args.csv}"
    df_train = pd.read_csv(args.csv)
    df_train.columns = df_train.columns.str.lower()
    fnames_lower = [f.lower() for f in FNAMES]
    X_raw = df_train[fnames_lower].values.astype(np.float64)
    Y_raw = df_train['label'].values
    mask = np.isfinite(X_raw).all(axis=1)
    X_raw, Y_raw = X_raw[mask], Y_raw[mask]

    n_samples = X_raw.shape[0]
    accum = np.zeros((n_samples, n_classes), dtype=np.int32)
    for tree_obj, cidx, gidx in all_trees:
        lvals = collect_leaf_values_sklearn(tree_obj)
        qfn = make_quantiser(lvals)
        t = tree_obj.tree_
        l2d = {nid: qfn(float(t.value[nid, 0, 0]))
               for nid in range(t.node_count)
               if t.children_left[nid] == t.children_right[nid]}
        leaf_ids = tree_obj.apply(X_raw)
        for si in range(n_samples):
            accum[si, cidx] += l2d.get(int(leaf_ids[si]), 0)

    from sklearn import tree as sklearn_tree
    y_enc = np.array([label_enc.get(y, 0) for y in Y_raw])
    proxy_dt = sklearn_tree.DecisionTreeClassifier(
        max_depth=args.proxy_max_depth, random_state=42)
    proxy_dt.fit(accum, y_enc)
    proxy_acc = np.mean(proxy_dt.predict(accum) == y_enc)
    print(f"Proxy DT accuracy: {proxy_acc:.4f}")

    proxy_rules = build_proxy_dt_rules(proxy_dt, n_classes, int(accum.max()) + 1)
    print(f"xgb_classify entries: {len(proxy_rules)}")

    with open(tree_output, "a") as tf:
        tf.write(f"\n\n# === Proxy DT (accuracy={proxy_acc:.4f}) ===\n")
        for line in sklearn_tree.export_text(
                proxy_dt, feature_names=[f"score_c{c}" for c in range(n_classes)]).splitlines():
            tf.write(f"# {line}\n")

    base = load_base_lines(outputfile, RULE_PREFIXES)
    # BMv2: higher priority number = matched first → most-specific rules get highest numbers
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        for rules in all_tree_rules:
            for prio, (_, rs) in zip(range(len(rules), 0, -1), rules):
                f.write(f"{rs} {prio}\n")
        for prio, (_, rs) in zip(range(len(proxy_rules), 0, -1), proxy_rules):
            f.write(f"{rs} {prio}\n")

    total = sum(len(r) for r in all_tree_rules)
    print(f"Total: {total} tree + {len(proxy_rules)} classify entries -> {outputfile}")


print("\nDone.")
