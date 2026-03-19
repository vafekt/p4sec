#!/usr/bin/env python3
"""
Universal P4 table_add entry generator for trained ML classifiers.

Supports seven classifier back-ends via --model-type:
  dt    DecisionTree       → single ml_code range-match table
  rf    RandomForest       → N rf_tree_i tables + rf_vote_classify
  xgb   XGBoost            → N*K xgb_tree_i tables + xgb_classify proxy DT
  gb    GradientBoosting   → same P4 layout as XGB (sklearn trees, no xgboost dep)
  knn   K-Nearest Neighbors → DT proxy in P4 (ml_code)
  svm   Support Vector Machine → DT proxy in P4 (ml_code)
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
        "  knn/svm: --csv (training CSV), --proxy-max-depth (proxy DT)\n"
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

def load_base_lines(path, drop_prefixes):
    """Load existing commands, stripping lines that match drop_prefixes."""
    if not os.path.exists(path):
        return []
    return [l for l in open(path) if not l.lstrip().startswith(drop_prefixes)]


def generate_cnn_entries(cnn_params, output_path):
    feature_names = cnn_params.get("feature_names", [])
    classes = cnn_params.get("classes", [])
    hidden1_units = int(cnn_params.get("hidden1_units", 0))
    hidden2_units = int(cnn_params.get("hidden2_units", 0))
    pool = int(cnn_params.get("pool", 2))
    input_bits = int(cnn_params.get("input_bits", 8))
    hidden_bits = int(cnn_params.get("hidden_bits", 8))
    w1_int = np.array(cnn_params.get("w1_int", []), dtype=np.int64)
    w2_int = np.array(cnn_params.get("w2_int", []), dtype=np.int64)
    w3_int = np.array(cnn_params.get("w3_int", []), dtype=np.int64)
    use_quanti = bool(cnn_params.get("use_quanti", False))
    q1_step = int(cnn_params.get("q1_step", 1))
    q1_max = int(cnn_params.get("q1_max_pos", 0))
    q2_step = int(cnn_params.get("q2_step", 1))
    q2_max = int(cnn_params.get("q2_max_pos", 0))

    if (not feature_names or hidden1_units <= 0 or hidden2_units <= 0 or
            w1_int.size == 0 or w2_int.size == 0 or w3_int.size == 0):
        print("ERROR: cnn_params.json missing required fields.")
        sys.exit(1)

    max_q = (1 << input_bits) - 1
    def to_u32(val):
        return int(val) % (1 << 32)
    base_lines = load_base_lines(output_path, ("table_add MyIngress.cnn",))
    with open(output_path, 'w') as f:
        f.writelines(base_lines)

        # Hidden layer tables
        for h in range(hidden1_units):
            for fi, _ in enumerate(feature_names):
                table_name = f"cnn1_h{h}_f{fi}"
                action_name = f"cnn1_add_h{h}"
                for q in range(max_q + 1):
                    delta = int(w1_int[h][fi] * q)
                    f.write(f"table_add MyIngress.{table_name} {action_name} {q} => {to_u32(delta)}\n")

        if use_quanti:
            levels = 1 << hidden_bits
            # Quantization tables for hidden1
            for h in range(hidden1_units):
                table_name = f"cnn1_quant_h{h}"
                action_name = f"set_cnn1_h{h}"
                for q in range(levels):
                    lo = q * q1_step
                    if lo > q1_max:
                        break
                    hi = min(q1_max, (q + 1) * q1_step - 1)
                    f.write(f"table_add MyIngress.{table_name} {action_name} {lo}->{hi} => {q}\n")

        # Hidden layer 2 tables (after pooling)
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
            # Quantization tables for hidden2
            for h in range(hidden2_units):
                table_name = f"cnn2_quant_h{h}"
                action_name = f"set_cnn2_h{h}"
                for q in range(levels):
                    lo = q * q2_step
                    if lo > q2_max:
                        break
                    hi = min(q2_max, (q + 1) * q2_step - 1)
                    f.write(f"table_add MyIngress.{table_name} {action_name} {lo}->{hi} => {q}\n")

        # Output layer tables
        for c in range(len(classes)):
            for h in range(hidden2_units):
                table_name = f"cnn_out_c{c}_h{h}"
                action_name = f"cnn_out_add_c{c}"
                for a in range(0, (1 << hidden_bits)):
                    delta = int(w3_int[c][h] * a)
                    f.write(f"table_add MyIngress.{table_name} {action_name} {a} => {to_u32(delta)}\n")

    print(f"Wrote CNN table entries to {output_path}")


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
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        for prio, (_, rs) in enumerate(rules, 1):
            f.write(f"{rs} {prio}\n")
    print(f"Wrote {len(rules)} DT entries to {outputfile}")

    with open(tree_output, "w") as fh:
        write_sklearn_tree_text(model, FNAMES, "DecisionTree", fh)
    print(f"Tree structure: {tree_output}")
    write_dt_params(tables_dir, "dt", model.classes_)


# ═════════════════════════════════════════════════════════════════════════
# KNN / SVM  (DT proxy in P4)
# ═════════════════════════════════════════════════════════════════════════
elif model_type in ('knn', 'svm'):
    from sklearn import tree as sklearn_tree

    RULE_PREFIXES = ("table_add MyIngress.ml_code",)

    assert os.path.exists(args.csv), f"CSV not found: {args.csv}"
    df_train = pd.read_csv(args.csv)
    df_train.columns = df_train.columns.str.lower()
    fnames_lower = [f.lower() for f in FNAMES]
    X_raw = df_train[fnames_lower].values.astype(np.float64)

    # Use the original model's predictions as surrogate targets
    y_pred = model.predict(X_raw)
    label_enc = {label: idx for idx, label in enumerate(model.classes_)}

    proxy_dt = sklearn_tree.DecisionTreeClassifier(
        max_depth=args.proxy_max_depth, random_state=42)
    proxy_dt.fit(X_raw, y_pred)
    proxy_acc = np.mean(proxy_dt.predict(X_raw) == y_pred)
    print(f"Proxy DT accuracy (vs {model_type.upper()}): {proxy_acc:.4f}")

    rules = walk_sklearn_tree_rules(
        proxy_dt, FNAMES, fmax, "ml_code", "set_result",
        class_mapper=lambda lbl: label_enc.get(lbl, label_enc.get(str(lbl), 0)))
    rules.sort(key=lambda x: x[0], reverse=True)

    base = load_base_lines(outputfile, RULE_PREFIXES)
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        for prio, (_, rs) in enumerate(rules, 1):
            f.write(f"{rs} {prio}\n")
    print(f"Wrote {len(rules)} {model_type.upper()} proxy DT entries to {outputfile}")

    with open(tree_output, "w") as fh:
        fh.write(f"# Proxy DT for {model_type.upper()} (acc={proxy_acc:.4f})\n")
        write_sklearn_tree_text(proxy_dt, FNAMES, f"ProxyDT({model_type.upper()})", fh)
    print(f"Tree structure: {tree_output}")

    write_dt_params(
        tables_dir,
        model_type,
        model.classes_,
        proxy_info={"proxy_accuracy": float(proxy_acc), "proxy_max_depth": args.proxy_max_depth},
    )


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
                class_mapper=lambda lbl: label_enc.get(lbl, label_enc.get(str(lbl), 0)))
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
        for rules in all_tree_rules:
            for prio, (_, rs) in enumerate(rules, 1):
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
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        for rules in all_tree_rules:
            for prio, (_, rs) in enumerate(rules, 1):
                f.write(f"{rs} {prio}\n")
        for prio, (_, rs) in enumerate(proxy_rules, 1):
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
    with open(outputfile, "w") as f:
        for line in base:
            f.write(line)
        for rules in all_tree_rules:
            for prio, (_, rs) in enumerate(rules, 1):
                f.write(f"{rs} {prio}\n")
        for prio, (_, rs) in enumerate(proxy_rules, 1):
            f.write(f"{rs} {prio}\n")

    total = sum(len(r) for r in all_tree_rules)
    print(f"Total: {total} tree + {len(proxy_rules)} classify entries -> {outputfile}")


print("\nDone.")
