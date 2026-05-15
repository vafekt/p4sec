#!/usr/bin/env python3
"""
P4 table_add entry generator for trained DT/RF classifiers.

Supports two classifier back-ends via --model-type:
  dt    DecisionTree    → single ml_code range-match table
  rf    RandomForest    → N rf_tree_i tables + rf_vote_classify

Works with any step-2 reduction method (PCA / LDA / Autoencoder).
Per-feature max values are auto-detected from tables/reduction_config.json.

Input:   model/<model_type>.model
Output:  tables/s1-commands.txt  (appended with classifier entries)
         tables/model_trees.txt  (human-readable rules)
"""

import os
import sys
import json
import math
import argparse
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
    description="Generate P4 classifier entries for DT/RF",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument('--model-type', '-m', required=True,
                    choices=['dt', 'rf'],
                    help='Classifier: dt | rf')
parser.add_argument('-i', default=None,
                    help='Model path (default: model/<model_type>.model)')
parser.add_argument('-o', default="tables/s1-commands.txt",
                    help='Output P4 commands file')
parser.add_argument('--tree-out', default=None,
                    help='Human-readable tree rules (default: tables/model_trees.txt)')
parser.add_argument('--params', default=None,
                    help='Params JSON (default: tables/model_params.json)')

args = parser.parse_args()

model_type = args.model_type
inputfile  = args.i or f"model/{model_type}.model"
outputfile = args.o
tree_output = args.tree_out or "tables/model_trees.txt"

tables_dir = os.path.join(os.path.dirname(__file__), 'tables')
for d in [os.path.dirname(outputfile), os.path.dirname(tree_output)]:
    if d:
        os.makedirs(d, exist_ok=True)

FEAT_MAX = detect_feature_max_values(tables_dir)


# ─── Shared utilities ────────────────────────────────────────────────────

# Model-specific table prefixes across DT/RF.
# load_base_lines() always strips ALL of these so that switching from one
# model type to another never leaves stale entries in s1-commands.txt.
ALL_MODEL_PREFIXES = (
    "table_add MyIngress.ml_code",           # DT
    "table_add MyIngress.rf_tree_",          # RF
    "table_add MyIngress.rf_vote_classify",  # RF
)


def load_base_lines(path, drop_prefixes=None):
    """Load existing s1-commands.txt, stripping all model-specific entries.

    Always strips ALL_MODEL_PREFIXES regardless of drop_prefixes, so switching
    model types never leaves stale table entries behind in the file.
    """
    if not os.path.exists(path):
        return []
    return [l for l in open(path) if not l.lstrip().startswith(ALL_MODEL_PREFIXES)]


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


# ─── sklearn tree walking (DT, RF) ───────────────────────────────────────

def walk_sklearn_tree_rules(dt, feature_names, feat_max, table_name, action_name,
                            class_mapper=None):
    """Walk a sklearn DecisionTree and produce P4 table_add rules.

    class_mapper: maps dt.classes_[idx] → integer vote/class_id.
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


def write_sklearn_tree_text(dt, feature_names, label, fh):
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
            a = list(tree_.value[node_id][0])
            fh.write(f"\tIF {cond} THEN {dt.classes_[a.index(max(a))]};\n")
            return
        dfs(left[node_id], path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    dfs(0, [])


def write_dt_params(tables_dir, model_type_label, classes):
    params = {
        "model_type": model_type_label,
        "classes": list(classes),
    }
    out_path = os.path.join(tables_dir, "model_params.json")
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
    _rcfg = load_reduction_config(tables_dir)
    if _rcfg and 'feature_columns' in _rcfg:
        FNAMES = _rcfg['feature_columns']
    else:
        _mp = args.params or os.path.join(tables_dir, 'model_params.json')
        if os.path.exists(_mp):
            with open(_mp) as f:
                _fn = json.load(f).get('feature_names')
            FNAMES = _fn if _fn else None
        else:
            FNAMES = None
    if FNAMES is None:
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

    # Write ml_code entries FIRST so they load even if pca loading is interrupted.
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
# RF  (P4 vote deployment)
# ═════════════════════════════════════════════════════════════════════════
elif model_type == 'rf':
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
