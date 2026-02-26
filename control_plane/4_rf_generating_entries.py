#!/usr/bin/env python3
"""
Generate P4 table_add entries from a trained RandomForestClassifier.

For each tree in the forest a separate table is produced:
    table_add MyIngress.rf_tree_{i}  set_rf_tree_{i}_vote  <pc_ranges>  =>  <vote>  <priority>

A pre-computed vote-aggregation table finishes classification:
    table_add MyIngress.rf_vote_classify  set_result  <packed_votes>  =>  <majority_class>

Vote packing
------------
Each tree's class vote (0 … n_classes-1) occupies vote_bits bits inside a
single metadata field  meta.rf_votes  of width  n_estimators * vote_bits:

    rf_votes = vote_0 | (vote_1 << vote_bits) | … | (vote_{N-1} << ((N-1)*vote_bits))

The aggregation table has 2^(N*vote_bits) entries — one per vote pattern.

Defaults
--------
  Input model  : model/rf.model
  Output file  : tables/s1-commands.txt  (appended, not overwritten)
  Human tree   : tables/rf_trees.txt
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from itertools import product


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate P4 RF classification entries from trained model")

parser.add_argument('-i',          default="model/rf.model",
                    help='Path to trained RF model')
parser.add_argument('-o',          default="tables/s1-commands.txt",
                    help='Path to output P4 commands file (appended)')
parser.add_argument('--tree-out',  default="tables/rf_trees.txt",
                    help='Path to human-readable tree rules')
parser.add_argument('--bits', '-b', type=int, default=16,
                    help='PCA quantisation bits (default: 16)')
parser.add_argument('--params',    default="tables/rf_params.json",
                    help='RF params JSON written by 3_rf_training_model.py')

args = parser.parse_args()

inputfile   = args.i
outputfile  = args.o
tree_output = args.tree_out
BITS        = args.bits
MAX_VAL     = 2 ** BITS - 1

for d in [os.path.dirname(outputfile), os.path.dirname(tree_output)]:
    if d:
        os.makedirs(d, exist_ok=True)

MODEL_RULE_PREFIXES = (
    "table_add MyIngress.rf_tree_",
    "table_add MyIngress.rf_vote_classify",
)

def load_base_lines(path, drop_prefixes):
    """Load existing commands while removing old RF model rules."""
    if not os.path.exists(path):
        return []
    base_lines = []
    with open(path, "r") as f:
        for line in f:
            line_strip = line.lstrip()
            if line_strip.startswith(drop_prefixes):
                continue
            base_lines.append(line)
    return base_lines

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
rf = pd.read_pickle(inputfile)

if hasattr(rf, 'feature_names_in_'):
    FEATURE_NAMES = rf.feature_names_in_.tolist()
else:
    n_feat = rf.n_features_in_ if hasattr(rf, 'n_features_in_') else \
             rf.estimators_[0].n_features_in_
    FEATURE_NAMES = [f"pc{i+1}_code" for i in range(n_feat)]

n_classes    = len(rf.classes_)
n_estimators = len(rf.estimators_)
vote_bits    = max(1, math.ceil(math.log2(n_classes))) if n_classes > 1 else 1

label_encoding = {label: idx for idx, label in enumerate(rf.classes_)}
print("\n=== Label Encoding ===")
for label, code in label_encoding.items():
    print(f"  {label} -> {code}")

print(f"\nn_estimators={n_estimators}, n_classes={n_classes}, vote_bits={vote_bits}")
print(f"Vote aggregation table size: {2**(n_estimators*vote_bits):,} entries\n")

# Try to load params file for consistency check
if os.path.exists(args.params):
    with open(args.params) as f:
        params = json.load(f)
    assert params.get('vote_bits') == vote_bits, "vote_bits mismatch with params file"


# ---------------------------------------------------------------------------
# Shared tree-walking utilities  (mirroring 4_dt_generating_entries.py)
# ---------------------------------------------------------------------------

def minimize(path):
    """Collapse root-to-leaf constraints to {feature: {min, max}}."""
    domain = {}
    for (feature, sign, threshold) in path:
        domain.setdefault(feature, {"min": None, "max": None})
        if sign == "<=":
            domain[feature]["max"] = threshold
        else:
            domain[feature]["min"] = threshold
    return domain


def build_clause(domain, feature_names, max_val):
    """Convert a constraint domain to a list of 'lo->hi' range strings."""
    clause = []
    total_width = 0
    for fe in feature_names:
        val = domain.get(fe, {"min": None, "max": None})
        lo  = val["min"] if val["min"] is not None else -1
        hi  = val["max"] if val["max"] is not None else max_val
        lo  = int(lo) + 1
        hi  = int(hi)
        total_width += hi - lo + 1
        clause.append(f"{lo}->{hi}")
    return clause, total_width


def collect_tree_rules(dt, feature_names, max_val, table_name, action_name, class_to_index):
    """
    Walk a sklearn DecisionTree and return a list of
    (specificity, "table_add …") strings.
    """
    tree_  = dt.tree_
    left   = tree_.children_left
    right  = tree_.children_right
    # map tree node index → feature name
    feat_at = [feature_names[i] if i >= 0 else None for i in tree_.feature]

    rules = []

    def dfs(node_id, path):
        if left[node_id] == right[node_id]:          # leaf
            new_path = []
            for (n_id, sign) in path:
                new_path.append((feat_at[n_id], sign, tree_.threshold[n_id]))
            domain        = minimize(new_path)
            clause, width = build_clause(domain, feature_names, max_val)
            a             = list(tree_.value[node_id][0])
            local_idx     = a.index(max(a))
            local_label   = dt.classes_[local_idx]
            if local_label in class_to_index:
                vote = class_to_index[local_label]
            else:
                # Some sklearn trees store class indices (e.g., 0.0, 1.0) instead of labels
                vote = int(local_label)

            rule_str    = (f"table_add MyIngress.{table_name} {action_name} "
                           f"{' '.join(clause)} => {vote}")
            specificity = (len(clause), -width)
            rules.append((specificity, rule_str))
            return

        dfs(left[node_id],  path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    dfs(0, [])
    return rules


def write_tree_if_rules(dt, feature_names, tree_idx, file_handle):
    """Append human-readable IF/THEN rules for one tree."""
    tree_  = dt.tree_
    left   = tree_.children_left
    right  = tree_.children_right
    feat_at = [feature_names[i] if i >= 0 else None for i in tree_.feature]

    file_handle.write(f"\n# --- Tree {tree_idx} ---\n")

    def dfs(node_id, path):
        if left[node_id] == right[node_id]:
            clauses = [f"{feat_at[n_id]} {sign} {tree_.threshold[n_id]:.4f}"
                       for (n_id, sign) in path]
            a          = list(tree_.value[node_id][0])
            class_idx  = a.index(max(a))
            label      = dt.classes_[class_idx]
            cond_str   = " AND ".join(clauses) if clauses else "TRUE"
            file_handle.write(f"\tIF {cond_str} THEN {label};\n")
            return
        dfs(left[node_id],  path + [(node_id, "<=")])
        dfs(right[node_id], path + [(node_id, ">")])

    dfs(0, [])


# ---------------------------------------------------------------------------
# Generate per-tree table entries
# ---------------------------------------------------------------------------
print(f"Generating entries for {n_estimators} RF trees …")

all_tree_rules = []   # list of lists (one per estimator)

with open(tree_output, "w") as tf:
    tf.write("# Random Forest — human-readable tree rules\n")
    tf.write(f"# n_estimators={n_estimators}, n_classes={n_classes}\n")

    class_to_index = {label: idx for idx, label in enumerate(rf.classes_)}

    for i, est in enumerate(rf.estimators_):
        table_name  = f"rf_tree_{i}"
        action_name = f"set_rf_tree_{i}_vote"
        rules       = collect_tree_rules(est, FEATURE_NAMES, MAX_VAL,
                                         table_name, action_name,
                                         class_to_index)
        # Sort by specificity (most specific = lowest priority number)
        rules.sort(key=lambda x: x[0], reverse=True)
        all_tree_rules.append(rules)
        print(f"  tree {i}: {len(rules)} entries")
        write_tree_if_rules(est, FEATURE_NAMES, i, tf)

# ---------------------------------------------------------------------------
# Generate vote-aggregation table entries
# ---------------------------------------------------------------------------
print(f"\nGenerating vote-aggregation table ({2**(n_estimators*vote_bits):,} entries) …")

# Enumerate every possible packed-vote combination
aggregation_rules = []
for votes_tuple in product(range(n_classes), repeat=n_estimators):
    # Majority vote
    counter = Counter(votes_tuple)
    majority_class = counter.most_common(1)[0][0]

    # Pack votes into a single integer
    packed = 0
    for i, v in enumerate(votes_tuple):
        packed |= (v << (i * vote_bits))

    rule_str = (f"table_add MyIngress.rf_vote_classify set_result "
                f"{packed} => {majority_class}")
    aggregation_rules.append(rule_str)

# ---------------------------------------------------------------------------
# Write everything to the output file
# ---------------------------------------------------------------------------
print(f"Writing entries to {outputfile} …")

base_lines = load_base_lines(outputfile, MODEL_RULE_PREFIXES)
with open(outputfile, "w") as f:
    for line in base_lines:
        f.write(line)
    # Per-tree tables
    for i, rules in enumerate(all_tree_rules):
        for priority, (_, rule_str) in enumerate(rules, 1):
            f.write(f"{rule_str} {priority}\n")

    # Vote aggregation table (exact match; no priority)
    for rule_str in aggregation_rules:
        f.write(f"{rule_str}\n")
print(f"Human-readable rules   : {tree_output}")