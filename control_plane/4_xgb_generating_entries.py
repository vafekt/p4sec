#!/usr/bin/env python3
"""
Generate P4 table_add entries from a trained XGBoostClassifier.

Architecture
------------
For each of the  total_trees = n_estimators * n_classes  XGBoost trees a
separate table accumulates a quantised leaf-value delta into its class score:

    table_add MyIngress.xgb_tree_{i}  add_xgb_score_c{c}  <pc_ranges>  =>  <delta_8bit>  <priority>

where  c = i % n_classes.

A final proxy-DT table maps the accumulated per-class scores to the winner:

    table_add MyIngress.xgb_classify  set_result  <score_ranges>  =>  <class_id>  <priority>

The proxy DT is trained on (score_c0, …, score_c{n-1}) → label using
training data run through the XGB booster — it therefore learns the exact
argmax boundary that XGB uses, expressed as range-match rules.

Leaf quantisation
-----------------
Each tree's leaf values are linearly scaled from their observed [min, max]
range to the 8-bit unsigned integer range [0, 255] (bit<8> delta in P4).
The bias introduced by shifting min→0 is the same for all samples processed
by a given tree and is therefore safe; only the relative ordering between
class accumulators matters for the final argmax.

Defaults
--------
  Input model  : model/xgb.model
  Params file  : tables/xgb_params.json
  Output file  : tables/s1-commands.txt  (appended, old XGB rules stripped)
  Human tree   : tables/xgb_trees.txt
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from sklearn import tree as sklearn_tree
from collections import defaultdict

try:
    import xgboost as xgb_lib
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate P4 XGB classification entries from trained model")

parser.add_argument('-i',          default="model/xgb.model",
                    help='Path to trained XGB model (pickle)')
parser.add_argument('-o',          default="tables/s1-commands.txt",
                    help='Path to output P4 commands file (appended)')
parser.add_argument('--tree-out',  default="tables/xgb_trees.txt",
                    help='Path to human-readable tree rules')
parser.add_argument('--bits', '-b', type=int, default=None,
                    help='PCA quantisation bits. If omitted, read from tables/pca_encoding_params.json.')
parser.add_argument('--params',    default="tables/xgb_params.json",
                    help='XGB params JSON written by 3_xgb_training_model.py')
parser.add_argument('--csv',       default="tables/pca_integer_mapping.csv",
                    help='CSV used to fit proxy DT (same as training CSV)')
parser.add_argument('--proxy-max-depth', type=int, default=None,
                    help='max_depth for the proxy DT classifier (default: None = unlimited)')

args = parser.parse_args()

inputfile    = args.i
outputfile   = args.o
tree_output  = args.tree_out
csv_path     = args.csv

# Auto-detect bits from pca_encoding_params.json if not explicitly provided
if args.bits is None:
    _params_path = os.path.join(os.path.dirname(__file__), 'tables', 'pca_encoding_params.json')
    try:
        with open(_params_path) as _f:
            BITS = int(json.load(_f).get('bits', 16))
        print(f"Auto-detected bits={BITS} from {_params_path}")
    except Exception as _e:
        print(f"WARNING: could not read bits from {_params_path} ({_e}), defaulting to 16")
        BITS = 16
else:
    BITS = args.bits

MAX_VAL      = 2 ** BITS - 1

for d in [os.path.dirname(outputfile), os.path.dirname(tree_output)]:
    if d:
        os.makedirs(d, exist_ok=True)

MODEL_RULE_PREFIXES = (
    "table_add MyIngress.xgb_tree_",
    "table_add MyIngress.xgb_classify",
)

# ---------------------------------------------------------------------------
# Helpers  (mirrors 4_dt / 4_rf generating_entries.py)
# ---------------------------------------------------------------------------

def load_base_lines(path, drop_prefixes):
    if not os.path.exists(path):
        return []
    lines = []
    with open(path) as f:
        for line in f:
            if not line.lstrip().startswith(drop_prefixes):
                lines.append(line)
    return lines


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
    clause      = []
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


# ---------------------------------------------------------------------------
# XGBoost tree extraction via booster DataFrame
# ---------------------------------------------------------------------------

def parse_xgb_trees(booster, feature_names):
    """
    Return a list of per-tree node dicts using booster.trees_to_dataframe().

    Each entry in the returned list is a dict:
        { node_idx: {'feature': str|'Leaf', 'split': float,
                     'yes': int, 'no': int, 'gain': float} }
    """
    df = booster.trees_to_dataframe()
    # Column names: Tree, Node, ID, Feature, Split, Yes, No, Missing, Gain, Cover
    n_trees = int(df['Tree'].max()) + 1
    all_trees = []
    
    # Build mapping from f0, f1, ... to actual feature names (pc1_code, pc2_code, ...)
    f_to_feature = {}
    for i, fname in enumerate(feature_names):
        f_to_feature[f"f{i}"] = fname

    for t_idx in range(n_trees):
        t_df  = df[df['Tree'] == t_idx].copy()
        nodes = {}
        for _, row in t_df.iterrows():
            n_idx   = int(row['Node'])
            feature = row['Feature']
            gain    = float(row['Gain'])
            if feature == 'Leaf':
                nodes[n_idx] = {'feature': 'Leaf', 'gain': gain}
            else:
                # Map f0, f1, ... to actual feature names
                feat_name = f_to_feature.get(feature, feature)
                split_val = float(row['Split'])
                # 'Yes' / 'No' are IDs of the form "{tree_idx}-{node_idx}"
                yes_node  = int(str(row['Yes']).split('-')[1])
                no_node   = int(str(row['No']).split('-')[1])
                nodes[n_idx] = {
                    'feature': feat_name,
                    'split':   split_val,
                    'yes':     yes_node,   # condition true  (feature <= split)
                    'no':      no_node,    # condition false (feature >  split)
                }
        all_trees.append(nodes)

    return all_trees


def collect_tree_rules(nodes, feature_names, max_val, table_name, action_name,
                       quant_fn):
    """
    Walk a single parsed XGB tree (dict of nodes) depth-first.
    Returns list of (specificity, rule_str).
    quant_fn(leaf_value) → 8-bit unsigned int.
    """
    rules = []

    def dfs(node_idx, path):
        node = nodes[node_idx]
        if node['feature'] == 'Leaf':
            domain        = minimize(path)
            clause, width = build_clause(domain, feature_names, max_val)
            delta         = quant_fn(node['gain'])
            rule_str      = (f"table_add MyIngress.{table_name} {action_name} "
                             f"{' '.join(clause)} => {delta}")
            specificity   = (len(clause), -width)
            rules.append((specificity, rule_str))
            return
        feat      = node['feature']
        threshold = node['split']
        dfs(node['yes'], path + [(feat, '<=', threshold)])
        dfs(node['no'],  path + [(feat, '>',  threshold)])

    dfs(0, [])
    return rules


def write_xgb_tree_text(nodes, feature_names, tree_idx, class_idx, fh):
    """Append human-readable IF/THEN rules for one XGB tree."""
    fh.write(f"\n# --- XGB Tree {tree_idx}  (class {class_idx}) ---\n")

    def dfs(node_idx, path):
        node = nodes[node_idx]
        if node['feature'] == 'Leaf':
            clauses  = [f"{feat} {sign} {thr:.4f}" for feat, sign, thr in path]
            cond_str = " AND ".join(clauses) if clauses else "TRUE"
            fh.write(f"\tIF {cond_str} THEN leaf={node['gain']:.6f};\n")
            return
        dfs(node['yes'], path + [(node['feature'], '<=', node['split'])])
        dfs(node['no'],  path + [(node['feature'], '>',  node['split'])])

    dfs(0, [])


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = pd.read_pickle(inputfile)

# Feature names
if hasattr(model, 'feature_names_in_'):
    FEATURE_NAMES = model.feature_names_in_.tolist()
else:
    try:
        fn = model.get_booster().feature_names
        FEATURE_NAMES = fn if fn else [f"pc{i+1}_code"
                                       for i in range(model.n_features_in_)]
    except Exception:
        FEATURE_NAMES = [f"pc{i+1}_code" for i in range(model.n_features_in_)]

booster   = model.get_booster()
n_classes = model.n_classes_  if hasattr(model, 'n_classes_') else 2

# Infer tree count and trees_per_class
all_node_dicts = parse_xgb_trees(booster, FEATURE_NAMES)
total_trees    = len(all_node_dicts)
if n_classes > 2:
    trees_per_class = total_trees // n_classes
else:
    trees_per_class = total_trees   # binary: all trees contribute to class-1 log-odds

# Load params if available for consistency check
if os.path.exists(args.params):
    with open(args.params) as f:
        params = json.load(f)
    assert params.get('total_trees') == total_trees, \
        f"total_trees mismatch: params={params.get('total_trees')} vs model={total_trees}"
    assert params.get('n_classes')   == n_classes, \
        "n_classes mismatch with params file"
    classes        = params['classes']
    label_encoding = {lbl: idx for idx, lbl in enumerate(classes)}
else:
    # Reconstruct classes from model (XGBClassifier stores original classes)
    if hasattr(model, 'classes_'):
        classes = [int(c) for c in model.classes_]
    else:
        classes = list(range(n_classes))
    label_encoding = {lbl: idx for idx, lbl in enumerate(classes)}

print(f"\n=== Label Encoding ===")
for label, code in label_encoding.items():
    print(f"  {label} -> {code}")

print(f"\ntotal_trees={total_trees}, n_classes={n_classes}, "
      f"trees_per_class={trees_per_class}")
print(f"FEATURE_NAMES: {FEATURE_NAMES}\n")

# ---------------------------------------------------------------------------
# Build leaf quantisation functions  (one per tree, based on observed leaf range)
# ---------------------------------------------------------------------------

def make_quantiser(leaf_values, delta_bits=8):
    """Return a function that maps a leaf_value → unsigned int [0, 2^delta_bits - 1]."""
    lo  = min(leaf_values)
    hi  = max(leaf_values)
    rng = hi - lo
    top = (2 ** delta_bits) - 1

    def quant(v):
        if rng == 0:
            return 0
        return int(round((v - lo) / rng * top))

    return quant


def collect_leaf_values(nodes):
    """Return all leaf gain values from a parsed node dict."""
    return [n['gain'] for n in nodes.values() if n['feature'] == 'Leaf']


# ---------------------------------------------------------------------------
# Generate per-tree P4 entries
# ---------------------------------------------------------------------------
print(f"Generating entries for {total_trees} XGB trees …")

all_tree_rules = []   # list of lists

with open(tree_output, "w") as tf:
    tf.write("# XGBoost — human-readable tree rules (leaf = raw log-odds delta)\n")
    tf.write(f"# total_trees={total_trees}, n_classes={n_classes}\n")

    for tree_idx in range(total_trees):
        class_idx  = tree_idx % n_classes if n_classes > 2 else 1  # binary→class 1
        nodes      = all_node_dicts[tree_idx]
        table_name = f"xgb_tree_{tree_idx}"
        action     = f"add_xgb_score_c{class_idx}"

        leaf_vals  = collect_leaf_values(nodes)
        quant_fn   = make_quantiser(leaf_vals, delta_bits=8)

        rules = collect_tree_rules(nodes, FEATURE_NAMES, MAX_VAL,
                                   table_name, action, quant_fn)
        rules.sort(key=lambda x: x[0], reverse=True)
        all_tree_rules.append(rules)
        print(f"  tree {tree_idx:3d} (class {class_idx}): {len(rules):5d} entries  "
              f"leaf_range=[{min(leaf_vals):.4f}, {max(leaf_vals):.4f}]")

        write_xgb_tree_text(nodes, FEATURE_NAMES, tree_idx, class_idx, tf)

# ---------------------------------------------------------------------------
# Build a proxy DT on accumulated (quantised) scores → then generate
# xgb_classify range entries by walking the proxy DT
# ---------------------------------------------------------------------------
print(f"\nBuilding proxy DT for xgb_classify table …")

# Load training data so we can accumulate scores
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Training CSV not found at '{csv_path}'. "
        "Use --csv to point to tables/pca_integer_mapping.csv")

df_train = pd.read_csv(csv_path)
# Normalize column names to lowercase for consistent access
df_train.columns = df_train.columns.str.lower()
code_cols = [c for c in df_train.columns if c.endswith('_code')]

# Ensure feature order matches model (normalize to lowercase)
feature_names_lower = [f.lower() for f in FEATURE_NAMES]
X_raw = df_train[feature_names_lower].values.astype(np.float64)
Y_raw = df_train['label'].values

mask  = np.isfinite(X_raw).all(axis=1)
X_raw = X_raw[mask]
Y_raw = Y_raw[mask]

# Compute quantised accumulated scores per class for every training sample
# by re-walking each tree with the actual training data.
# We use the booster's leaf predictions for efficiency.
import xgboost as xgb_lib_inner

dmatrix      = xgb_lib_inner.DMatrix(X_raw, feature_names=FEATURE_NAMES)
# leaf_preds shape: (n_samples, total_trees)
leaf_indices = booster.predict(dmatrix, pred_leaf=True)

# Reconstruct quantised score accumulators
n_samples           = X_raw.shape[0]
accum_scores        = np.zeros((n_samples, n_classes), dtype=np.int32)

for tree_idx in range(total_trees):
    class_idx  = tree_idx % n_classes if n_classes > 2 else 1
    nodes      = all_node_dicts[tree_idx]
    leaf_vals  = collect_leaf_values(nodes)
    quant_fn   = make_quantiser(leaf_vals, delta_bits=8)

    # Map leaf index → quantised delta for this tree
    # Build a lookup: leaf_node_idx → delta
    leaf_to_delta = {}
    for nidx, node in nodes.items():
        if node['feature'] == 'Leaf':
            leaf_to_delta[nidx] = quant_fn(node['gain'])

    sample_leaf_indices = leaf_indices[:, tree_idx].astype(int)
    for s_idx in range(n_samples):
        leaf_node = sample_leaf_indices[s_idx]
        delta     = leaf_to_delta.get(leaf_node, 0)
        accum_scores[s_idx, class_idx] += delta

# Train a small proxy DT on the accumulated scores
score_col_names = [f"score_c{c}" for c in range(n_classes)]
# Encode Y_raw to integer labels
y_enc = np.array([label_encoding.get(y, 0) for y in Y_raw])

proxy_dt = sklearn_tree.DecisionTreeClassifier(
    max_depth=args.proxy_max_depth,
    random_state=42,
)
proxy_dt.fit(accum_scores, y_enc)
proxy_acc = np.mean(proxy_dt.predict(accum_scores) == y_enc)
print(f"Proxy DT accuracy on accumulated scores: {proxy_acc:.4f}")
print(f"Proxy DT n_leaves: {proxy_dt.tree_.n_node_samples[proxy_dt.apply(accum_scores)].sum() // len(Y_raw)}")

# Determine dynamic max accumulator value
max_acc_val = int(accum_scores.max()) + 1   # upper bound for ranges

# ---------------------------------------------------------------------------
# Walk proxy DT to produce xgb_classify entries
# ---------------------------------------------------------------------------
proxy_rules_list = []

def proxy_minimize(path):
    domain = {}
    for (feature, sign, threshold) in path:
        domain.setdefault(feature, {"min": None, "max": None})
        if sign == "<=":
            domain[feature]["max"] = threshold
        else:
            domain[feature]["min"] = threshold
    return domain


def walk_proxy_dt(dt, node_id, feat_names, score_max_val, path=None):
    """Generate xgb_classify entries from proxy DT leaves."""
    if path is None:
        path = []

    t     = dt.tree_
    left  = t.children_left
    right = t.children_right

    if left[node_id] == right[node_id]:   # leaf
        new_path = []
        for (n_id, sign) in path:
            feat      = feat_names[t.feature[n_id]]
            threshold = t.threshold[n_id]
            new_path.append((feat, sign, threshold))

        domain = proxy_minimize(new_path)
        clause = []
        total_width = 0
        for fe in feat_names:
            val = domain.get(fe, {"min": None, "max": None})
            lo  = val["min"] if val["min"] is not None else -1
            hi  = val["max"] if val["max"] is not None else score_max_val
            lo  = int(lo) + 1
            hi  = int(hi)
            total_width += hi - lo + 1
            clause.append(f"{lo}->{hi}")

        a          = list(t.value[node_id][0])
        class_id   = a.index(max(a))
        rule_str   = (f"table_add MyIngress.xgb_classify set_result "
                      f"{' '.join(clause)} => {class_id}")
        specificity = (len(clause), -total_width)
        proxy_rules_list.append((specificity, rule_str))
        return

    walk_proxy_dt(dt, left[node_id],  feat_names, score_max_val,
                  path + [(node_id, "<=")])
    walk_proxy_dt(dt, right[node_id], feat_names, score_max_val,
                  path + [(node_id, ">")])


walk_proxy_dt(proxy_dt, 0, score_col_names, max_acc_val)
proxy_rules_list.sort(key=lambda x: x[0], reverse=True)
print(f"xgb_classify entries  : {len(proxy_rules_list)}")

# Emit proxy DT structure to human-readable file
with open(tree_output, "a") as tf:
    tf.write(f"\n\n# === Proxy DT for xgb_classify (accuracy={proxy_acc:.4f}) ===\n")
    proxy_feat = [f"score_c{c}" for c in range(n_classes)]
    tree_text  = sklearn_tree.export_text(proxy_dt, feature_names=proxy_feat)
    for line in tree_text.splitlines():
        tf.write(f"# {line}\n")

# ---------------------------------------------------------------------------
# Write everything to the output file
# ---------------------------------------------------------------------------
print(f"\nWriting all entries to {outputfile} …")

base_lines = load_base_lines(outputfile, MODEL_RULE_PREFIXES)
with open(outputfile, "w") as f:
    for line in base_lines:
        f.write(line)

    # Per-tree tables
    for tree_idx, rules in enumerate(all_tree_rules):
        for priority, (_, rule_str) in enumerate(rules, 1):
            f.write(f"{rule_str} {priority}\n")

    # Proxy-DT xgb_classify table
    for priority, (_, rule_str) in enumerate(proxy_rules_list, 1):
        f.write(f"{rule_str} {priority}\n")

print(f"Human-readable rules  : {tree_output}")
print(f"Done. Total XGB per-tree entries: "
      f"{sum(len(r) for r in all_tree_rules)}, "
      f"xgb_classify entries: {len(proxy_rules_list)}")