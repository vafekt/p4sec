#!/usr/bin/env python3
"""
Generate P4 table_add entries from a trained DecisionTreeClassifier.

Defaults:
 - Input model:  model/dt.model
 - Output file:  tables/s1-commands.txt

Each path from root to leaf is converted to:
    table_add MyIngress.ml_code set_result <ranges> => <class_id> <priority>
where class_id is a numeric encoding of the model classes.
"""

import os
import pandas as pd
import argparse



parser = argparse.ArgumentParser(description="Generate P4 DT classification entries from trained model")

# Add arguments
parser.add_argument('-i', default="model/dt.model", help='Path to the input DecisionTree model')
parser.add_argument('-o', default="tables/s1-commands.txt", help='Path to the output P4 commands file')
parser.add_argument('--tree-out', default="tables/dt_tree.txt", help='Path to the human-readable tree structure')
parser.add_argument('--bits', '-b', type=int, default=16,
                    help='Quantization bits for PCA codes (default: 16). Supports 8, 16, 24, 32 bits.')

args = parser.parse_args()
inputfile  = args.i
outputfile = args.o
tree_output = args.tree_out
BITS = args.bits
MAX_VAL = 2**BITS - 1  # Dynamic max value based on BITS (e.g., 255 for 8-bit, 65535 for 16-bit, 4294967295 for 32-bit)

# Validate BITS parameter
if BITS not in [8, 16, 24, 32]:
    print(f"WARNING: BITS={BITS}. Recommended values are 8, 16, 24, 32 for P4 compatibility.")
    print(f"P4 range match can handle arbitrary widths, but ensure switch pipeline width supports it.")

print(f"Quantization bits: {BITS}, MAX_VAL: {MAX_VAL}")
print()
out_dir = os.path.dirname(outputfile)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
# Ensure tree output directory exists
tree_out_dir = os.path.dirname(tree_output)
if tree_out_dir:
    os.makedirs(tree_out_dir, exist_ok=True)


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


priority = 0
rules_list = []  # Collect all rules before assigning priorities

def write_entry(f, domain, classification):
    """Collect rule entry (don't write immediately - priorities will be assigned later)."""
    clause = []
    total_width = 0  # Calculate specificity as total range width

    for fe in FEATURE_NAMES:
        if fe not in domain:
            continue
        val = domain[fe]
        lo = val["min"]
        hi = val["max"]

        # Skip unconstrained feature
        if lo is None and hi is None:
            continue

        # Defaults for missing bounds
        if lo is None:
            lo = -1  # will become 0 after +1
        if hi is None:
            hi = MAX_VAL  # Use dynamic max value based on BITS parameter
        
        # Decision tree uses (lo, hi]; convert to [lo+1, hi]
        lo = int(lo) + 1
        hi = int(hi)
        
        # Calculate range width (narrower = more specific)
        width = hi - lo + 1
        total_width += width

        clause.append(f"{lo}->{hi}")

    rule_str = f"table_add MyIngress.ml_code set_result {' '.join(clause)} => {classification}"
    # Store rule with tuple specificity: (num_constraints, -total_width)
    # More constrained features = higher priority (lower priority number)
    # Narrower ranges = higher priority (lower priority number)
    num_constraints = len(clause)
    specificity = (num_constraints, -total_width) if clause else (0, 2**32)
    rules_list.append((specificity, rule_str))


def visit_commands(dt, node_id, features, file=None, path=None):
    if path is None:
        path = []

    classes = dt.classes_
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right

    is_leaf = (left[node_id] == right[node_id])

    if is_leaf:
        new_path = []
        for (n_id, sign) in path:
            threshold = tree_.threshold[n_id]
            feature = features[n_id]
            new_path.append((feature, sign, threshold))

        clause = minimize(new_path)

        a = list(tree_.value[node_id][0])
        class_index = a.index(max(a))
        classification = class_index  # numeric encoding

        write_entry(file, clause, classification)
        return

    # Left branch (<=)
    left_path = path.copy()
    left_path.append((node_id, "<="))
    visit_commands(dt, left[node_id], features, file, left_path)

    # Right branch (>)
    right_path = path.copy()
    right_path.append((node_id, ">"))
    visit_commands(dt, right[node_id], features, file, right_path)


def visit_tree_text(dt, node_id, features, thresholds, file, path=None):
    """Depth-first traversal to emit IF/THEN rules with class labels."""
    if path is None:
        path = []

    classes = dt.classes_
    tree_ = dt.tree_
    left = tree_.children_left
    right = tree_.children_right

    is_leaf = (left[node_id] == right[node_id])

    if is_leaf:
        clauses = []
        for (n_id, sign) in path:
            feat_name = features[n_id]
            thr = thresholds[n_id]
            clauses.append(f"{feat_name} {sign} {thr}")

        a = list(tree_.value[node_id][0])
        class_index = a.index(max(a))
        classification = classes[class_index]

        clause_str = " and ".join(clauses) if clauses else "TRUE"
        file.write(f"\tIF {clause_str} THEN {classification};\n")
        return

    # Left branch (<=)
    left_path = path.copy()
    left_path.append((node_id, "<="))
    visit_tree_text(dt, left[node_id], features, thresholds, file, left_path)

    # Right branch (>)
    right_path = path.copy()
    right_path.append((node_id, ">"))
    visit_tree_text(dt, right[node_id], features, thresholds, file, right_path)


# structure of model: DecisionTreeClassifier
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
dt = pd.read_pickle( inputfile )

# Get feature names from the model (if available), otherwise use default names
if hasattr(dt, 'feature_names_in_'):
    FEATURE_NAMES = dt.feature_names_in_.tolist()
else:
    # Fallback: generate feature names based on number of features
    n_features = dt.n_features_in_ if hasattr(dt, 'n_features_in_') else dt.tree_.n_features
    FEATURE_NAMES = [f"feature_{i}" for i in range(n_features)]

# Prepare feature list aligned with tree_.feature order
features = [FEATURE_NAMES[i] for i in dt.tree_.feature]
thresholds = dt.tree_.threshold.tolist()

# Label encoding info
label_encoding = {label: idx for idx, label in enumerate(dt.classes_)}
print("\n=== Label Encoding ===")
for label, code in label_encoding.items():
    print(f"  {label} -> {code}")
print()

print("write output to", outputfile)
# Collect all rules first
rules_list.clear()
visit_commands(dt, 0, features, None)

# Sort rules by specificity (descending): first by num_constraints (desc), then by range width (asc)
# More constrained = higher specificity
# Narrower ranges = higher specificity  
rules_list.sort(key=lambda x: x[0], reverse=True)

# Write rules with assigned priorities
with open(outputfile, "a") as f:
    for priority_num, (specificity, rule_str) in enumerate(rules_list, 1):
        f.write(f"{rule_str} {priority_num}\n")

# Also emit a human-readable tree with thresholds and class labels
print("write tree structure to", tree_output)
with open(tree_output, "a") as f:
    visit_tree_text(dt, 0, features, thresholds, f)
