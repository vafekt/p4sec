#!/usr/bin/env python3
"""
Train a RandomForestClassifier using PCA code features and save model + metrics.

Input CSV  : tables/pca_integer_mapping.csv  (columns: PC1_code … PCk_code, Label)
Output model : model/rf.model
Output metrics: tables/rf_metrics.json
Output params : tables/rf_params.json   <- consumed by 5_generating_p4_code.py

Design notes
------------
For P4 deployment each tree in the forest becomes its own match-action table
(rf_tree_0 … rf_tree_{N-1}).  Votes from all trees are packed into a single
metadata integer whose width is  N * ceil(log2(n_classes))  bits.
A pre-computed vote-aggregation table maps every possible packed-vote value
to the majority class.

Keeping n_estimators small (≤ 16) keeps the vote table tractable:
  n_estimators=8, n_classes=4  → vote_bits=2 → 2^16 = 65 536 entries  ✓
  n_estimators=16, n_classes=4 → vote_bits=2 → 2^32 = 4 G entries      ✗
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train RandomForestClassifier on PCA code features")

parser.add_argument('-i', default="tables/pca_integer_mapping.csv",
                    help='Path to input mapping CSV')
parser.add_argument('-o', default="model/rf.model",
                    help='Path to output model file')
parser.add_argument('--test-size',       type=float, default=0.3)
parser.add_argument('--random-state',    type=int,   default=42)
parser.add_argument('--n-estimators',    type=int,   default=8,
                    help='Number of trees (default: 8; keep ≤16 for P4 vote table)')
parser.add_argument('--max-depth',       type=int,   default=None)
parser.add_argument('--min-samples-leaf',type=int,   default=1)

args = parser.parse_args()

inputfile        = args.i
outputfile       = args.o
random_state     = args.random_state
n_estimators     = args.n_estimators
max_depth        = args.max_depth
min_samples_leaf = args.min_samples_leaf

# ---------------------------------------------------------------------------
# Ensure output dirs exist
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(outputfile)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

tables_dir   = os.path.join(os.path.dirname(__file__), 'tables')
os.makedirs(tables_dir, exist_ok=True)

metrics_path = os.path.join(tables_dir, 'rf_metrics.json')
params_path  = os.path.join(tables_dir, 'rf_params.json')

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(inputfile)

code_columns = [col for col in df.columns if col.endswith('_code')]
if not code_columns:
    raise ValueError("No *_code feature columns found in input CSV")
if 'Label' not in df.columns:
    raise ValueError("Target column 'Label' not found in input CSV")

X = np.asarray(df[code_columns].values, dtype=np.float64)
Y = np.asarray(df['Label'].values)

# Clean NaN / Inf
mask = np.isfinite(X).all(axis=1)
X, Y = X[mask], Y[mask]

# Train on 100 % of data (consistent with DT pipeline)
X_train = X_test = X
y_train = y_test = Y

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=random_state,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# feature_names_in_ is automatically set by fit() in newer scikit-learn versions

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
y_pred   = rf.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
labels   = sorted(np.unique(Y))
cm       = confusion_matrix(y_test, y_pred, labels=labels)
report   = classification_report(y_test, y_pred, labels=labels, output_dict=True)

misclassified = y_pred != y_test
if np.any(misclassified):
    print(f"\nMisclassified samples (up to 20):")
    for i in np.where(misclassified)[0][:20]:
        print(f"  Sample {i}: true={y_test[i]}, pred={y_pred[i]}")
    print(f"Total misclassified: {misclassified.sum()} / {len(y_test)}")
else:
    print("\nNo misclassified samples — perfect accuracy.")

print("Accuracy:", acc)
print("Labels:", labels)
print("Confusion Matrix:\n", cm)

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "labels":                  labels,
    "accuracy":                float(acc),
    "confusion_matrix":        cm.tolist(),
    "classification_report":   report,
    "feature_names":           code_columns,
    "n_estimators":            n_estimators,
    "max_depth":               (None if max_depth is None else int(max_depth)),
    "min_samples_leaf":        int(min_samples_leaf),
    "random_state":            int(random_state),
}
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics to", metrics_path)

# ---------------------------------------------------------------------------
# Compute P4 vote packing parameters and save params file
# ---------------------------------------------------------------------------
n_classes  = len(labels)
vote_bits  = max(1, math.ceil(math.log2(n_classes))) if n_classes > 1 else 1
total_vote_bits = n_estimators * vote_bits
vote_table_entries = 2 ** total_vote_bits

print(f"\n=== P4 vote-packing parameters ===")
print(f"  n_estimators    : {n_estimators}")
print(f"  n_classes       : {n_classes}")
print(f"  vote_bits       : {vote_bits}")
print(f"  total_vote_bits : {total_vote_bits}")
print(f"  vote_table_entries : {vote_table_entries:,}")

if total_vote_bits > 24:
    print(f"WARNING: {vote_table_entries:,} vote-table entries may be too large for hardware P4 targets.")
    print("         Consider reducing --n-estimators or using fewer label classes.")

params = {
    "model_type":        "rf",
    "n_estimators":      n_estimators,
    "n_classes":         n_classes,
    "vote_bits":         vote_bits,
    "total_vote_bits":   total_vote_bits,
    "classes":           labels,
    "feature_names":     code_columns,
}
with open(params_path, 'w') as f:
    json.dump(params, f, indent=2)
print("Saved RF P4 params to", params_path)

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
with open(outputfile, 'wb') as f:
    pickle.dump(rf, f)
print("Model saved to", outputfile)