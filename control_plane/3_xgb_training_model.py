#!/usr/bin/env python3
"""
Train an XGBoostClassifier using PCA code features and save model + metrics.

Input CSV  : tables/pca_integer_mapping.csv  (columns: PC1_code … PCk_code, Label)
Output model  : model/xgb.model
Output metrics: tables/xgb_metrics.json
Output params : tables/xgb_params.json   <- consumed by 5_generating_p4_code.py
                                           and 4_xgb_generating_entries.py

P4 Deployment Design
--------------------
For multi-class XGBoost with n_estimators rounds and n_classes classes the
booster produces exactly  n_estimators * n_classes  trees, ordered as:

    tree 0  → class 0,   tree 1  → class 1, …
    tree k  → class (k % n_classes)

Each tree outputs a raw leaf value (floating-point log-odds delta).
The entry generator quantises these to 8-bit unsigned integers [0..255]
(action: add_xgb_score_c{c}(bit<8> delta)).  Accumulated class scores
(bit<16>) are compared by a proxy DT in the final xgb_classify table.

To keep the number of tables manageable, keep n_estimators small:
  n_estimators=4, n_classes=4  →  16 tables   ✓
  n_estimators=8, n_classes=4  →  32 tables   ✓  (reasonable)
  n_estimators=16, n_classes=4 →  64 tables   ✗  (large but possible)
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is required: pip install xgboost")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train XGBoostClassifier on PCA code features")

parser.add_argument('-i', default="tables/pca_integer_mapping.csv",
                    help='Path to input mapping CSV')
parser.add_argument('-o', default="model/xgb.model",
                    help='Path to output model file')
parser.add_argument('--test-size',        type=float, default=0.3)
parser.add_argument('--random-state',     type=int,   default=42)
parser.add_argument('--n-estimators',     type=int,   default=8,
                    help='Number of boosting rounds per class (default: 8; '
                         'total P4 trees = n_estimators * n_classes)')
parser.add_argument('--max-depth',        type=int,   default=3,
                    help='XGBoost max_depth (default: 3; shallower = fewer P4 entries)')
parser.add_argument('--learning-rate',    type=float, default=0.3,
                    help='XGBoost learning_rate / eta (default: 0.3)')
parser.add_argument('--subsample',        type=float, default=1.0,
                    help='XGBoost subsample ratio (default: 1.0)')
parser.add_argument('--min-child-weight', type=int,   default=1,
                    help='XGBoost min_child_weight (default: 1)')

args = parser.parse_args()

inputfile        = args.i
outputfile       = args.o
random_state     = args.random_state
n_estimators     = args.n_estimators
max_depth        = args.max_depth
learning_rate    = args.learning_rate
subsample        = args.subsample
min_child_weight = args.min_child_weight

# ---------------------------------------------------------------------------
# Ensure output dirs exist
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(outputfile)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

tables_dir   = os.path.join(os.path.dirname(__file__), 'tables')
os.makedirs(tables_dir, exist_ok=True)

metrics_path = os.path.join(tables_dir, 'xgb_metrics.json')
params_path  = os.path.join(tables_dir, 'xgb_params.json')

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

# Train on 100% of data (consistent with DT / RF pipeline)
X_train = X_test = X
y_train = y_test = Y

# ---------------------------------------------------------------------------
# Encode labels to integers (XGBoost requires integer targets)
# ---------------------------------------------------------------------------
classes        = sorted(np.unique(Y))
label_to_int   = {label: idx for idx, label in enumerate(classes)}
int_to_label   = {idx: label for label, idx in label_to_int.items()}
y_train_enc    = np.array([label_to_int[y] for y in y_train])
y_test_enc     = np.array([label_to_int[y] for y in y_test])

n_classes = len(classes)
objective = 'binary:logistic' if n_classes == 2 else 'multi:softmax'

print(f"Classes      : {classes}")
print(f"n_classes    : {n_classes}")
print(f"objective    : {objective}")
print(f"n_estimators : {n_estimators}")
print(f"max_depth    : {max_depth}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    min_child_weight=min_child_weight,
    objective=objective,
    num_class=(n_classes if n_classes > 2 else None),
    random_state=random_state,
    use_label_encoder=False,
    eval_metric='logloss' if n_classes == 2 else 'mlogloss',
    verbosity=0,
)
xgb.fit(X_train, y_train_enc)

# feature_names_in_ is automatically set by fit() in newer scikit-learn versions

# ---------------------------------------------------------------------------
# Evaluate  (decode back to original string labels for the report)
# ---------------------------------------------------------------------------
y_pred_enc = xgb.predict(X_test)
y_pred     = np.array([int_to_label[int(p)] for p in y_pred_enc])

acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred, labels=classes)
report = classification_report(y_test, y_pred, labels=classes, output_dict=True)

misclassified = y_pred != y_test
if np.any(misclassified):
    print(f"\nMisclassified samples (up to 20):")
    for i in np.where(misclassified)[0][:20]:
        print(f"  Sample {i}: true={y_test[i]}, pred={y_pred[i]}")
    print(f"Total misclassified: {misclassified.sum()} / {len(y_test)}")
else:
    print("\nNo misclassified samples — perfect accuracy.")

print("Accuracy:", acc)
print("Labels:",   classes)
print("Confusion Matrix:\n", cm)

# ---------------------------------------------------------------------------
# P4 deployment parameters
# ---------------------------------------------------------------------------
# For multi-class: tree i belongs to class (i % n_classes)
# For binary:      all trees belong to class 1 (treated as class 1 scorer)
if n_classes > 2:
    total_trees = n_estimators * n_classes
    trees_per_class = n_estimators
else:
    total_trees     = n_estimators
    trees_per_class = n_estimators

# Each tree leaf value is quantised to an 8-bit unsigned int [0..255].
# The per-class 16-bit accumulator can hold up to 255 * n_estimators.
max_score_per_tree = 255
max_accum_per_class = max_score_per_tree * trees_per_class

print(f"\n=== P4 deployment parameters ===")
print(f"  total_trees         : {total_trees}")
print(f"  trees_per_class     : {trees_per_class}")
print(f"  max_accum_per_class : {max_accum_per_class}  (fits in bit<16>: {max_accum_per_class < 2**16})")

if total_trees > 64:
    print(f"WARNING: {total_trees} tables may be too many for hardware P4 targets.")
    print("         Consider reducing --n-estimators.")

# ---------------------------------------------------------------------------
# Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "labels":                  classes,
    "accuracy":                float(acc),
    "confusion_matrix":        cm.tolist(),
    "classification_report":   report,
    "feature_names":           code_columns,
    "n_estimators":            n_estimators,
    "max_depth":               int(max_depth),
    "learning_rate":           float(learning_rate),
    "random_state":            int(random_state),
}
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved metrics to {metrics_path}")

# ---------------------------------------------------------------------------
# Save P4 params  (consumed by 5_generating_p4_code.py)
# ---------------------------------------------------------------------------
params = {
    "model_type":       "xgb",
    "n_estimators":     n_estimators,
    "n_classes":        n_classes,
    "trees_per_class":  trees_per_class,
    "total_trees":      total_trees,
    "classes":          classes,
    "feature_names":    code_columns,
    "objective":        objective,
    "max_depth":        int(max_depth),
    # quantisation: 8-bit delta per leaf; 16-bit accumulator per class
    "delta_bits":       8,
    "accum_bits":       16,
}
with open(params_path, 'w') as f:
    json.dump(params, f, indent=2)
print(f"Saved XGB P4 params to {params_path}")

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
with open(outputfile, 'wb') as f:
    pickle.dump(xgb, f)
print(f"Model saved to {outputfile}")