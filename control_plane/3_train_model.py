#!/usr/bin/env python3
"""
ML classifier training for the P4 in-network classification pipeline.

Supports two classifier back-ends via --model-type:
  dt    DecisionTree   (sklearn)
  rf    RandomForest   (sklearn)

Works with any step-2 reduction method (PCA / LDA / Autoencoder).
Feature columns are auto-detected from tables/reduction_config.json.

Input:   tables/transform_mapping.csv      (written by any step 2)
Output:  model/<model_type>.model          (pickled sklearn model)
         tables/model_metrics.json         (accuracy, confusion matrix)
         tables/model_params.json          (P4 deployment parameters, RF only)
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pipeline_utils import detect_feature_columns

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
    description="Train DT/RF classifier for P4 deployment",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Model-specific option prefixes:\n"
        "  dt/rf : --max-depth --min-samples-leaf --class-weight\n"
        "  rf    : --n-estimators\n"
    ),
)
parser.add_argument('--model-type', '-m', required=True,
                    choices=['dt', 'rf'],
                    help='Classifier: dt | rf')
parser.add_argument('-i', default="tables/transform_mapping.csv",
                    help='Input CSV (default: tables/transform_mapping.csv)')
parser.add_argument('-o', default=None,
                    help='Output model path (default: model/<model_type>.model)')

# Shared hyperparams
parser.add_argument('--random-state',     type=int, default=42)
parser.add_argument('--max-depth',        type=int, default=None,
                    help='Tree max depth (default: None — unlimited)')
parser.add_argument('--min-samples-leaf', type=int, default=1)

# RF-specific
parser.add_argument('--max-features', default='sqrt',
                    help=(
                        "RF per-split feature subset.  sklearn default = 'sqrt' "
                        "(~4 of 18 features per split) weakens each tree.  "
                        "Use 'all' / 'none' to consider every feature at each "
                        "split (stronger trees, less variance reduction across "
                        "trees), or a float fraction like 0.8.  default: sqrt"
                    ))
parser.add_argument('--n-estimators',     type=int, default=4,
                    help='Number of trees for RF (default: 4)')

# Class weighting for imbalanced datasets
parser.add_argument('--class-weight', default='balanced',
                    choices=['balanced', 'none'],
                    help='Class weighting for DT/RF (default: balanced — '
                         'strongly recommended for imbalanced datasets)')

args = parser.parse_args()

model_type = args.model_type
inputfile  = args.i
outputfile = args.o or f"model/{model_type}.model"

tables_dir = os.path.join(os.path.dirname(__file__), 'tables')
for d in [os.path.dirname(outputfile), tables_dir]:
    if d:
        os.makedirs(d, exist_ok=True)


# ─── Load dataset ────────────────────────────────────────────────────────
df = pd.read_csv(inputfile)
feature_columns = detect_feature_columns(inputfile, tables_dir)
print(f"Model type      : {model_type.upper()}")
print(f"Feature columns : {feature_columns}")

assert all(c in df.columns for c in feature_columns), \
    f"Missing columns: {[c for c in feature_columns if c not in df.columns]}"
assert 'Label' in df.columns, "No 'Label' column in CSV"

X = pd.DataFrame(df[feature_columns].values, columns=feature_columns, dtype=np.float64)
Y = np.asarray(df['Label'].values)
mask = np.isfinite(X.values).all(axis=1)
X, Y = X[mask].reset_index(drop=True), Y[mask]

print(f"Samples         : {X.shape[0]}")
print(f"Features        : {X.shape[1]}")

# ── Holdout split (evaluation only, 80/20 stratified) ────────────────────
# The PRODUCTION model is trained on 100% of data so P4 table rules cover
# all known patterns.  We hold out 20% only to report honest out-of-sample
# accuracy before retraining on the full set.
_stratify = Y if len(np.unique(Y)) > 1 else None
X_hold_eval, X_hold_test, y_hold_eval, y_hold_test = train_test_split(
    X, Y, test_size=0.2, random_state=args.random_state,
    stratify=_stratify)
print(f"\nHoldout evaluation split: {len(y_hold_eval)} train / {len(y_hold_test)} test")

# Train on 100% of data for production deployment
X_train = X_test = X
y_train = y_test = Y

labels    = sorted(np.unique(Y))
n_classes = len(labels)
max_depth = args.max_depth


# ─── Training helper ─────────────────────────────────────────────────────
def train_model(mt, X_train_in, y_train_in, X_test_in):
    """Train DT or RF and return (model, y_pred)."""
    tree_class_weight = None if args.class_weight == 'none' else 'balanced'

    if mt == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        model_local = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight=tree_class_weight,
        )
    elif mt == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        # max_features: sklearn default is "sqrt".  For 18 features that limits
        # each split to ~4 random features which weakens each tree heavily on
        # this dataset, where the discriminative signal is concentrated in a
        # handful of features (ports, packet counts).  Allow the user to widen
        # via --max-features.
        mf = getattr(args, "max_features", None)
        if mf in (None, "default", "sqrt"):
            mf = "sqrt"
        elif mf in ("all", "none"):
            mf = None
        else:
            try: mf = float(mf)
            except: pass
        model_local = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight=tree_class_weight,
            max_features=mf,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model type: {mt}")

    model_local.fit(X_train_in, y_train_in)
    y_pred_local = model_local.predict(X_test_in)
    return model_local, y_pred_local


# ─── Train ───────────────────────────────────────────────────────────────
print(f"\nTraining {model_type.upper()}...")
model, y_pred = train_model(model_type, X_train, y_train, X_test)


# ─── Holdout evaluation (honest out-of-sample accuracy) ──────────────────
_hold_model, _hold_pred = train_model(
    model_type, X_hold_eval, y_hold_eval, X_hold_test)
_hold_acc    = accuracy_score(y_hold_test, _hold_pred)
_hold_labels = sorted(np.unique(Y))
_hold_cm     = confusion_matrix(y_hold_test, _hold_pred, labels=_hold_labels)
_hold_rep    = classification_report(y_hold_test, _hold_pred,
                                     labels=_hold_labels, output_dict=True,
                                     zero_division=0)
print(f"\n{'='*60}")
print(f"HOLDOUT accuracy (80/20 split, UNSEEN test set): {_hold_acc:.4f}")
print(f"Holdout Confusion Matrix:\n{_hold_cm}")
for cls in _hold_labels:
    r = _hold_rep.get(cls, {})
    print(f"  {cls:10s}: prec={r.get('precision',0):.3f}  "
          f"rec={r.get('recall',0):.3f}  "
          f"f1={r.get('f1-score',0):.3f}  "
          f"support={int(r.get('support',0))}")
print(f"{'='*60}\n")
del _hold_model  # free memory before retraining on full data

# ─── Evaluate (production model, train==test) ─────────────────────────────
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred, labels=labels)
report = classification_report(y_test, y_pred, labels=labels,
                               output_dict=True, zero_division=0)

mis = y_pred != y_test
if np.any(mis):
    print(f"Misclassified on training set (up to 5):")
    for i in np.where(mis)[0][:5]:
        print(f"  {i}: true={y_test[i]}, pred={y_pred[i]}")
    print(f"Total misclassified on training set: {mis.sum()} / {len(y_test)}")
else:
    print("Perfect fit on training set (expected — model trained on all data).")

print(f"Training accuracy: {acc:.4f}")
print(f"Labels:   {labels}")
print(f"Confusion Matrix:\n{cm}")


# ─── Save metrics ────────────────────────────────────────────────────────
metrics = {
    "model_type":                    model_type,
    "labels":                        labels,
    "accuracy":                      float(acc),
    "confusion_matrix":              cm.tolist(),
    "classification_report":         report,
    "feature_names":                 feature_columns,
    "n_estimators":                  args.n_estimators if model_type == 'rf' else None,
    "max_depth":                     max_depth,
    "class_weight":                  args.class_weight,
    "holdout_accuracy":              float(_hold_acc),
    "holdout_confusion_matrix":      _hold_cm.tolist(),
    "holdout_classification_report": _hold_rep,
}
metrics_path = os.path.join(tables_dir, 'model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to {metrics_path}")


# ─── Save P4 deployment parameters (RF only) ─────────────────────────────
if model_type == 'rf':
    # RF uses the P4 vote architecture (per-tree vote + aggregation)
    vote_bits          = max(1, math.ceil(math.log2(n_classes))) if n_classes > 1 else 1
    total_vote_bits    = args.n_estimators * vote_bits
    vote_table_entries = 2 ** total_vote_bits

    print(f"\nP4 vote packing: n_estimators={args.n_estimators}, "
          f"vote_bits={vote_bits}, total={total_vote_bits}, "
          f"vote_table={vote_table_entries:,} entries")
    if total_vote_bits > 24:
        print("WARNING: vote table may be too large for hardware targets.")

    params = {
        "model_type":      model_type,
        "n_estimators":    args.n_estimators,
        "n_classes":       n_classes,
        "vote_bits":       vote_bits,
        "total_vote_bits": total_vote_bits,
        "classes":         labels,
        "feature_names":   feature_columns,
    }
    ppath = os.path.join(tables_dir, 'model_params.json')
    with open(ppath, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved P4 params to {ppath}")


# ─── Save model ──────────────────────────────────────────────────────────
with open(outputfile, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {outputfile}")
