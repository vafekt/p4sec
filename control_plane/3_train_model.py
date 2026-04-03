#!/usr/bin/env python3
"""
Universal ML classifier training for the P4 in-network classification pipeline.

Supports seven classifier back-ends via --model-type:
  dt    DecisionTree           (sklearn)
  rf    RandomForest           (sklearn)
  xgb   XGBoost                (requires xgboost)
  gb    GradientBoosting       (sklearn)  — deploys as XGB in P4
  knn   K-Nearest Neighbors    (sklearn, software-only)
  svm   Support Vector Machine (sklearn, software-only)
  cnn   1D CNN                  (PyTorch, software-only)

Works with any step-2 reduction method (PCA / LDA / Autoencoder / UMAP / Feature Selection).
Feature columns are auto-detected from tables/reduction_config.json.

Input:   tables/transform_mapping.csv      (written by any step 2)
Output:  model/<model_type>.model         (pickled sklearn/xgb model)
         tables/<model_type>_metrics.json  (accuracy, confusion matrix)
         tables/<model_type>_params.json   (P4 deployment parameters — RF/XGB/GB only)
         Optional: when --model-type cnn --distill-to <type>,
                   writes model/<type>.model and matching metrics/params for P4 deployment.
"""

import os
import sys
import json
import math
import argparse
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pipeline_utils import detect_feature_columns, detect_bits, load_reduction_config, P4_FEATURE_MAX

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
    description="Train ML classifier for P4 deployment (universal)",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Model-specific option prefixes:\n"
        "  dt/rf : --max-depth --min-samples-leaf\n"
        "  rf    : --n-estimators\n"
        "  xgb   : --n-estimators --learning-rate --subsample --min-child-weight\n"
        "  gb    : --n-estimators --learning-rate --subsample --loss\n"
        "  knn   : --knn-*\n"
        "  svm   : --svm-*\n"
        "  cnn   : --epochs --batch-size --lr --cnn-* --dropout --p4-* --distill-to\n"
        "\n"
        "Notes:\n"
        "  - KNN/SVM are software-only (skip step 4/5).\n"
        "  - GB deploys with the XGB P4 architecture.\n"
    )
)
parser.add_argument('--model-type', '-m', required=True,
                    choices=['dt', 'rf', 'xgb', 'gb', 'knn', 'svm', 'cnn'],
                    help='Classifier: dt | rf | xgb | gb | knn | svm | cnn')
parser.add_argument('-i', default="tables/transform_mapping.csv",
                    help='Input CSV (default: tables/transform_mapping.csv)')
parser.add_argument('-o', default=None,
                    help='Output model path (default: model/<model_type>.model)')

# Shared hyperparams
parser.add_argument('--random-state',     type=int,   default=42)
parser.add_argument('--max-depth',        type=int,   default=None,
                    help='Tree max depth (default: None for DT/RF, 3 for XGB/GB)')
parser.add_argument('--min-samples-leaf', type=int,   default=1)

# Ensemble hyperparams (RF, XGB, GB)
parser.add_argument('--n-estimators',     type=int,   default=4,
                    help='Number of trees/rounds (default: 4)')

# Boosting hyperparams (XGB, GB)
parser.add_argument('--learning-rate',    type=float, default=None,
                    help='Learning rate (default: 0.3 for XGB, 0.1 for GB)')
parser.add_argument('--subsample',        type=float, default=1.0)
parser.add_argument('--min-child-weight', type=int,   default=1,
                    help='XGB min_child_weight (default: 1)')

# GB-specific
parser.add_argument('--loss', default='log_loss',
                    choices=['log_loss', 'exponential'],
                    help='GB loss function (default: log_loss)')

# KNN-specific
parser.add_argument('--knn-k', type=int, default=5,
                    help='KNN neighbors (default: 5)')
parser.add_argument('--knn-weights', default='uniform',
                    choices=['uniform', 'distance'],
                    help='KNN weighting (default: uniform)')
parser.add_argument('--knn-metric', default='minkowski',
                    help='KNN distance metric (default: minkowski)')
parser.add_argument('--knn-p', type=int, default=2,
                    help='KNN minkowski power (default: 2)')

# SVM-specific
parser.add_argument('--svm-kernel', default='rbf',
                    choices=['linear', 'poly', 'rbf', 'sigmoid'],
                    help='SVM kernel (default: rbf)')
parser.add_argument('--svm-c', type=float, default=1.0,
                    help='SVM regularization C (default: 1.0)')
parser.add_argument('--svm-gamma', default='scale',
                    help='SVM gamma (default: scale; or "auto")')
parser.add_argument('--svm-degree', type=int, default=3,
                    help='SVM polynomial degree (default: 3)')
parser.add_argument('--svm-class-weight', default='none',
                    choices=['balanced', 'none'],
                    help='SVM class weighting (default: none)')

# DT / RF class weighting (separate from SVM)
parser.add_argument('--class-weight', default='balanced',
                    choices=['balanced', 'none'],
                    help='Class weighting for DT/RF (default: balanced — strongly recommended '
                         'for imbalanced datasets; use none to disable)')

# CNN-specific (PyTorch)
parser.add_argument('--epochs', type=int, default=30,
                    help='CNN training epochs (default: 30)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='CNN batch size (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='CNN learning rate (default: 1e-3)')
parser.add_argument('--cnn-channels', type=str, default='16,32',
                    help='CNN conv channels, comma-separated (default: 16,32)')
parser.add_argument('--cnn-kernel', type=int, default=3,
                    help='CNN kernel size (default: 3)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='CNN dropout (default: 0.0)')
parser.add_argument('--cnn-class-weight', type=str, default='balanced',
                    choices=['balanced', 'none'],
                    help='CNN class weighting (default: balanced)')
parser.add_argument('--cnn-label-smoothing', type=float, default=0.0,
                    help='CNN label smoothing (default: 0.0)')
parser.add_argument('--distill-to', type=str, default=None,
                    choices=['dt', 'rf', 'xgb', 'gb'],
                    help='For --model-type cnn only: train a deployable surrogate (dt|rf|xgb|gb) '
                         'on CNN predictions and save as model/<type>.model')
parser.add_argument('--p4-export', action='store_true',
                    help='For --model-type cnn: export a P4-deployable neural model '
                         '(writes tables/cnn_params.json)')
parser.add_argument('--p4-hidden', type=int, default=16,
                    help='CNN P4 hidden units (default: 16)')
parser.add_argument('--p4-hidden2', type=int, default=None,
                    help='CNN P4 second hidden units (default: p4_hidden//2)')
parser.add_argument('--p4-pool', type=int, default=2,
                    help='CNN P4 maxpool size over hidden1 (default: 2, or 1 for no pooling)')
parser.add_argument('--p4-input-bits', type=int, default=None,
                    help='CNN P4 input quantization bits (default: auto)')
parser.add_argument('--p4-hidden-bits', type=int, default=8,
                    help='CNN P4 hidden activation bits (default: 8)')
parser.add_argument('--p4-w1-scale', type=float, default=64.0,
                    help='CNN P4 weight scale for first layer (default: 64.0)')
parser.add_argument('--p4-w2-scale', type=float, default=64.0,
                    help='CNN P4 weight scale for second layer (default: 64.0)')
parser.add_argument('--p4-w3-scale', type=float, default=64.0,
                    help='CNN P4 weight scale for output layer (default: 64.0)')

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
# all known patterns.  But to report honest out-of-sample accuracy we hold
# out 20% and measure on that before retraining on the full set.
from sklearn.model_selection import train_test_split as _tts
_stratify = Y if len(np.unique(Y)) > 1 else None
X_hold_eval, X_hold_test, y_hold_eval, y_hold_test = _tts(
    X, Y, test_size=0.2, random_state=args.random_state,
    stratify=_stratify)
print(f"\nHoldout evaluation split: {len(y_hold_eval)} train / {len(y_hold_test)} test")

# Train on 100% of data for production deployment
X_train = X_test = X
y_train = y_test = Y


# ─── Encode labels for XGB/GB (require integer targets) ─────────────────
classes        = sorted(np.unique(Y))
label_to_int   = {l: i for i, l in enumerate(classes)}
int_to_label   = {i: l for l, i in label_to_int.items()}
n_classes      = len(classes)


# ─── Select max_depth and learning_rate defaults per model ───────────────
max_depth = args.max_depth
if max_depth is None and model_type in ('xgb', 'gb'):
    max_depth = 3  # shallow trees for boosting

learning_rate = args.learning_rate
if learning_rate is None:
    learning_rate = 0.3 if model_type == 'xgb' else 0.1


# ─── Helpers ─────────────────────────────────────────────────────────────
def train_non_cnn(mt, X_train_in, y_train_in, X_test_in, y_test_in):
    """Train DT/RF/XGB/GB/KNN/SVM models and return (model, y_pred, labels, n_classes_used, max_depth_used)."""
    local_classes = sorted(np.unique(y_train_in))
    local_label_to_int = {l: i for i, l in enumerate(local_classes)}
    local_int_to_label = {i: l for l, i in local_label_to_int.items()}
    local_n_classes = len(local_classes)

    local_max_depth = args.max_depth
    if local_max_depth is None and mt in ('xgb', 'gb'):
        local_max_depth = 3

    local_lr = args.learning_rate
    if local_lr is None:
        local_lr = 0.3 if mt == 'xgb' else 0.1

    # Resolve class_weight for DT/RF (None means no weighting)
    tree_class_weight = None if args.class_weight == 'none' else 'balanced'

    if mt == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        model_local = DecisionTreeClassifier(
            max_depth=local_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight=tree_class_weight,
        )
        model_local.fit(X_train_in, y_train_in)
        y_pred_local = model_local.predict(X_test_in)
        return model_local, y_pred_local, local_classes, local_n_classes, local_max_depth

    if mt == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model_local = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=local_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight=tree_class_weight,
            n_jobs=-1,
        )
        model_local.fit(X_train_in, y_train_in)
        y_pred_local = model_local.predict(X_test_in)
        return model_local, y_pred_local, local_classes, local_n_classes, local_max_depth

    if mt == 'xgb':
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("ERROR: xgboost required. Install with: pip install xgboost")
            sys.exit(1)

        objective = 'binary:logistic' if local_n_classes == 2 else 'multi:softmax'
        y_train_enc = np.array([local_label_to_int[y] for y in y_train_in])

        model_local = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=local_max_depth,
            learning_rate=local_lr,
            subsample=args.subsample,
            min_child_weight=args.min_child_weight,
            objective=objective,
            num_class=(local_n_classes if local_n_classes > 2 else None),
            random_state=args.random_state,
            use_label_encoder=False,
            eval_metric='logloss' if local_n_classes == 2 else 'mlogloss',
            verbosity=0,
        )
        model_local.fit(X_train_in, y_train_enc)

        y_pred_enc = model_local.predict(X_test_in)
        y_pred_local = np.array([local_int_to_label[int(p)] for p in y_pred_enc])
        return model_local, y_pred_local, local_classes, local_n_classes, local_max_depth

    if mt == 'gb':
        from sklearn.ensemble import GradientBoostingClassifier
        y_train_enc = np.array([local_label_to_int[y] for y in y_train_in])

        loss = args.loss
        if loss == 'exponential' and local_n_classes > 2:
            print("WARNING: exponential loss only supports binary. Using log_loss.")
            loss = 'log_loss'

        model_local = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=local_max_depth,
            learning_rate=local_lr,
            subsample=args.subsample,
            min_samples_leaf=args.min_samples_leaf,
            loss=loss,
            random_state=args.random_state,
        )
        model_local.fit(X_train_in, y_train_enc)

        y_pred_enc = model_local.predict(X_test_in)
        y_pred_local = np.array([local_int_to_label[int(p)] for p in y_pred_enc])
        return model_local, y_pred_local, local_classes, local_n_classes, local_max_depth

    if mt == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model_local = KNeighborsClassifier(
            n_neighbors=args.knn_k,
            weights=args.knn_weights,
            metric=args.knn_metric,
            p=args.knn_p,
            n_jobs=-1,
        )
        model_local.fit(X_train_in, y_train_in)
        y_pred_local = model_local.predict(X_test_in)
        return model_local, y_pred_local, local_classes, local_n_classes, local_max_depth

    if mt == 'svm':
        from sklearn.svm import SVC
        class_weight = None if args.svm_class_weight == 'none' else 'balanced'
        model_local = SVC(
            kernel=args.svm_kernel,
            C=args.svm_c,
            gamma=args.svm_gamma,
            degree=args.svm_degree,
            class_weight=class_weight,
        )
        model_local.fit(X_train_in, y_train_in)
        y_pred_local = model_local.predict(X_test_in)
        return model_local, y_pred_local, local_classes, local_n_classes, local_max_depth

    raise ValueError(f"Unsupported model type: {mt}")


def _feature_bit_width(feature_name, tables_dir):
    if feature_name in P4_FEATURE_MAX:
        max_val = P4_FEATURE_MAX[feature_name]
        return int(round(math.log2(max_val + 1)))
    bits = detect_bits(tables_dir)
    return int(bits or 16)


def quantize_inputs_for_p4(X_in, feature_names, tables_dir, input_bits):
    max_q = (1 << input_bits) - 1
    X_q = np.zeros_like(X_in, dtype=np.int64)
    for idx, fname in enumerate(feature_names):
        width = _feature_bit_width(fname, tables_dir)
        shift = max(0, width - input_bits)
        vals = X_in[:, idx].astype(np.int64)
        q = vals >> shift if shift > 0 else vals
        q = np.clip(q, 0, max_q)
        X_q[:, idx] = q
    return X_q


# ─── Train ───────────────────────────────────────────────────────────────
print(f"\nTraining {model_type.upper()}...")

if model_type == 'cnn':
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch required for CNN. Install with: pip install torch")
        sys.exit(1)

    def parse_channels(value):
        parts = [p.strip() for p in value.split(',') if p.strip()]
        channels = [int(p) for p in parts]
        if not channels:
            raise ValueError("cnn-channels must contain at least one value")
        return channels

    torch.manual_seed(args.random_state)

    channels = parse_channels(args.cnn_channels)
    kernel_size = max(1, int(args.cnn_kernel))

    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    if args.p4_export:
        if args.p4_input_bits is None:
            if all(str(c).endswith('_code') for c in feature_columns):
                auto_bits = int(detect_bits(tables_dir) or 16)
                p4_input_bits = min(10, auto_bits)
            else:
                p4_input_bits = 10
        else:
            p4_input_bits = int(args.p4_input_bits)

        X_train_q = quantize_inputs_for_p4(
            X_train_np, feature_columns, tables_dir, p4_input_bits
        )
        X_test_q = quantize_inputs_for_p4(
            X_test_np, feature_columns, tables_dir, p4_input_bits
        )
        X_train_norm = X_train_q.astype(np.float32)
        X_test_norm = X_test_q.astype(np.float32)
        mean = np.zeros(X_train_np.shape[1], dtype=np.float32)
        std = np.ones(X_train_np.shape[1], dtype=np.float32)
    else:
        mean = X_train_np.mean(axis=0)
        std = X_train_np.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        X_train_norm = (X_train_np - mean) / std
        X_test_norm = (X_test_np - mean) / std

    y_train_enc = np.array([label_to_int[y] for y in y_train], dtype=np.int64)

    if args.p4_export:
        X_train_t = torch.from_numpy(X_train_norm)
        X_test_t = torch.from_numpy(X_test_norm)
    else:
        X_train_t = torch.from_numpy(X_train_norm).unsqueeze(1)
        X_test_t = torch.from_numpy(X_test_norm).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train_enc)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    class SimpleCNN(nn.Module):
        def __init__(self, in_channels, channels, kernel_size, n_classes, dropout):
            super().__init__()
            layers = []
            c_in = in_channels
            for c_out in channels:
                layers.append(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2))
                layers.append(nn.ReLU())
                c_in = c_out
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc = nn.Linear(c_in, n_classes)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            return self.fc(x)

    class P4CNN2(nn.Module):
        def __init__(self, n_features, hidden1, hidden2, pool, n_classes):
            super().__init__()
            self.hidden1 = hidden1
            self.hidden2 = hidden2
            self.pool = pool
            self.fc1 = nn.Linear(n_features, hidden1, bias=True)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(kernel_size=pool, stride=pool)
            self.fc2 = nn.Linear(hidden1 // pool, hidden2, bias=True)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden2, n_classes, bias=True)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = x.unsqueeze(1)  # (N, 1, H1)
            x = self.pool1(x).squeeze(1)
            x = self.fc2(x)
            x = self.relu2(x)
            return self.fc3(x)

    if args.p4_export:
        hidden1 = max(1, int(args.p4_hidden))
        pool = max(1, int(args.p4_pool))
        if pool not in (1, 2):
            raise ValueError("p4_pool currently supports only size=1 or 2 for P4 deployment.")
        if hidden1 % pool != 0:
            raise ValueError(f"p4_hidden ({hidden1}) must be divisible by p4_pool ({pool})")
        hidden2 = int(args.p4_hidden2) if args.p4_hidden2 is not None else max(1, hidden1 // 2)
        model = P4CNN2(
            n_features=X_train_norm.shape[1],
            hidden1=hidden1,
            hidden2=hidden2,
            pool=pool,
            n_classes=n_classes,
        )
    else:
        model = SimpleCNN(
            in_channels=1,
            channels=channels,
            kernel_size=kernel_size,
            n_classes=n_classes,
            dropout=args.dropout,
        )

    if args.cnn_class_weight == 'balanced':
        classes_np, counts_np = np.unique(y_train_enc, return_counts=True)
        inv = (1.0 / counts_np.astype(np.float64))
        weights = inv / inv.sum() * len(classes_np)
        weights_t = torch.tensor(weights, dtype=torch.float32)
    else:
        weights_t = None

    try:
        criterion = nn.CrossEntropyLoss(weight=weights_t, label_smoothing=float(args.cnn_label_smoothing))
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for _ in range(max(1, args.epochs)):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        y_pred_enc = torch.argmax(logits, dim=1).cpu().numpy()
    y_pred = np.array([int_to_label[int(p)] for p in y_pred_enc])
else:
    model, y_pred, labels, n_classes, max_depth = train_non_cnn(
        model_type, X_train, y_train, X_test, y_test
    )


# ─── Holdout evaluation (honest out-of-sample accuracy) ──────────────────
if model_type not in ('cnn',):
    _hold_model, _hold_pred, _hold_classes, _, _ = train_non_cnn(
        model_type, X_hold_eval, y_hold_eval, X_hold_test, y_hold_test)
    _hold_acc = accuracy_score(y_hold_test, _hold_pred)
    _hold_labels = sorted(np.unique(Y))
    _hold_cm  = confusion_matrix(y_hold_test, _hold_pred, labels=_hold_labels)
    _hold_rep = classification_report(y_hold_test, _hold_pred,
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
labels = sorted(np.unique(Y))
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
    "model_type":            model_type,
    "labels":                labels,
    "accuracy":              float(acc),
    "confusion_matrix":      cm.tolist(),
    "classification_report": report,
    "feature_names":         feature_columns,
    "n_estimators":          args.n_estimators if model_type in ('rf', 'xgb', 'gb') else None,
    "max_depth":             max_depth,
    "class_weight":          args.class_weight if model_type in ('dt', 'rf') else None,
}
if model_type not in ('cnn',) and '_hold_acc' in dir():
    metrics["holdout_accuracy"]              = float(_hold_acc)
    metrics["holdout_confusion_matrix"]      = _hold_cm.tolist()
    metrics["holdout_classification_report"] = _hold_rep
if model_type == 'cnn':
    metrics["framework"] = "torch"
    metrics["epochs"] = int(args.epochs)
    metrics["batch_size"] = int(args.batch_size)
    metrics["learning_rate"] = float(args.lr)
    metrics["cnn_channels"] = [int(c) for c in args.cnn_channels.split(',') if c.strip()]
    metrics["cnn_kernel"] = int(args.cnn_kernel)
    metrics["dropout"] = float(args.dropout)
    metrics["cnn_class_weight"] = args.cnn_class_weight
    metrics["cnn_label_smoothing"] = float(args.cnn_label_smoothing)
    metrics["p4_export"] = bool(args.p4_export)
    if args.p4_export:
        metrics["p4_hidden"] = int(args.p4_hidden)
        metrics["p4_hidden2"] = int(args.p4_hidden2) if args.p4_hidden2 is not None else max(1, int(args.p4_hidden) // 2)
        metrics["p4_pool"] = int(args.p4_pool)
        metrics["p4_input_bits"] = int(p4_input_bits)
        metrics["p4_hidden_bits"] = int(args.p4_hidden_bits)
elif model_type == 'knn':
    metrics["knn_k"] = int(args.knn_k)
    metrics["knn_weights"] = args.knn_weights
    metrics["knn_metric"] = args.knn_metric
    metrics["knn_p"] = int(args.knn_p)
elif model_type == 'svm':
    metrics["svm_kernel"] = args.svm_kernel
    metrics["svm_c"] = float(args.svm_c)
    metrics["svm_gamma"] = args.svm_gamma
    metrics["svm_degree"] = int(args.svm_degree)
    metrics["svm_class_weight"] = args.svm_class_weight
metrics_path = os.path.join(tables_dir, f'{model_type}_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics to {metrics_path}")


# ─── Save P4 deployment parameters (ensemble/boosting models only) ───────
if model_type in ('rf',):
    # RF uses the P4 vote architecture (per-tree vote + aggregation)
    vote_bits       = max(1, math.ceil(math.log2(n_classes))) if n_classes > 1 else 1
    total_vote_bits = args.n_estimators * vote_bits
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
    # Write to model-specific params AND rf_params (for step 4/5 compat)
    for pname in [f'{model_type}_params.json', 'rf_params.json']:
        ppath = os.path.join(tables_dir, pname)
        with open(ppath, 'w') as f:
            json.dump(params, f, indent=2)
    print(f"Saved P4 params to {model_type}_params.json + rf_params.json")

elif model_type in ('xgb', 'gb'):
    # XGB and GB share the same P4 architecture (per-tree score accumulation)
    if n_classes > 2:
        total_trees     = args.n_estimators * n_classes
        trees_per_class = args.n_estimators
    else:
        total_trees     = args.n_estimators
        trees_per_class = args.n_estimators

    print(f"\nP4 deployment: total_trees={total_trees}, trees_per_class={trees_per_class}")
    if total_trees > 64:
        print("WARNING: too many tables for hardware P4.")

    params = {
        "model_type":      model_type,
        "n_estimators":    args.n_estimators,
        "n_classes":       n_classes,
        "trees_per_class": trees_per_class,
        "total_trees":     total_trees,
        "classes":         labels,
        "feature_names":   feature_columns,
        "max_depth":       max_depth,
        "delta_bits":      8,
        "accum_bits":      16,
    }
    if model_type == 'xgb':
        params["objective"] = 'binary:logistic' if n_classes == 2 else 'multi:softmax'
    if model_type == 'gb':
        params["loss"] = args.loss

    # Write to model-specific params AND xgb_params (for step 4/5 compat)
    for pname in [f'{model_type}_params.json', 'xgb_params.json']:
        ppath = os.path.join(tables_dir, pname)
        with open(ppath, 'w') as f:
            json.dump(params, f, indent=2)
    print(f"Saved P4 params to {model_type}_params.json + xgb_params.json")


# ─── Save model ──────────────────────────────────────────────────────────
if model_type == 'cnn':
    bundle = {
        "model_type": "cnn",
        "classes": labels,
        "feature_names": feature_columns,
        "n_features_in_": int(X.shape[1]),
        "state_dict": model.state_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "cnn_channels": [int(c) for c in args.cnn_channels.split(',') if c.strip()],
        "cnn_kernel": int(args.cnn_kernel),
        "dropout": float(args.dropout),
    }
    with open(outputfile, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {outputfile}")
else:
    with open(outputfile, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {outputfile}")

if model_type == 'cnn' and args.p4_export:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is None or not hasattr(model, 'fc1') or not hasattr(model, 'fc3'):
        print("WARNING: CNN P4 export skipped (model is not P4CNN2).")
    else:
        w1 = model.fc1.weight.detach().cpu().numpy()
        b1 = model.fc1.bias.detach().cpu().numpy()
        w2 = model.fc2.weight.detach().cpu().numpy()
        b2 = model.fc2.bias.detach().cpu().numpy()
        w3 = model.fc3.weight.detach().cpu().numpy()
        b3 = model.fc3.bias.detach().cpu().numpy()

        w1_scale = float(args.p4_w1_scale)
        w2_scale = float(args.p4_w2_scale)
        w3_scale = float(args.p4_w3_scale)
        w1_int = np.rint(w1 * w1_scale).astype(int)
        b1_int = np.rint(b1 * w1_scale).astype(int)
        w2_int = np.rint(w2 * w2_scale).astype(int)
        b2_int = np.rint(b2 * w2_scale).astype(int)
        w3_int = np.rint(w3 * w3_scale).astype(int)
        b3_int = np.rint(b3 * w3_scale).astype(int)

        input_bits = int(p4_input_bits)
        hidden_bits = int(args.p4_hidden_bits)
        h_max = (1 << hidden_bits) - 1
        input_max = (1 << input_bits) - 1

        abs_w1 = np.abs(w1_int)
        hidden_max = int(np.max(abs_w1.sum(axis=1) * input_max + np.abs(b1_int)))
        h1_shift = 0
        if hidden_max > h_max and hidden_max > 0:
            h1_shift = int(math.ceil(math.log2(hidden_max / float(h_max))))

        abs_w2 = np.abs(w2_int)
        hidden2_max = int(np.max(abs_w2.sum(axis=1) * h_max + np.abs(b2_int)))
        h2_shift = 0
        if hidden2_max > h_max and hidden2_max > 0:
            h2_shift = int(math.ceil(math.log2(hidden2_max / float(h_max))))

        abs_w3 = np.abs(w3_int)
        score_max = int(np.max(abs_w3.sum(axis=1) * h_max + np.abs(b3_int)))
        if score_max > (2**31 - 1):
            print("WARNING: CNN score range may exceed int32. "
                  "Consider lowering --p4-w3-scale or --p4-hidden.")

        # Quantization tables (Tofino-style): learn uniform bins on sums
        levels = int(2 ** hidden_bits)
        Xq = X_train_q.astype(np.int64)
        S1 = Xq @ w1_int.T + b1_int.reshape(1, -1)
        max_pos1 = int(np.max(S1[S1 > 0])) if np.any(S1 > 0) else 0
        step1 = max(1, int(math.ceil(max_pos1 / float(levels - 1)))) if levels > 1 else 1
        Q1 = np.clip((np.maximum(S1, 0) // step1), 0, levels - 1).astype(np.int64)

        if int(args.p4_pool) == 1:
            P1 = Q1
        else:
            # pool=2 only
            P1 = np.maximum(Q1[:, 0::2], Q1[:, 1::2])

        S2 = P1 @ w2_int.T + b2_int.reshape(1, -1)
        max_pos2 = int(np.max(S2[S2 > 0])) if np.any(S2 > 0) else 0
        step2 = max(1, int(math.ceil(max_pos2 / float(levels - 1)))) if levels > 1 else 1

        cnn_params = {
            "model_type": "cnn",
            "classes": labels,
            "feature_names": feature_columns,
            "input_bits": input_bits,
            "hidden_bits": hidden_bits,
            "hidden1_units": int(args.p4_hidden),
            "hidden2_units": int(args.p4_hidden2) if args.p4_hidden2 is not None else max(1, int(args.p4_hidden) // 2),
            "pool": int(args.p4_pool),
            "use_quanti": True,
            "q1_step": step1,
            "q1_max_pos": max_pos1,
            "q2_step": step2,
            "q2_max_pos": max_pos2,
            "h1_shift": h1_shift,
            "h2_shift": h2_shift,
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
            "w3_scale": w3_scale,
            "w1_int": w1_int.tolist(),
            "b1_int": b1_int.tolist(),
            "w2_int": w2_int.tolist(),
            "b2_int": b2_int.tolist(),
            "w3_int": w3_int.tolist(),
            "b3_int": b3_int.tolist(),
        }
        cnn_params_path = os.path.join(tables_dir, "cnn_params.json")
        with open(cnn_params_path, 'w') as f:
            json.dump(cnn_params, f, indent=2)
        print(f"Saved CNN P4 params to {cnn_params_path}")


# ─── Optional CNN distillation ───────────────────────────────────────────
if model_type == 'cnn' and args.distill_to:
    distill_type = args.distill_to
    print(f"\nDistilling CNN → {distill_type.upper()} surrogate for P4 deployment...")

    # Use CNN predictions as pseudo-labels
    y_distill = y_pred  # predictions on X_test (same as X for this pipeline)
    y_distill_train = y_distill
    y_distill_test = y_distill

    surrogate_model, surrogate_pred, surrogate_labels, surrogate_n_classes, surrogate_max_depth = train_non_cnn(
        distill_type, X_train, y_distill_train, X_test, y_distill_test
    )

    # Evaluate surrogate against true labels (original Y)
    acc_s = accuracy_score(y_test, surrogate_pred)
    labels_s = sorted(np.unique(Y))
    cm_s = confusion_matrix(y_test, surrogate_pred, labels=labels_s)
    report_s = classification_report(y_test, surrogate_pred, labels=labels_s, output_dict=True)

    metrics_s = {
        "model_type":            distill_type,
        "distilled_from":        "cnn",
        "labels":                labels_s,
        "accuracy":              float(acc_s),
        "confusion_matrix":      cm_s.tolist(),
        "classification_report": report_s,
        "feature_names":         feature_columns,
        "n_estimators":          args.n_estimators if distill_type not in ('dt',) else None,
        "max_depth":             surrogate_max_depth,
    }
    metrics_path_s = os.path.join(tables_dir, f'{distill_type}_metrics.json')
    with open(metrics_path_s, 'w') as f:
        json.dump(metrics_s, f, indent=2)
    print(f"Saved distill metrics to {metrics_path_s}")

    # Save P4 params for deployable ensembles
    if distill_type in ('rf',):
        vote_bits       = max(1, math.ceil(math.log2(surrogate_n_classes))) if surrogate_n_classes > 1 else 1
        total_vote_bits = args.n_estimators * vote_bits
        vote_table_entries = 2 ** total_vote_bits

        print(f"\nP4 vote packing: n_estimators={args.n_estimators}, "
              f"vote_bits={vote_bits}, total={total_vote_bits}, "
              f"vote_table={vote_table_entries:,} entries")
        if total_vote_bits > 24:
            print("WARNING: vote table may be too large for hardware targets.")

        params = {
            "model_type":      distill_type,
            "n_estimators":    args.n_estimators,
            "n_classes":       surrogate_n_classes,
            "vote_bits":       vote_bits,
            "total_vote_bits": total_vote_bits,
            "classes":         labels_s,
            "feature_names":   feature_columns,
        }
        for pname in [f'{distill_type}_params.json', 'rf_params.json']:
            ppath = os.path.join(tables_dir, pname)
            with open(ppath, 'w') as f:
                json.dump(params, f, indent=2)
        print(f"Saved P4 params to {distill_type}_params.json + rf_params.json")

    elif distill_type in ('xgb', 'gb'):
        if surrogate_n_classes > 2:
            total_trees     = args.n_estimators * surrogate_n_classes
            trees_per_class = args.n_estimators
        else:
            total_trees     = args.n_estimators
            trees_per_class = args.n_estimators

        print(f"\nP4 deployment: total_trees={total_trees}, trees_per_class={trees_per_class}")
        if total_trees > 64:
            print("WARNING: too many tables for hardware P4.")

        params = {
            "model_type":      distill_type,
            "n_estimators":    args.n_estimators,
            "n_classes":       surrogate_n_classes,
            "trees_per_class": trees_per_class,
            "total_trees":     total_trees,
            "classes":         labels_s,
            "feature_names":   feature_columns,
            "max_depth":       surrogate_max_depth,
            "delta_bits":      8,
            "accum_bits":      16,
        }
        if distill_type == 'xgb':
            params["objective"] = 'binary:logistic' if surrogate_n_classes == 2 else 'multi:softmax'
        if distill_type == 'gb':
            params["loss"] = args.loss

        for pname in [f'{distill_type}_params.json', 'xgb_params.json']:
            ppath = os.path.join(tables_dir, pname)
            with open(ppath, 'w') as f:
                json.dump(params, f, indent=2)
        print(f"Saved P4 params to {distill_type}_params.json + xgb_params.json")

    # Save surrogate model to deployable filename
    distill_output = os.path.join(os.path.dirname(outputfile), f"{distill_type}.model")
    with open(distill_output, 'wb') as f:
        pickle.dump(surrogate_model, f)
    print(f"Distilled model saved to {distill_output}")
