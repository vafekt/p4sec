#!/usr/bin/env python3
"""
Feature Selection pipeline — direct classification on selected raw features.

NO transform encoder.  NO surrogate tree.  NO quantization.

This script selects the top-k most discriminative raw features and passes
them DIRECTLY to the classifier (steps 3/4).  The classifier's range-match
tables will use the raw P4 feature values as match keys.

Outputs:
  tables/transform_mapping.csv     Columns: <selected_feature_names>, Label
  tables/reduction_config.json     Universal config for steps 3/4/5
  tables/transform_metrics.json    Accuracy comparison
  tables/s1-commands.txt           EMPTY (no transform tables needed)

Available methods (--method):
  mi     Mutual Information (non-parametric, handles non-linear)
  chi2   Chi-squared test (fast, assumes non-negative features)
  anova  ANOVA F-value (linear separability test)

P4 deployment
-------------
With feature selection the classifier tables (ml_code / rf_tree_* / xgb_tree_*)
match DIRECTLY on raw flow features instead of transform codes.  This means:
  - No pca_component* tables in the P4 data plane
  - Classifier match keys are the selected subset of raw features
  - Each match key has its own P4 bit width (not uniform BITS-wide codes)
"""

import numpy as np
import pandas as pd
import json
import os
import glob
import argparse
import sys

from sklearn.feature_selection import (
    mutual_info_classif, chi2, f_classif,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from pipeline_utils import P4_FEATURE_MAX

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

# ==========================================================
# 0. CONFIG
# ==========================================================
parser = P4secArgumentParser(
    description="Feature Selection pipeline (direct, works with any classifier in step 3)",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
            "Outputs:\n"
            "  tables/reduction_config.json     (universal config)\n"
            "  tables/transform_mapping.csv     (selected features + labels)\n"
            "\n"
            "Note: Feature Selection produces NO transform tables.\n"
            "Classifier operates directly on raw features.\n"
        )
    )
parser.add_argument("--components", "-k", type=int, default=None,
                    help="Number of features to select (default: auto)")
parser.add_argument("--method", "-m", choices=["mi", "chi2", "anova"],
                    default="mi", help="Selection method (default: mi)")
args = parser.parse_args()

USER_K = args.components
METHOD = args.method

# ==========================================================
# 1. Load dataset
# ==========================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def find_dataset_csv():
    for d in [os.path.join(os.path.dirname(__file__), "dataset"),
              os.path.join(ROOT_DIR, "dataset")]:
        if not os.path.isdir(d):
            continue
        csvs = glob.glob(os.path.join(d, "*.csv"))
        if not csvs:
            continue
        preferred = [p for p in csvs if os.path.basename(p).lower() == "dataset.csv"]
        return preferred[0] if preferred else sorted(csvs, key=os.path.getmtime, reverse=True)[0]
    raise FileNotFoundError("No CSV dataset found in any dataset/ directory")

csv_path = find_dataset_csv()
print("Using dataset:", csv_path)
df = pd.read_csv(csv_path)
label_col = df.columns[-1]
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

P4_FEATURE_COLS = [
    "Protocol", "SrcPort", "DstPort",
    "Duration", "MaxIAT", "UrgCount",
    "FwdPktCount", "BwdPktCount", "FwdBytes", "BwdBytes",
    "MaxWinSize", "FlagsSyn", "FlagsAck", "FlagsFin", "FlagsRst",
    "MinIAT", "FwdMaxPktLen", "BwdMaxPktLen", "FlagsPsh", "InitFwdWinBytes",
]
X_df = df_clean[P4_FEATURE_COLS].astype(int)
feature_cols = X_df.columns.tolist()
X = X_df.values
y = df_clean[label_col].values

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print("Labels:", np.unique(y)[:10])

# ==========================================================
# 2. Score features
# ==========================================================
le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"\nScoring features with: {METHOD}")
if METHOD == "mi":
    scores = mutual_info_classif(X, y_enc, discrete_features=True, random_state=42)
    score_name = "Mutual Information"
elif METHOD == "chi2":
    scores, _ = chi2(X, y_enc)
    score_name = "Chi-squared"
elif METHOD == "anova":
    scores, _ = f_classif(X, y_enc)
    score_name = "ANOVA F-value"

ranked_idx = np.argsort(scores)[::-1]
print(f"\n=== Feature Ranking ({score_name}) ===")
for rank, idx in enumerate(ranked_idx, 1):
    print(f"  {rank}. {feature_cols[idx]:15s}  score={scores[idx]:.4f}")

# ==========================================================
# 3. Select top-k
# ==========================================================
if USER_K is not None:
    k = min(USER_K, len(feature_cols))
else:
    median_score = np.median(scores)
    k = max(2, int(np.sum(scores > median_score)))
    k = min(k, len(feature_cols) - 1)
    print(f"\nAuto-selected {k} features (score > median={median_score:.4f})")

selected_idx = ranked_idx[:k]
selected_features = [feature_cols[i] for i in selected_idx]
print(f"\nSelected {k} features: {selected_features}")

X_selected = X[:, selected_idx]

# ==========================================================
# 4. Accuracy comparison
# ==========================================================
clf_all = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_all.fit(X, y)
acc_all = accuracy_score(y, clf_all.predict(X))

clf_sel = DecisionTreeClassifier(max_depth=100, random_state=42)
clf_sel.fit(X_selected, y)
acc_sel = accuracy_score(y, clf_sel.predict(X_selected))

print(f"\n=== Classification Accuracy ===")
print(f"All {len(feature_cols)} features : {acc_all}")
print(f"Selected {k} features  : {acc_sel}")

labels = sorted(np.unique(y))
metrics = {
    "method": f"FeatureSelection_{METHOD}",
    "n_components": k,
    "selected_features": selected_features,
    "feature_scores": {feature_cols[i]: float(scores[i]) for i in range(len(feature_cols))},
    "labels": labels,
    "all_features":       {"accuracy": float(acc_all)},
    "selected_features_acc": {"accuracy": float(acc_sel)},
}
with open(os.path.join(TABLES_DIR, "transform_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ==========================================================
# 5. Save mapping CSV — RAW feature values, NOT codes
# ==========================================================
# The CSV has the selected feature columns (by their real names) + Label.
# Steps 3/4 will read feature_columns from reduction_config.json.
mapping = {}
for j, feat_name in enumerate(selected_features):
    mapping[feat_name] = X_selected[:, j]
mapping["Label"] = y

# Duplicate rows (train=test=full) to match PCA pipeline convention
# where train and test are stacked.  This is wasteful but keeps the
# interface consistent for step 3 scripts that expect it.
y_all = np.concatenate([y, y])
mapping_all = {}
for j, feat_name in enumerate(selected_features):
    vals = X_selected[:, j]
    mapping_all[feat_name] = np.concatenate([vals, vals])
mapping_all["Label"] = y_all

pd.DataFrame(mapping_all).to_csv(
    os.path.join(TABLES_DIR, "transform_mapping.csv"), index=False)
print("Saved transform_mapping.csv (raw feature columns)")

# ==========================================================
# 6. Save reduction_config.json — THE universal contract
# ==========================================================
feature_max_values = {}
for feat_name in selected_features:
    feature_max_values[feat_name] = int(P4_FEATURE_MAX.get(feat_name, 2**32 - 1))

reduction_config = {
    "method": "feature_selection",
    "selection_method": METHOD,
    "feature_columns": selected_features,
    "feature_max_values": feature_max_values,
    "needs_transform_tables": False,
    "n_components": int(k),
    "bits": None,    # not applicable — raw features have native P4 widths
    "all_feature_scores": {feature_cols[i]: float(scores[i]) for i in range(len(feature_cols))},
}
with open(os.path.join(TABLES_DIR, "reduction_config.json"), "w") as f:
    json.dump(reduction_config, f, indent=2)
print("Saved reduction_config.json")

# ==========================================================
# 7. Save encoding_params.json (minimal stub for compat)
# ==========================================================
# Some scripts may try to read this.  Provide minimal stub.
encoding_params = {
    "method": f"FeatureSelection_{METHOD}",
    "n_components": int(k),
    "bits": None,
    "max_val": None,
    "selected_features": selected_features,
}
with open(os.path.join(TABLES_DIR, "encoding_params.json"), "w") as f:
    json.dump(encoding_params, f, indent=2)

# ==========================================================
# 8. Write empty s1-commands.txt (no transform tables)
# ==========================================================
with open(os.path.join(TABLES_DIR, "s1-commands.txt"), "w") as f:
    f.write("# Feature Selection — no transform tables needed.\n")
    f.write("# Classifier entries will be appended by step 4.\n")
print("Saved empty s1-commands.txt")

# ==========================================================
# 9. Write human-readable feature selection summary
# ==========================================================
with open(os.path.join(TABLES_DIR, "transform_rules.txt"), "w") as f:
    f.write(f"# Feature Selection ({METHOD})\n")
    f.write(f"# No transformation rules — classifier operates on raw features.\n")
    f.write(f"# Selected features ({k} of {len(feature_cols)}):\n")
    for j, feat_name in enumerate(selected_features):
        f.write(f"#   {j+1}. {feat_name}  "
                f"(score={scores[selected_idx[j]]:.4f}, "
                f"P4 max={feature_max_values[feat_name]})\n")

print("\n" + "=" * 60)
print("Feature selection complete.  Run any step 3/4 classifier next:")
print("  python3 3_train_model.py --model-type dt")
print("  python3 3_train_model.py --model-type rf")
print("  python3 3_train_model.py --model-type xgb")
print("  python3 3_train_model.py --model-type gb")
print("  python3 3_train_model.py --model-type knn")
print("  python3 3_train_model.py --model-type svm")
print("  python3 3_train_model.py --model-type cnn")
print("=" * 60)
