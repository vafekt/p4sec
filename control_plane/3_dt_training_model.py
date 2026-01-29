#!/usr/bin/env python3

"""
Train a DecisionTreeClassifier using PCA code features and save model + metrics.

Input CSV (from tables/pca_integer_mapping.csv) contains:
 - Feature columns ending with `_code` (e.g., PC1_code ... PCk_code)
 - Target column `Label` (string class names)

This script:
 - Loads the CSV from `tables/` by default
 - Selects only `_code` columns for X and `Label` for y
 - Cleans NaN/Inf rows
 - Splits data into train/test
 - Trains DecisionTreeClassifier
 - Reports accuracy, precision, recall, F1, confusion matrix
 - Saves model to `model/dt.model` and metrics to `tables/dt_metrics.json`
"""

import os
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle


parser = argparse.ArgumentParser(description="Train DecisionTreeClassifier on PCA code features")

# Paths
parser.add_argument('-i', default="tables/pca_integer_mapping.csv", help='path to input mapping CSV')
parser.add_argument('-o', default="model/dt.model", help='path to output model file')

parser.add_argument('--test-size', type=float, default=0.3, help='test split size (default: 0.3)')
parser.add_argument('--random-state', type=int, default=42, help='random seed (default: 42)')
parser.add_argument('--max-depth', type=int, default=None, help='DecisionTree max_depth (default: None)')
parser.add_argument('--min-samples-leaf', type=int, default=1, help='DecisionTree min_samples_leaf (default: 1)')

args = parser.parse_args()

# extract argument
inputfile  = args.i
outputfile = args.o
random_state = args.random_state
max_depth = args.max_depth
min_samples_leaf = args.min_samples_leaf

# Ensure output directories exist
out_dir = os.path.dirname(outputfile)
if out_dir:
	os.makedirs(out_dir, exist_ok=True)
metrics_path = os.path.join(os.path.dirname(__file__), 'tables', 'dt_metrics.json')
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

# Load dataset
df = pd.read_csv(inputfile)

# Select features and target
code_columns = [col for col in df.columns if col.endswith('_code')]
if not code_columns:
	raise ValueError("No feature columns ending with '_code' found in input CSV")
if 'Label' not in df.columns:
	raise ValueError("Target column 'Label' not found in input CSV")

X = df[code_columns].values
Y = df['Label'].values

# Clean NaN/Inf
X = np.asarray(X, dtype=np.float64)
Y = np.asarray(Y)
mask = np.isfinite(X).all(axis=1)
X = X[mask]
Y = Y[mask]


# Train and test on 100% of the data
X_train = X
X_test = X
y_train = Y
y_test = Y

# Train classifier
dt = tree.DecisionTreeClassifier(
	max_depth=max_depth,
	min_samples_leaf=min_samples_leaf,
	random_state=random_state,
)
dt.fit(X_train, y_train)


# Predict and metrics
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
labels = sorted(np.unique(Y))
cm = confusion_matrix(y_test, y_pred, labels=labels)
report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

# Print misclassified samples for inspection
misclassified = (y_pred != y_test)
if np.any(misclassified):
	print("\nMisclassified samples (up to 20 shown):")
	shown = 0
	for i in np.where(misclassified)[0]:
		print(f"Sample {i}: features={X_test[i]}, true={y_test[i]}, pred={y_pred[i]}")
		shown += 1
		if shown >= 20:
			break
	print(f"Total misclassified: {misclassified.sum()} / {len(y_test)}")
else:
	print("\nNo misclassified samples: perfect accuracy.")

print("Accuracy:", acc)
print("Labels:", labels)
print("Confusion Matrix:\n", cm)

# Save metrics
metrics = {
    "labels": labels,
    "accuracy": float(acc),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_names": code_columns,
    "random_state": int(random_state),
    "max_depth": (None if max_depth is None else int(max_depth)),
    "min_samples_leaf": int(min_samples_leaf),
    "input_csv": inputfile,
    "model_path": outputfile,
}

with open(metrics_path, 'w') as f:
	json.dump(metrics, f, indent=2)
print("Saved metrics to", metrics_path)

# Save model
with open(outputfile, 'wb') as f:
	pickle.dump(dt, f)
print("Model written to", outputfile)