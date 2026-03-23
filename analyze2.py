import pandas as pd
import numpy as np

pred = pd.read_csv("control_plane/logs/predictions.csv")
ds = pd.read_csv("control_plane/dataset/dataset.csv")

print("=== PREDICTIONS (839 flows) ===")
print(pred['class_label'].value_counts().to_string())

# Find boundary row 522
print(f"\n=== Row around 522 (Malware->Web transition) ===")
print(pred.iloc[519:525][['src_ip','src_port','class_label']].to_string())

# Zero PCA check
pc_cols = [c for c in pred.columns if c.startswith('pc') and c.endswith('_code')]
zero_mask = (pred[pc_cols] == 0).all(axis=1)
print(f"\n=== Zero PCA codes: {zero_mask.sum()} flows ===")
if zero_mask.sum() > 0:
    print(pred[zero_mask][['duration','max_iat','fwd_pkt_count','bwd_pkt_count','class_label']].to_string())

# Cross-check: Malware rows vs Malware in dataset
print("\n=== Dataset label distribution ===")
print(ds['Label'].value_counts().to_string())

# Check if predicted Malware flows exist in Malware training data
pred_malware = pred[pred['class_label'] == 'Malware']
pred_web     = pred[pred['class_label'] == 'Web']

# Map pred column names to dataset column names
col_map = {
    'fwd_bytes': 'FwdBytes', 'bwd_bytes': 'BwdBytes',
    'fwd_pkt_count': 'FwdPktCount', 'bwd_pkt_count': 'BwdPktCount',
    'duration': 'Duration', 'max_iat': 'MaxIAT', 'min_iat': 'MinIAT',
}

# Check: are any predicted-Web rows actually Malware in dataset?
ds_malware = ds[ds['Label'].str.contains('Backdoor|Malware', case=False, na=False)]
ds_web     = ds[ds['Label'] == 'Web']

wrong_web = 0
for _, row in pred_web.iterrows():
    match = ds_malware[
        (ds_malware['FwdBytes'] == row['fwd_bytes']) &
        (ds_malware['BwdBytes'] == row['bwd_bytes']) &
        (ds_malware['FwdPktCount'] == row['fwd_pkt_count'])
    ]
    if len(match) > 0:
        wrong_web += 1

wrong_malware = 0
for _, row in pred_malware.head(100).iterrows():
    match = ds_web[
        (ds_web['FwdBytes'] == row['fwd_bytes']) &
        (ds_web['BwdBytes'] == row['bwd_bytes']) &
        (ds_web['FwdPktCount'] == row['fwd_pkt_count'])
    ]
    if len(match) > 0:
        wrong_malware += 1

print(f"\n=== Cross-label check ===")
print(f"Predicted Web but matched Malware in dataset: {wrong_web} / {len(pred_web)}")
print(f"Predicted Malware but matched Web in dataset (sample 100): {wrong_malware} / 100")

print(f"\n=== Feature ranges: Malware predictions ===")
for c in ['duration','max_iat','min_iat','fwd_bytes','bwd_bytes']:
    print(f"  {c}: {pred_malware[c].min()} -> {pred_malware[c].max()}")

print(f"\n=== Feature ranges: Web predictions ===")
for c in ['duration','max_iat','min_iat','fwd_bytes','bwd_bytes']:
    print(f"  {c}: {pred_web[c].min()} -> {pred_web[c].max()}")
