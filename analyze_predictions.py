import pandas as pd
import numpy as np

pred = pd.read_csv("control_plane/logs/predictions.csv")
ds = pd.read_csv("control_plane/dataset/dataset.csv")

print("=== PREDICTIONS SUMMARY ===")
print("Total flows:", len(pred))
print(pred['class_label'].value_counts().to_string())

pc_cols = ['pc1_code','pc2_code','pc3_code','pc4_code','pc5_code']
zero_mask = (pred[pc_cols] == 0).all(axis=1)
zero_pca = pred[zero_mask]
nonzero_pca = pred[~zero_mask]

print(f"\n=== ALL-ZERO PCA CODES (P4 table miss) ===")
print(f"Count: {len(zero_pca)} / {len(pred)} ({100*len(zero_pca)/len(pred):.1f}%)")
print("Class distribution:")
print(zero_pca['class_label'].value_counts().to_string())

print(f"\n=== NON-ZERO PCA CODES ===")
print(f"Count: {len(nonzero_pca)}")
print("Class distribution:")
print(nonzero_pca['class_label'].value_counts().to_string())

print("\n=== FEATURE COMPARISON: predictions vs dataset ===")
feat_map = {
    'duration': 'Duration', 'max_iat': 'MaxIAT', 'min_iat': 'MinIAT',
    'fwd_bytes': 'FwdBytes', 'bwd_bytes': 'BwdBytes',
    'fwd_pkt_count': 'FwdPktCount', 'bwd_pkt_count': 'BwdPktCount',
    'fwd_max_pkt_len': 'FwdMaxPktLen', 'bwd_max_pkt_len': 'BwdMaxPktLen',
}
print(f"{'Feature':<20} {'pred_max':>15} {'dataset_max':>15} {'exceeds?':>10}")
print("-" * 65)
for pcol, dcol in feat_map.items():
    if pcol in pred.columns and dcol in ds.columns:
        pm = pred[pcol].max()
        dm = ds[dcol].max()
        flag = " <-- EXCEEDS!" if pm > dm else ""
        print(f"{pcol:<20} {pm:>15,} {dm:>15,}{flag}")

print("\n=== ZERO PCA flows: feature ranges ===")
for pcol in ['duration','max_iat','min_iat','fwd_bytes','bwd_bytes','fwd_pkt_count','bwd_pkt_count']:
    if pcol in pred.columns:
        print(f"  {pcol}: {zero_pca[pcol].min()} -> {zero_pca[pcol].max()}")

print("\n=== NONZERO PCA flows: feature ranges ===")
for pcol in ['duration','max_iat','min_iat','fwd_bytes','bwd_bytes','fwd_pkt_count','bwd_pkt_count']:
    if pcol in pred.columns:
        print(f"  {pcol}: {nonzero_pca[pcol].min()} -> {nonzero_pca[pcol].max()}")

# Check if zero-pca flows exist in dataset (exact feature match)
print("\n=== CHECKING: do zero-PCA flows appear in dataset? ===")
ds_web = ds[ds['Label'] == 'Web']
web_features = ['FwdBytes','BwdBytes','FwdPktCount','BwdPktCount','FlagsAck']
if len(zero_pca) > 0:
    sample = zero_pca.head(3)
    for _, row in sample.iterrows():
        match = ds_web[
            (ds_web['FwdBytes'] == row.get('fwd_bytes', -1)) &
            (ds_web['BwdBytes'] == row.get('bwd_bytes', -1))
        ]
        print(f"  flow fwd_bytes={row.get('fwd_bytes')}, bwd_bytes={row.get('bwd_bytes')}, "
              f"class={row['class_label']}: found {len(match)} matches in Web dataset")
