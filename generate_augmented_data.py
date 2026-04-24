"""
generate_augmented_data.py
--------------------------
Adds 541 synthetic HEALTHY entries to parkinsons.data
(195 real entries + 541 synthetic = 736 total)

  Status breakdown: 589 healthy (80.0%)  /  147 PD (20.0%, just under)
  Names follow original format: phon_R01_S{nn}_{rec}, subjects starting S51
"""
import numpy as np
import pandas as pd
import re

REAL_DATA    = "parkinsons.data"
OUT_DATA     = "parkinsons_augmented.data"
N_SYNTHETIC  = 541       # 532 requested + 9 to guarantee < 20% PD
RECS_PER_SUB = 6
RANDOM_SEED  = 42

np.random.seed(RANDOM_SEED)

df   = pd.read_csv(REAL_DATA)
feat = [c for c in df.columns if c not in ("name", "status")]
feat_min = df[feat].min()
feat_max = df[feat].max()

healthy_real = df[df["status"] == 0]
mu_h  = healthy_real[feat].mean().values
cov_h = np.cov(healthy_real[feat].values.T) + np.eye(len(mu_h)) * 1e-8

# Find first free subject ID (max real subject + 1)
existing_ids = df["name"].str.extract(r"S(\d+)")[0].dropna().astype(int)
next_subj = int(existing_ids.max()) + 1   # typically S51

def make_names(n, start_subj, recs=RECS_PER_SUB):
    names = []
    subj  = start_subj
    while len(names) < n:
        for rec in range(1, recs + 1):
            names.append(f"phon_R01_S{subj:02d}_{rec}")
            if len(names) >= n:
                break
        subj += 1
    return names[:n]

# Generate synthetic healthy samples
samples = np.random.multivariate_normal(mu_h, cov_h, size=N_SYNTHETIC)
for i, col in enumerate(feat):
    samples[:, i] = np.clip(samples[:, i], feat_min[col], feat_max[col])

names_syn = make_names(N_SYNTHETIC, next_subj)
syn_rows  = [{"name": n, "status": 0, **dict(zip(feat, row))}
             for n, row in zip(names_syn, samples)]

combined = pd.concat([df, pd.DataFrame(syn_rows)], ignore_index=True)
combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
combined = combined[list(df.columns)]
combined.to_csv(OUT_DATA, index=False)

n_h = (combined["status"] == 0).sum()
n_p = (combined["status"] == 1).sum()
pct = n_p / len(combined) * 100
print(f"Output   : {OUT_DATA}  ({len(combined)} total rows)")
print(f"Healthy  : {n_h}  ({100-pct:.1f}%)")
print(f"PD       : {n_p}  ({pct:.1f}%)")
print(f"NaN      : {combined[feat].isnull().sum().sum()}")
print(f"Name ex  : {names_syn[0]} ... {names_syn[-1]}")
