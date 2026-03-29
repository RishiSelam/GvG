# =============================================================================
# GAN-vs-GAN IDS PROJECT — Data Preprocessing
# Dataset: CICIDS 2017 (MachineLearningCSV.zip)
# Author: [Your Name] — Data Preprocessing Module
# Team role: Provides clean train/val/test splits + TimeGAN sequences
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
!pip install imbalanced-learn -q

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Step 1: Unzip and load all CSVs
# ─────────────────────────────────────────────────────────────────────────────

ZIP_PATH = '/content/MachineLearningCSV.zip'
EXTRACT_DIR = '/content/CICIDS2017'

# Unzip
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_DIR)
    print("Extracted files:")
    print('\n'.join(z.namelist()))

# Find all CSVs (they may be in a subfolder)
csv_files = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.endswith('.csv'):
            csv_files.append(os.path.join(root, f))

print(f"\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  {f}")

# Load and concatenate all CSVs
dfs = []
for f in csv_files:
    print(f"\nLoading {os.path.basename(f)}...")
    tmp = pd.read_csv(f, encoding='utf-8', low_memory=False)
    print(f"  Shape: {tmp.shape}")
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
print(f"\n✓ Merged dataset shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Step 2: Fix column names
# CICIDS 2017 columns have leading/trailing spaces — this breaks everything
# ─────────────────────────────────────────────────────────────────────────────

print("Columns BEFORE stripping:")
print(df.columns.tolist()[:5], "...")

# Strip whitespace from all column names
df.columns = df.columns.str.strip()

# The label column is called ' Label' or 'Label' depending on the file
# After stripping it should be 'Label' — verify:
assert 'Label' in df.columns, f"Label column not found! Columns: {df.columns.tolist()}"
print("\n✓ Column names cleaned. Label column found.")
print(f"Unique labels: {df['Label'].unique()}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Step 3: Clean the data
# CICIDS 2017 has known issues: Inf values, NaNs, duplicate rows
# ─────────────────────────────────────────────────────────────────────────────

print(f"Shape before cleaning: {df.shape}")

# 3a. Replace Inf and -Inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3b. Drop rows with any NaN
nan_count = df.isnull().sum().sum()
print(f"NaN values found: {nan_count}")
df.dropna(inplace=True)
print(f"Shape after dropping NaN: {df.shape}")

# 3c. Drop fully duplicate rows
dup_count = df.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")
df.drop_duplicates(inplace=True)
print(f"Shape after dropping duplicates: {df.shape}")

# 3d. Separate features and label
X = df.drop(columns=['Label'])
y = df['Label']

# 3e. Drop any non-numeric columns (e.g. if a flow ID column slipped in)
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric}")
    X.drop(columns=non_numeric, inplace=True)

print(f"\n✓ Clean feature matrix: {X.shape}")
print(f"✓ Label series: {y.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Step 4: Encode labels
# Map multi-class labels → integers AND binary (BENIGN=0, ATTACK=1)
# Your team needs BOTH: multi-class for full IDS, binary for GAN training
# ─────────────────────────────────────────────────────────────────────────────

# Multi-class encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Label mapping (multi-class):")
for i, cls in enumerate(le.classes_):
    count = (y_encoded == i).sum()
    print(f"  {i:2d} → {cls:<40s} count: {count:>8,}")

# Binary encoding: BENIGN = 0, any attack = 1
y_binary = (y != 'BENIGN').astype(int)
print(f"\nBinary split:")
print(f"  BENIGN (0): {(y_binary == 0).sum():,}")
print(f"  ATTACK (1): {(y_binary == 1).sum():,}")

# Save the encoder — teammates need this to decode predictions
with open('/content/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("\n✓ label_encoder.pkl saved")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Step 5: Class distribution plot (EDA)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Multi-class bar chart
class_counts = pd.Series(y_encoded).value_counts().sort_index()
class_names = [le.classes_[i] for i in class_counts.index]
axes[0].barh(class_names, class_counts.values, color='steelblue')
axes[0].set_xlabel('Sample count')
axes[0].set_title('Class distribution (multi-class)')
for i, v in enumerate(class_counts.values):
    axes[0].text(v + 1000, i, f'{v:,}', va='center', fontsize=8)

# Binary pie chart
binary_counts = y_binary.value_counts()
axes[1].pie(binary_counts, labels=['BENIGN', 'ATTACK'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
axes[1].set_title('Binary distribution')

plt.tight_layout()
plt.savefig('/content/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ class_distribution.png saved")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Step 6: Feature engineering
# Remove zero-variance and near-zero-variance features
# They carry no information and can break GAN training
# ─────────────────────────────────────────────────────────────────────────────

print(f"Features before variance filtering: {X.shape[1]}")

# Remove constant columns (zero variance)
vt = VarianceThreshold(threshold=0.0)
X_vt = vt.fit_transform(X)
removed_const = X.shape[1] - X_vt.shape[1]
print(f"Removed {removed_const} constant features")

# Get surviving feature names
feature_names = X.columns[vt.get_support()].tolist()
X = pd.DataFrame(X_vt, columns=feature_names)
print(f"Features after variance filtering: {X.shape[1]}")

# Optional: correlation-based removal (remove one of any pair with |corr| > 0.98)
# Uncomment if you want to reduce features further for faster GAN training:
# corr_matrix = X.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
# print(f"Removing {len(to_drop)} highly correlated features")
# X.drop(columns=to_drop, inplace=True)
# feature_names = X.columns.tolist()

print(f"\n✓ Final feature count: {X.shape[1]}")
print(f"Feature names saved for reference.")

# Save feature names so teammates know what's in X
with open('/content/feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Step 7: Train / Validation / Test split
# Do this BEFORE scaling — scaler must only be fit on training data
# Stratify on multi-class label to preserve class ratios in each split
# ─────────────────────────────────────────────────────────────────────────────

X_arr = X.values
y_mc = y_encoded          # multi-class
y_bin = y_binary.values   # binary

# First split off test set (20%)
X_temp, X_test, y_mc_temp, y_mc_test, y_bin_temp, y_bin_test = train_test_split(
    X_arr, y_mc, y_bin,
    test_size=0.20,
    random_state=42,
    stratify=y_mc
)

# Split remaining 80% into train (75%) and val (25%) → 60/20/20 overall
X_train, X_val, y_mc_train, y_mc_val, y_bin_train, y_bin_val = train_test_split(
    X_temp, y_mc_temp, y_bin_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_mc_temp
)

print(f"Train:      {X_train.shape[0]:>8,} samples")
print(f"Validation: {X_val.shape[0]:>8,} samples")
print(f"Test:       {X_test.shape[0]:>8,} samples")
print(f"\nTrain class distribution: {Counter(y_mc_train)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Step 8: Scale features
# Fit StandardScaler on TRAINING SET ONLY, then transform all splits
# This prevents data leakage from val/test into the scaler
# ─────────────────────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only

# Also create a [0,1] scaled version for TimeGAN (it requires [0,1] input)
minmax = MinMaxScaler(feature_range=(0, 1))
X_train_mm = minmax.fit_transform(X_train)
X_val_mm   = minmax.transform(X_val)
X_test_mm  = minmax.transform(X_test)

# Save scalers — teammates MUST use these to inverse-transform generated samples
with open('/content/scaler_standard.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('/content/scaler_minmax.pkl', 'wb') as f:
    pickle.dump(minmax, f)

print("✓ StandardScaler and MinMaxScaler fitted and saved")
print(f"  Train mean (first 3 features): {X_train_scaled.mean(axis=0)[:3].round(4)}")
print(f"  Train std  (first 3 features): {X_train_scaled.std(axis=0)[:3].round(4)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Step 9: Handle class imbalance (SMOTE on training set ONLY)
# CICIDS 2017 is severely imbalanced — BENIGN vastly outnumbers attacks
# Apply SMOTE only to training data, never touch val/test
# ─────────────────────────────────────────────────────────────────────────────

print("Class distribution BEFORE balancing:")
print(Counter(y_bin_train))

# Strategy: oversample minority (attack) to 50% of majority (benign)
# then undersample majority to avoid explosion in dataset size
over  = SMOTE(sampling_strategy=0.5, random_state=42)
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
pipeline = ImbPipeline([('over', over), ('under', under)])

X_train_bal, y_bin_train_bal = pipeline.fit_resample(X_train_scaled, y_bin_train)

print("\nClass distribution AFTER balancing (binary):")
print(Counter(y_bin_train_bal))
print(f"New training set size: {X_train_bal.shape[0]:,}")

# NOTE: For multi-class use, keep unbalanced X_train_scaled with class weights
# in model training instead. SMOTE on 15 classes can distort minority classes.
# Provide both to the team:
print("\n✓ Both balanced (binary) and unbalanced (multi-class) training sets prepared")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Step 10: Build sequences for TimeGAN
# TimeGAN needs sliding windows of consecutive samples grouped by flow
# Window size = 20 time steps is standard for network traffic
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN = 20   # number of time steps per sequence (adjustable)

def make_sequences(X_data, y_data, seq_len=SEQ_LEN):
    """
    Slide a window of seq_len over the data.
    Returns:
        sequences: (N, seq_len, n_features)
        seq_labels: (N,) — label of the LAST sample in each window
    """
    sequences = []
    seq_labels = []
    for i in range(len(X_data) - seq_len):
        sequences.append(X_data[i : i + seq_len])
        seq_labels.append(y_data[i + seq_len - 1])
    return np.array(sequences), np.array(seq_labels)

print("Building TimeGAN sequences (this may take a minute)...")

# Use MinMax-scaled data for TimeGAN (requires [0,1] range)
# Use a subset of training data to keep memory manageable
# Adjust N_SAMPLES based on your Colab RAM (50k is safe, 200k if Pro)
N_SAMPLES = 100_000
idx = np.random.choice(len(X_train_mm), size=min(N_SAMPLES, len(X_train_mm)), replace=False)
X_seq_input = X_train_mm[idx]
y_seq_input = y_bin_train[idx]

seq_train, seq_labels_train = make_sequences(X_seq_input, y_seq_input, SEQ_LEN)
print(f"✓ Training sequences shape:  {seq_train.shape}")
print(f"  (samples, timesteps, features) = {seq_train.shape}")

# Also split sequences for attack-only and benign-only
# (TimeGAN is trained separately on each class)
attack_mask  = seq_labels_train == 1
benign_mask  = seq_labels_train == 0
seq_attack   = seq_train[attack_mask]
seq_benign   = seq_train[benign_mask]
print(f"\n  Attack sequences:  {seq_attack.shape[0]:,}")
print(f"  Benign sequences:  {seq_benign.shape[0]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Step 11: Save all outputs for the team
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = '/content/preprocessed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Tabular data (for cGAN and IDS classifier) ─────────────────────────────
# Train — unbalanced, multi-class (for IDS + weighted loss)
pd.DataFrame(X_train_scaled, columns=feature_names).to_csv(
    f'{OUTPUT_DIR}/X_train.csv', index=False)
pd.Series(y_mc_train, name='label').to_csv(
    f'{OUTPUT_DIR}/y_train_multiclass.csv', index=False)
pd.Series(y_bin_train, name='label').to_csv(
    f'{OUTPUT_DIR}/y_train_binary.csv', index=False)

# Train — balanced, binary (for GAN discriminator training)
pd.DataFrame(X_train_bal, columns=feature_names).to_csv(
    f'{OUTPUT_DIR}/X_train_balanced.csv', index=False)
pd.Series(y_bin_train_bal, name='label').to_csv(
    f'{OUTPUT_DIR}/y_train_balanced_binary.csv', index=False)

# Validation
pd.DataFrame(X_val_scaled, columns=feature_names).to_csv(
    f'{OUTPUT_DIR}/X_val.csv', index=False)
pd.Series(y_mc_val, name='label').to_csv(
    f'{OUTPUT_DIR}/y_val_multiclass.csv', index=False)
pd.Series(y_bin_val, name='label').to_csv(
    f'{OUTPUT_DIR}/y_val_binary.csv', index=False)

# Test (DO NOT TOUCH until final evaluation)
pd.DataFrame(X_test_scaled, columns=feature_names).to_csv(
    f'{OUTPUT_DIR}/X_test.csv', index=False)
pd.Series(y_mc_test, name='label').to_csv(
    f'{OUTPUT_DIR}/y_test_multiclass.csv', index=False)
pd.Series(y_bin_test, name='label').to_csv(
    f'{OUTPUT_DIR}/y_test_binary.csv', index=False)

# ── Sequence data (for TimeGAN) ─────────────────────────────────────────────
np.save(f'{OUTPUT_DIR}/sequences_attack_train.npy', seq_attack)
np.save(f'{OUTPUT_DIR}/sequences_benign_train.npy', seq_benign)
np.save(f'{OUTPUT_DIR}/sequences_all_train.npy', seq_train)
np.save(f'{OUTPUT_DIR}/sequences_labels_train.npy', seq_labels_train)

# ── Artifacts (scalers, encoder, feature names) ─────────────────────────────
import shutil
shutil.copy('/content/label_encoder.pkl',   f'{OUTPUT_DIR}/label_encoder.pkl')
shutil.copy('/content/scaler_standard.pkl', f'{OUTPUT_DIR}/scaler_standard.pkl')
shutil.copy('/content/scaler_minmax.pkl',   f'{OUTPUT_DIR}/scaler_minmax.pkl')
shutil.copy('/content/feature_names.txt',   f'{OUTPUT_DIR}/feature_names.txt')

print("✓ All outputs saved to /content/preprocessed/")
print("\nFiles:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f'{OUTPUT_DIR}/{f}')
    print(f"  {f:<45s} {size/1e6:>7.2f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Step 12: EDA — Feature correlation heatmap (top 20 features)
# ─────────────────────────────────────────────────────────────────────────────

# Use a sample to keep this fast
sample_idx = np.random.choice(len(X_train_scaled), size=5000, replace=False)
X_sample = pd.DataFrame(X_train_scaled[sample_idx], columns=feature_names)

# Top 20 most correlated features with the binary label
correlations = X_sample.corrwith(
    pd.Series(y_bin_train[sample_idx], name='label')
).abs().sort_values(ascending=False)

top20 = correlations.head(20).index.tolist()
corr_matrix = X_sample[top20].corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
            linewidths=0.3, ax=ax)
ax.set_title('Feature correlation matrix (top 20 features by correlation with label)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Correlation heatmap saved")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — Step 13: EDA — Feature importance via variance (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

variances = X_sample.var().sort_values(ascending=False).head(25)

fig, ax = plt.subplots(figsize=(12, 6))
variances.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Top 25 features by variance (normalized training data)')
ax.set_ylabel('Variance')
ax.set_xlabel('Feature')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — Final summary printout for README / handoff doc
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("PREPROCESSING COMPLETE — TEAM HANDOFF SUMMARY")
print("=" * 60)
print(f"\nDataset: CICIDS 2017")
print(f"Original samples:     {df.shape[0]:>10,}")
print(f"Features (final):     {len(feature_names):>10,}")
print(f"\nSplits:")
print(f"  Train (unbalanced): {X_train_scaled.shape[0]:>10,}")
print(f"  Train (balanced):   {X_train_bal.shape[0]:>10,}")
print(f"  Validation:         {X_val_scaled.shape[0]:>10,}")
print(f"  Test:               {X_test_scaled.shape[0]:>10,}")
print(f"\nTimeGAN sequences (seq_len={SEQ_LEN}):")
print(f"  Attack sequences:   {seq_attack.shape[0]:>10,}")
print(f"  Benign sequences:   {seq_benign.shape[0]:>10,}")
print(f"  Shape per array:    (N, {SEQ_LEN}, {seq_attack.shape[2]})")
print(f"\nFiles in {OUTPUT_DIR}/:")
print(f"  X_train.csv                    → StandardScaler normalized, multi-class")
print(f"  X_train_balanced.csv           → SMOTE balanced, binary (for GAN)")
print(f"  X_val.csv / X_test.csv         → Same scaler, do not re-fit")
print(f"  y_*_multiclass.csv             → Integer class labels (0–14)")
print(f"  y_*_binary.csv                 → 0=BENIGN, 1=ATTACK")
print(f"  sequences_attack_train.npy     → For TimeGAN attacker training")
print(f"  sequences_benign_train.npy     → For TimeGAN benign modeling")
print(f"  scaler_standard.pkl            → Must use to inverse-transform")
print(f"  scaler_minmax.pkl              → For TimeGAN (input was [0,1])")
print(f"  label_encoder.pkl              → Decode integer → class name")
print(f"  feature_names.txt              → Column order for all CSVs")
print(f"\n⚠️  TEST SET IS HELD OUT — teammates should only use it for final eval")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 17 — (Optional) Zip everything for GitHub / Google Drive upload
# ─────────────────────────────────────────────────────────────────────────────

import shutil
shutil.make_archive('/content/preprocessed_CICIDS2017', 'zip', OUTPUT_DIR)
print("✓ /content/preprocessed_CICIDS2017.zip ready for upload")

# To download to your local machine in Colab:
# from google.colab import files
# files.download('/content/preprocessed_CICIDS2017.zip')
