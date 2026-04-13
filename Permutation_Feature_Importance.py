# ============================================================
# Permutation Feature Importance (PFI) pipeline
#
# This script:
# - trains all models once on clean training data
# - computes permutation feature importance on:
#     * clean test data
#     * all degraded test scenarios
# - supports both tabular and sequence models
# - exports one unified feature-importance table
#
# Degradations are applied to the test split only.
#
# Supported models:
#   Tabular:
#     - LinearSVM
#     - Random Forest
#     - XGBoost
#
#   Sequence:
#     - LSTM
#     - CNN-LSTM
#     - Informer-style classifier
#
# Output:
#   /kaggle/working/feature_importance_pfi_all_scenarios_all_buildings_all_models.csv
#
# Main properties:
# - one unified feature-importance method for all models: permutation FI
# - no target leakage:
#       split raw data -> fit scaler on train only -> transform train/test
# - train once per building using clean training data
# - degrade test data only, before scaling
# - tabular PFI:
#       permute one feature column across samples
# - sequence PFI:
#       block permutation across sequences while preserving within-sequence order
# ============================================================

import os
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBClassifier


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
LABEL_MODE = "raw"          # "raw" or "family"
TEST_SIZE = 0.20
MAX_SENSOR_COLS = 19

# Sequence configuration
SEQ_LEN = 10
SEQ_STRIDE = 1

# Optional caps for runtime control
MAX_TAB_TRAIN = 100000
MAX_SEQ_TRAIN = 250000
MAX_SEQ_TEST = 200000   # sequence evaluation will be subsampled later anyway

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 6
BATCH = 1024
LR = 8e-4
GRAD_CLIP = 1.0

TAB_SVM_MAX_ITER = 8000
RF_TREES = 250
XGB_TREES = 200

# Permutation feature importance configuration
FI_SEED = 123

PFI_METRIC = "balanced_acc"   # "balanced_acc" or "macro_f1"
PFI_REPEATS = 2               # repeats per feature, used for mean/std
TAB_PFI_EVAL = 2000           # max tabular test rows used per scenario
SEQ_PFI_EVAL = 512            # max test sequences used per scenario

# Optional feature cap for very fast runs
PFI_MAX_FEATURES = None       # e.g. 13 for faster runs; None = use all

# Test degradations
levels = {
    "clean":    [0],
    "noise":    [0.01, 0.05, 0.10],
    "drift":    [0.01, 0.05, 0.10],
    "bias":     [0.01, 0.05, 0.10],
    "missing":  [0.05, 0.10, 0.20],
    "sampling": [2, 4, 6],
}

OUT_PATH = "./pfi_out/feature_importance_pfi_all_scenarios_all_buildings_all_models.csv"


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------
def log_header(title):
    """Print a high-level section header."""
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def log_sub(title):
    """Print a subsection header."""
    print("\n" + "-" * 95)
    print(title)
    print("-" * 95)


def log_kv(key, value):
    """Print a simple key-value line."""
    print(f"   • {key}: {value}")


# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------
def sensor_cols_generic(df, exclude):
    """
    Return numeric columns that are not explicitly excluded.
    """
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


def sanitize_sensors(df, sensors):
    """
    Remove columns that clearly correspond to labels, ground truth,
    timestamps, or other metadata that should not be used as inputs.
    """
    banned_exact = {
        "FaultCode",
        "FaultName",
        "labeling",
        "FaultFamily",
        "FaultFamilyCode",
        "Fault Detection Ground Truth",
        "FaultDetectionGroundTruth",
        "Datetime",
        "timestamp",
        "Time",
        "DATE",
        "AHU name"
    }

    clean = []
    for c in sensors:
        cl = str(c).lower()

        if c in banned_exact:
            continue

        if (
            ("fault" in cl)
            or ("label" in cl)
            or ("ground truth" in cl)
            or ("truth" in cl)
            or ("code" in cl)
        ):
            continue

        clean.append(c)

    return clean


def assert_no_leakage(sensors):
    """
    Fail immediately if a likely leakage column is still present.
    """
    bad = [
        s for s in sensors
        if ("fault" in s.lower()) or ("label" in s.lower()) or ("truth" in s.lower()) or ("code" in s.lower())
    ]
    assert len(bad) == 0, f"Leakage columns detected in sensors: {bad}"


def clean_fault_name(s):
    """
    Normalize a fault name by removing trailing bracketed descriptions.
    """
    s = str(s).strip()
    if "(" in s:
        return s.split("(")[0].strip()
    return s


# ------------------------------------------------------------
# Fault family mapping
# ------------------------------------------------------------
FAULT_FAMILY_ORDER = [
    "NORMAL",
    "SENSOR_BIAS",
    "TEMP_FAULT",
    "VALVE_FAULT",
    "DAMPER_VENT_FAULT",
    "FAN_PUMP_FAULT",
    "EQUIPMENT_HEATX_FAULT",
    "SCHEDULING_SETBACK_FAULT",
    "OTHER"
]


def map_fault_to_family(label: str) -> str:
    """
    Map a fine-grained fault label to a broader fault family.
    """
    s = str(label).strip().lower()

    if s in ["normal", "normal condition"]:
        return "NORMAL"

    if "outdoor air temperature sensor bias" in s:
        return "SENSOR_BIAS"
    if "thermostat measurement bias" in s:
        return "SENSOR_BIAS"
    if "bias" in s and ("sensor" in s or "thermostat" in s or "measurement" in s):
        return "SENSOR_BIAS"

    if "temperature fault" in s or "cooling supply temperature fault" in s:
        return "TEMP_FAULT"

    if "valve" in s:
        return "VALVE_FAULT"

    if "damper" in s or "infiltration" in s:
        return "DAMPER_VENT_FAULT"

    if "fan fault" in s or "pump fault" in s:
        return "FAN_PUMP_FAULT"

    if "condenser fouling" in s or "fouling" in s:
        return "EQUIPMENT_HEATX_FAULT"

    if "setback" in s or "overnight" in s or "early termination" in s or "delayed onset" in s:
        return "SCHEDULING_SETBACK_FAULT"

    return "OTHER"


def build_family_codec_from_labels(labels):
    """
    Build family-to-id and id-to-family mappings.
    """
    fams = [map_fault_to_family(x) for x in labels]

    ordered = [f for f in FAULT_FAMILY_ORDER if f in set(fams)]
    for f in sorted(set(fams)):
        if f not in ordered:
            ordered.append(f)

    fam2id = {f: i for i, f in enumerate(ordered)}
    id2fam = {i: f for f, i in fam2id.items()}

    return fam2id, id2fam


# ------------------------------------------------------------
# Robust stratified split
# ------------------------------------------------------------
def stratified_split_keep_singletons_in_train(X, y, test_size=0.2, seed=42):
    """
    Perform a stratified train/test split while forcing singleton classes
    into the training set to avoid sklearn stratification failures.
    """
    y = np.asarray(y, dtype=int)
    n = len(y)

    if n < 5:
        split = max(1, int(n * (1 - test_size)))
        return X[:split], X[split:], y[:split], y[split:]

    vals, cnts = np.unique(y, return_counts=True)
    singleton_classes = vals[cnts < 2]

    if len(singleton_classes) == 0:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            shuffle=True,
            random_state=seed
        )

    idx_all = np.arange(n)
    idx_single = idx_all[np.isin(y, singleton_classes)]
    idx_rest = idx_all[~np.isin(y, singleton_classes)]

    X_rest, y_rest = X[idx_rest], y[idx_rest]
    if len(X_rest) < 5:
        return X, X[:0], y, y[:0]

    vals2, cnts2 = np.unique(y_rest, return_counts=True)
    if cnts2.min() < 2:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_rest,
            y_rest,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=None
        )
    else:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_rest,
            y_rest,
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=y_rest
        )

    Xtr = np.concatenate([Xtr_r, X[idx_single]], axis=0)
    ytr = np.concatenate([ytr_r, y[idx_single]], axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))

    return Xtr[perm], Xte_r, ytr[perm], yte_r


# ------------------------------------------------------------
# Dataset loaders for individual buildings
# ------------------------------------------------------------
def merge_lbnl_sensor_fault(sensor_df, fault_df, max_sensor_cols=20):
    """
    Merge LBNL sensor measurements with fault intervals and create
    raw and family labels.
    """
    df = sensor_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df["FaultName"] = "Normal"

    fault_df = fault_df.rename(columns=lambda x: x.strip().lower())
    fault_col = next(c for c in fault_df.columns if "fault" in c)
    time_col = next(c for c in fault_df.columns if "time" in c)

    for _, row in fault_df.iterrows():
        fname = clean_fault_name(row[fault_col])
        t_raw = str(row[time_col])
        t = (
            t_raw.replace("TO", "to")
            .replace("To", "to")
            .replace(" - ", " to ")
            .replace("—", " to ")
            .replace("-", " to ")
        )
        parts = [p.strip() for p in t.split("to")]

        if len(parts) == 2:
            start = pd.to_datetime(parts[0], errors="coerce")
            end = pd.to_datetime(parts[1], errors="coerce")
        else:
            start = pd.to_datetime(parts[0], errors="coerce")
            end = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

        if pd.isna(start) or pd.isna(end):
            continue

        mask = (df["Datetime"] >= start) & (df["Datetime"] <= end)
        df.loc[mask, "FaultName"] = fname

    uniq = sorted(df["FaultName"].unique())
    fmap = {f: i for i, f in enumerate(uniq)}
    inv = {i: f for f, i in fmap.items()}
    df["FaultCode"] = df["FaultName"].map(fmap).astype(int)

    fam2id, id2fam = build_family_codec_from_labels(uniq)
    df["FaultFamily"] = df["FaultName"].map(map_fault_to_family)
    df["FaultFamilyCode"] = df["FaultFamily"].map(fam2id).astype(int)

    sensors = sensor_cols_generic(
        df,
        exclude=["Datetime", "FaultName", "FaultCode", "FaultFamily", "FaultFamilyCode"]
    )
    sensors = sanitize_sensors(df, sensors)
    assert_no_leakage(sensors)
    sensors = sensors[:min(max_sensor_cols, len(sensors))]

    df[sensors] = df[sensors].ffill().bfill()
    meta = {"inv_fault_map": inv, "inv_family_map": id2fam}
    return df, sensors, meta


def preprocess_wang_building(df, max_sensor_cols=20):
    """
    Preprocess one building from the Wang dataset and create raw/family labels.
    """
    df = df.copy()

    if "DATE" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
            dayfirst=True
        )
    elif "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Time"].astype(str), errors="coerce")
    else:
        raise ValueError("No time column found")

    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass

    df["labeling"] = df["labeling"].astype(str)
    df["FaultCode"] = df["labeling"].astype("category").cat.codes
    cats = df["labeling"].astype("category").cat.categories
    inv_map = {i: cat for i, cat in enumerate(cats)}

    fam2id, id2fam = build_family_codec_from_labels(list(cats))
    df["FaultFamily"] = df["labeling"].map(map_fault_to_family)
    df["FaultFamilyCode"] = df["FaultFamily"].map(fam2id).astype(int)

    exclude = [
        "timestamp", "Time", "DATE", "AHU name",
        "labeling", "FaultCode", "FaultFamily", "FaultFamilyCode"
    ]
    sensors = sensor_cols_generic(df, exclude=exclude)
    sensors = sanitize_sensors(df, sensors)
    assert_no_leakage(sensors)
    sensors = sensors[:min(max_sensor_cols, len(sensors))]

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df[sensors] = df[sensors].ffill().bfill()

    meta = {"inv_fault_map": inv_map, "inv_family_map": id2fam}
    return df, sensors, meta


# ------------------------------------------------------------
# Sequence helpers
# ------------------------------------------------------------
def create_sequences(X, y, seq_len=24, stride=1):
    """
    Convert row-wise time-ordered data into fixed-length sequences.

    Each sequence receives the label of its final time step.
    """
    Xs, ys = [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])

    if len(Xs) == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=int)

    return np.stack(Xs), np.array(ys, dtype=int)


def cap(X, y, max_n, seed=42):
    """
    Randomly subsample a dataset to at most max_n rows/sequences.
    """
    if max_n is None or len(X) <= max_n:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


def subsample_xy(X, y, max_n, seed=123):
    """
    Randomly subsample data for evaluation-time feature-importance computation.
    """
    if max_n is None or len(X) <= max_n:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


# ------------------------------------------------------------
# Torch models
# ------------------------------------------------------------
class TinyLSTM(nn.Module):
    """
    Lightweight LSTM classifier using the final time step output.
    """
    def __init__(self, input_dim, num_classes, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TinyCNNLSTM(nn.Module):
    """
    Lightweight Conv1D + LSTM classifier.
    """
    def __init__(self, input_dim, num_classes, channels=32, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            channels,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = torch.relu(self.cnn(x))
        x = x.transpose(2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TinyInformerClassifier(nn.Module):
    """
    Lightweight Transformer-based classifier in an Informer-style setup.
    """
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        z = self.encoder(x)
        return self.cls(z[:, -1, :])


def train_torch(model, Xtr, ytr, epochs=6, batch_size=1024, lr=8e-4, grad_clip=1.0):
    """
    Train a sequence model using cross-entropy loss.
    """
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long()),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

    return model


# ------------------------------------------------------------
# Test-set degradation
# ------------------------------------------------------------
def apply_degradation(X_raw, manip, level, seed, train_mu, train_sigma):
    """
    Apply one degradation scenario to raw test data before scaling.

    Returns:
    - degraded feature matrix
    - kept row indices (important for sampling degradation)
    """
    rng = np.random.default_rng(seed)
    X = X_raw.astype(np.float32, copy=True)
    N, F = X.shape

    m = str(manip).strip().lower()
    if m == "clean":
        return X, np.arange(N)

    sig = np.asarray(train_sigma, dtype=np.float32)
    sig = np.where(sig <= 1e-8, 1.0, sig)

    if m == "noise":
        eps = rng.normal(0.0, 1.0, size=X.shape).astype(np.float32)
        X = X + (level * sig)[None, :] * eps
        return X, np.arange(N)

    if m == "bias":
        X = X + (level * sig)[None, :]
        return X, np.arange(N)

    if m == "drift":
        t = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None]
        X = X + t * (level * sig)[None, :]
        return X, np.arange(N)

    if m == "missing":
        frac = float(level)
        mask = rng.uniform(0.0, 1.0, size=X.shape) < frac
        X = X.astype(np.float32)
        X[mask] = np.nan

        mu = np.asarray(train_mu, dtype=np.float32)
        inds = np.where(np.isnan(X))
        if len(inds[0]) > 0:
            X[inds] = np.take(mu, inds[1])

        return X, np.arange(N)

    if m == "sampling":
        k = int(level)
        k = max(1, k)
        idx = np.arange(0, N, k)
        return X[idx], idx

    return X, np.arange(N)


# ------------------------------------------------------------
# Building inventory
# ------------------------------------------------------------
def iter_all_buildings():
    """
    Yield metadata for all supported buildings.
    """
    RAW_LBNL = Path("./data_FDD/5_lbnl_data_synthesis_inventory/raw/")
    lbnl_map = {
        "SZCAV":     {"sensor": RAW_LBNL / "SZCAV.csv",     "faults": RAW_LBNL / "SZCAV-faults.csv"},
        "SZVAV":     {"sensor": RAW_LBNL / "SZVAV.csv",     "faults": RAW_LBNL / "SZVAV-faults.csv"},
        "MZVAV_1":   {"sensor": RAW_LBNL / "MZVAV-1.csv",   "faults": RAW_LBNL / "MZVAV-1-faults.csv"},
        "MZVAV_2_1": {"sensor": RAW_LBNL / "MZVAV-2-1.csv", "faults": RAW_LBNL / "MZVAV-2-1-faults.csv"},
        "MZVAV_2_2": {"sensor": RAW_LBNL / "MZVAV-2-2.csv", "faults": RAW_LBNL / "MZVAV-2-2-faults.csv"},
    }
    for building, paths in lbnl_map.items():
        yield {"dataset": "LBNL_DataFDD", "building": building, "paths": paths}

    ROOT_WANG = Path("./data_FDD/8_nature_lcu_wang/raw/")
    wang_files = {
        "auditorium": ROOT_WANG / "auditorium_scientific_data.csv",
        "office":     ROOT_WANG / "office_scientific_data.csv",
        "hospital":   ROOT_WANG / "hosptial_scientific_data.csv",
    }
    for building, fp in wang_files.items():
        if fp.exists():
            yield {"dataset": "Nature_LCU_Wang", "building": building, "paths": {"csv": fp}}


# ------------------------------------------------------------
# Scenario naming helpers
# ------------------------------------------------------------
def fmt_level(manip, lv):
    """
    Format scenario level for display and output.
    """
    m = str(manip).strip().lower()
    if m == "sampling":
        return str(int(lv))

    try:
        lv = float(lv)
        return f"{int(round(lv * 100))}%"
    except Exception:
        return str(lv)


def scenario_label(manip, lv):
    """
    Create a readable scenario label such as:
    - clean
    - noise_5%
    - sampling_4
    """
    m = str(manip).strip().lower()
    if m == "clean":
        return "clean"
    return f"{m}_{fmt_level(m, lv)}"


# ------------------------------------------------------------
# Prediction and metric helpers
# ------------------------------------------------------------
def _metric_score(y_true, y_pred, metric="balanced_acc"):
    """
    Compute the configured evaluation metric for feature importance.
    """
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(balanced_accuracy_score(y_true, y_pred))


def predict_tabular(model_name, model, X):
    """
    Unified prediction interface for tabular models.
    """
    if model_name == "LinearSVM":
        return model.predict(X)
    if model_name == "XGBoost":
        return model.predict(X)
    return model.predict(X)


@torch.no_grad()
def predict_sequence(model, X_seq, batch_size=1024):
    """
    Batched prediction helper for sequence models.
    """
    model.eval()
    preds = []

    loader = DataLoader(torch.tensor(X_seq).float(), batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


# ------------------------------------------------------------
# Permutation feature importance: tabular
# ------------------------------------------------------------
def pfi_tabular(model_name, model, X, y, metric="balanced_acc", repeats=2, seed=123, max_eval=2000, max_features=None):
    """
    Compute permutation feature importance for a tabular model.

    One feature column is permuted across samples at a time.
    """
    X, y = subsample_xy(X, y, max_eval, seed=seed)
    if len(X) == 0:
        return None, None, None

    rng = np.random.default_rng(seed)
    base_pred = predict_tabular(model_name, model, X)
    base_score = _metric_score(y, base_pred, metric)

    F = X.shape[1]
    feat_idx = np.arange(F)
    if max_features is not None:
        feat_idx = feat_idx[:min(F, int(max_features))]

    imps = np.zeros((len(feat_idx), repeats), dtype=np.float32)

    X_work = X.copy()
    for i, j in enumerate(feat_idx):
        col_orig = X_work[:, j].copy()
        for r in range(repeats):
            perm = rng.permutation(len(X_work))
            X_work[:, j] = col_orig[perm]
            pred = predict_tabular(model_name, model, X_work)
            score = _metric_score(y, pred, metric)
            imps[i, r] = base_score - score
        X_work[:, j] = col_orig

    mean_imp = imps.mean(axis=1)
    std_imp = imps.std(axis=1)
    return feat_idx, mean_imp, std_imp


# ------------------------------------------------------------
# Permutation feature importance: sequence
# ------------------------------------------------------------
def pfi_sequence(model, X_seq, y_seq, metric="balanced_acc", repeats=2, seed=123, max_eval=512, max_features=None):
    """
    Compute block-permutation feature importance for a sequence model.

    For each feature:
    - the full temporal block (T,) is shuffled across sequences
    - within-sequence time order is preserved
    """
    X_seq, y_seq = subsample_xy(X_seq, y_seq, max_eval, seed=seed)
    if len(X_seq) == 0:
        return None, None, None

    rng = np.random.default_rng(seed)
    base_pred = predict_sequence(model, X_seq, batch_size=1024)
    base_score = _metric_score(y_seq, base_pred, metric)

    N, T, F = X_seq.shape
    feat_idx = np.arange(F)
    if max_features is not None:
        feat_idx = feat_idx[:min(F, int(max_features))]

    imps = np.zeros((len(feat_idx), repeats), dtype=np.float32)

    X_work = X_seq.copy()
    for i, j in enumerate(feat_idx):
        block_orig = X_work[:, :, j].copy()
        for r in range(repeats):
            perm = rng.permutation(N)
            X_work[:, :, j] = block_orig[perm]
            pred = predict_sequence(model, X_work, batch_size=1024)
            score = _metric_score(y_seq, pred, metric)
            imps[i, r] = base_score - score
        X_work[:, :, j] = block_orig

    mean_imp = imps.mean(axis=1)
    std_imp = imps.std(axis=1)
    return feat_idx, mean_imp, std_imp


# ------------------------------------------------------------
# Output-row helper
# ------------------------------------------------------------
def add_rows(rows, dataset, building, label_mode, tag, fi_type, model_name,
             feat_names, feat_idx, mean_imp, std_imp, method, manip, level):
    """
    Append feature-importance rows to the global output list.
    """
    if mean_imp is None or feat_idx is None:
        return

    if std_imp is None:
        std_imp = np.zeros_like(mean_imp, dtype=float)

    for j, mu, sd in zip(feat_idx, mean_imp, std_imp):
        rows.append({
            "dataset": dataset,
            "building": building,
            "label_mode": label_mode,
            "tag": tag,
            "fi_type": fi_type,
            "model": model_name,
            "feature": str(feat_names[int(j)]),
            "importance_mean": float(mu),
            "importance_std": float(sd),
            "method": method,
            "manipulation": str(manip),
            "level": float(level) if str(manip).lower() != "sampling" else float(int(level)),
            "scenario": scenario_label(manip, level),
        })


# ============================================================
# Main run
# ============================================================
log_header("PFI run: all scenarios × all buildings × all models")
log_kv("Device", DEVICE)
log_kv("Label mode", LABEL_MODE)
log_kv("SEQ_LEN", SEQ_LEN)
log_kv("Metric", PFI_METRIC)
log_kv("Repeats", PFI_REPEATS)
log_kv("TAB_PFI_EVAL", TAB_PFI_EVAL)
log_kv("SEQ_PFI_EVAL", SEQ_PFI_EVAL)
log_kv("PFI_MAX_FEATURES", PFI_MAX_FEATURES)

all_rows = []
t_global = time.time()

for item in iter_all_buildings():
    dataset = item["dataset"]
    building = item["building"]
    tag = f"{dataset}__{building}__{LABEL_MODE.upper()}"

    log_header(f"{dataset} | {building}")

    # --------------------------------------------------------
    # Load and preprocess one building
    # --------------------------------------------------------
    t0 = time.time()

    if dataset == "LBNL_DataFDD":
        p = item["paths"]
        if (not p["sensor"].exists()) or (not p["faults"].exists()):
            log_kv("Skip", "missing files")
            continue

        sensor_df = pd.read_csv(p["sensor"])
        fault_df = pd.read_csv(p["faults"])
        df, sensors, meta = merge_lbnl_sensor_fault(
            sensor_df,
            fault_df,
            max_sensor_cols=MAX_SENSOR_COLS
        )
    else:
        fp = item["paths"]["csv"]
        df0 = pd.read_csv(fp)
        df, sensors, meta = preprocess_wang_building(
            df0,
            max_sensor_cols=MAX_SENSOR_COLS
        )

    log_kv("Load seconds", f"{time.time() - t0:.2f}")
    log_kv("Sensors", len(sensors))

    # --------------------------------------------------------
    # Select label mode
    # --------------------------------------------------------
    y_col = "FaultFamilyCode" if LABEL_MODE == "family" else "FaultCode"
    X_all_raw = df[sensors].to_numpy().astype(np.float32)
    y_all = df[y_col].to_numpy().astype(int)

    vals, cnts = np.unique(y_all, return_counts=True)
    log_kv("X shape", X_all_raw.shape)
    log_kv("Classes", len(vals))
    log_kv("Min class count", int(cnts.min()))

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    Xtr_raw, Xte_raw, ytr_raw, yte_raw = stratified_split_keep_singletons_in_train(
        X_all_raw,
        y_all,
        test_size=TEST_SIZE,
        seed=42
    )
    log_kv("Train/Test", f"{len(Xtr_raw)}/{len(Xte_raw)}")

    if len(Xte_raw) < 50 or len(Xtr_raw) < 100:
        log_kv("Skip", "too few samples after split")
        continue

    # --------------------------------------------------------
    # Train statistics for degradation scale and missing-value fill
    # --------------------------------------------------------
    train_mu = np.nanmean(Xtr_raw, axis=0)
    train_sigma = np.nanstd(Xtr_raw, axis=0)

    # --------------------------------------------------------
    # Fit scaler on clean train only
    # --------------------------------------------------------
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_m = scaler.transform(Xtr_raw)

    # Optional cap for tabular training
    Xtr_fit, ytr_fit = cap(Xtr_m, ytr_raw, MAX_TAB_TRAIN, seed=42)

    # --------------------------------------------------------
    # Build clean training sequences once
    # --------------------------------------------------------
    Xtr_seq, ytr_seq = create_sequences(Xtr_m, ytr_raw, seq_len=SEQ_LEN, stride=SEQ_STRIDE)
    Xtr_seq, ytr_seq = cap(Xtr_seq, ytr_seq, MAX_SEQ_TRAIN, seed=42)
    log_kv("Train seq count", len(Xtr_seq))

    # --------------------------------------------------------
    # Infer number of classes from clean training labels
    # --------------------------------------------------------
    n_classes = int(np.max(ytr_raw) + 1) if len(ytr_raw) else 0
    log_kv("n_classes", n_classes)

    # --------------------------------------------------------
    # Train models once on clean training data
    # --------------------------------------------------------
    t_train = time.time()

    tab_models = {
        "LinearSVM": LinearSVC(
            C=1.0,
            dual=False,
            max_iter=TAB_SVM_MAX_ITER,
            class_weight="balanced"
        ),
        "RF": RandomForestClassifier(
            n_estimators=RF_TREES,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2
        ),
    }

    for name, mdl in tab_models.items():
        tt = time.time()
        mdl.fit(Xtr_fit, ytr_fit)
        log_kv(f"Train {name} s", f"{time.time() - tt:.2f}")

    # XGBoost requires label encoding to contiguous class ids
    le = LabelEncoder()
    ytr_xgb = le.fit_transform(ytr_fit)
    K = len(le.classes_)

    if K >= 2:
        xgb_model = XGBClassifier(
            tree_method="hist",
            max_depth=5,
            n_estimators=XGB_TREES,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="multi:softmax",
            num_class=K,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42
        )
        tt = time.time()
        xgb_model.fit(Xtr_fit, ytr_xgb, verbose=False)
        log_kv("Train XGBoost s", f"{time.time() - tt:.2f}")
        tab_models["XGBoost"] = xgb_model
    else:
        log_kv("XGBoost", f"skip (K={K})")

    # Sequence models
    seq_models = {}
    if len(Xtr_seq) >= 200 and n_classes >= 2:
        tt = time.time()
        seq_models["LSTM"] = train_torch(
            TinyLSTM(Xtr_seq.shape[2], n_classes),
            Xtr_seq,
            ytr_seq,
            epochs=EPOCHS,
            batch_size=BATCH,
            lr=LR,
            grad_clip=GRAD_CLIP
        )
        log_kv("Train LSTM s", f"{time.time() - tt:.2f}")

        tt = time.time()
        seq_models["CNN-LSTM"] = train_torch(
            TinyCNNLSTM(Xtr_seq.shape[2], n_classes),
            Xtr_seq,
            ytr_seq,
            epochs=EPOCHS,
            batch_size=BATCH,
            lr=LR,
            grad_clip=GRAD_CLIP
        )
        log_kv("Train CNN-LSTM s", f"{time.time() - tt:.2f}")

        tt = time.time()
        seq_models["Informer"] = train_torch(
            TinyInformerClassifier(Xtr_seq.shape[2], n_classes),
            Xtr_seq,
            ytr_seq,
            epochs=EPOCHS,
            batch_size=BATCH,
            lr=LR,
            grad_clip=GRAD_CLIP
        )
        log_kv("Train Informer s", f"{time.time() - tt:.2f}")
    else:
        log_kv("Seq models", "skip (too few sequences or classes)")

    log_kv("Total train phase s", f"{time.time() - t_train:.2f}")

    # ========================================================
    # Evaluate PFI on clean and degraded test scenarios
    # ========================================================
    for manip, lv_list in levels.items():
        for lv in lv_list:
            scen = scenario_label(manip, lv)
            log_sub(f"PFI scenario: {scen}")

            # ------------------------------------------------
            # Degrade raw test data before scaling
            # ------------------------------------------------
            Xte_raw_deg, keep_idx = apply_degradation(
                Xte_raw,
                manip,
                lv,
                seed=FI_SEED,
                train_mu=train_mu,
                train_sigma=train_sigma
            )
            yte_deg = yte_raw[keep_idx]

            if len(Xte_raw_deg) < 50:
                log_kv("Skip", f"too few test samples after degradation ({len(Xte_raw_deg)})")
                continue

            # ------------------------------------------------
            # Scale degraded test data with clean-train scaler
            # ------------------------------------------------
            Xte_m = scaler.transform(Xte_raw_deg)

            # ------------------------------------------------
            # Build degraded test sequences
            # ------------------------------------------------
            Xte_seq, yte_seq = create_sequences(Xte_m, yte_deg, seq_len=SEQ_LEN, stride=SEQ_STRIDE)
            Xte_seq, yte_seq = cap(Xte_seq, yte_seq, MAX_SEQ_TEST, seed=43)
            log_kv("Test seq count", len(Xte_seq))

            # ------------------------------------------------
            # Tabular PFI
            # ------------------------------------------------
            method_tab = f"pfi_tab_metric={PFI_METRIC}_R={PFI_REPEATS}_eval={TAB_PFI_EVAL}"

            for name, mdl in tab_models.items():
                tfi = time.time()

                if name == "XGBoost":
                    # Keep only classes seen during XGBoost training
                    mask = np.isin(yte_deg, le.classes_)
                    X_eval = Xte_m[mask]
                    y_eval = yte_deg[mask]

                    if len(X_eval) < 50:
                        log_kv("PFI XGBoost", "skip (too few masked test samples)")
                        continue

                    y_eval_enc = le.transform(y_eval)

                    feat_idx, mean_imp, std_imp = pfi_tabular(
                        "XGBoost",
                        mdl,
                        X_eval,
                        y_eval_enc,
                        metric=PFI_METRIC,
                        repeats=PFI_REPEATS,
                        seed=FI_SEED,
                        max_eval=TAB_PFI_EVAL,
                        max_features=PFI_MAX_FEATURES
                    )

                    add_rows(
                        all_rows,
                        dataset,
                        building,
                        LABEL_MODE,
                        tag,
                        "pfi_tab",
                        "XGBoost",
                        sensors,
                        feat_idx,
                        mean_imp,
                        std_imp,
                        method_tab,
                        manip,
                        lv
                    )
                else:
                    feat_idx, mean_imp, std_imp = pfi_tabular(
                        name,
                        mdl,
                        Xte_m,
                        yte_deg,
                        metric=PFI_METRIC,
                        repeats=PFI_REPEATS,
                        seed=FI_SEED,
                        max_eval=TAB_PFI_EVAL,
                        max_features=PFI_MAX_FEATURES
                    )

                    add_rows(
                        all_rows,
                        dataset,
                        building,
                        LABEL_MODE,
                        tag,
                        "pfi_tab",
                        name,
                        sensors,
                        feat_idx,
                        mean_imp,
                        std_imp,
                        method_tab,
                        manip,
                        lv
                    )

                log_kv(f"PFI {name} s", f"{time.time() - tfi:.2f}")

            # ------------------------------------------------
            # Sequence PFI
            # ------------------------------------------------
            if len(seq_models) == 0 or len(Xte_seq) < 80:
                log_kv("Seq PFI", f"skip (seq_models={len(seq_models)}, Xte_seq={len(Xte_seq)})")
            else:
                method_seq = f"pfi_seq_metric={PFI_METRIC}_R={PFI_REPEATS}_eval={SEQ_PFI_EVAL}_block"

                for name, mdl in seq_models.items():
                    tfi = time.time()

                    feat_idx, mean_imp, std_imp = pfi_sequence(
                        mdl,
                        Xte_seq,
                        yte_seq,
                        metric=PFI_METRIC,
                        repeats=PFI_REPEATS,
                        seed=FI_SEED,
                        max_eval=SEQ_PFI_EVAL,
                        max_features=PFI_MAX_FEATURES
                    )

                    add_rows(
                        all_rows,
                        dataset,
                        building,
                        LABEL_MODE,
                        tag,
                        "pfi_seq",
                        name,
                        sensors,
                        feat_idx,
                        mean_imp,
                        std_imp,
                        method_seq,
                        manip,
                        lv
                    )

                    log_kv(f"PFI {name} s", f"{time.time() - tfi:.2f}")


# ============================================================
# Save output
# ============================================================
log_header("Save")

fi_df = pd.DataFrame(all_rows)
fi_df.to_csv(OUT_PATH, index=False)

log_kv("Saved", OUT_PATH)
log_kv("Rows", len(fi_df))
log_kv("Buildings", fi_df[["dataset", "building"]].drop_duplicates().shape[0] if len(fi_df) else 0)
log_kv("Models", sorted(fi_df["model"].unique()) if len(fi_df) else "none")
log_kv("FI types", sorted(fi_df["fi_type"].unique()) if len(fi_df) else "none")
log_kv("Scenarios", len(sorted(fi_df["scenario"].unique())) if len(fi_df) else 0)
log_kv("Total runtime s", f"{time.time() - t_global:.2f}")

display(fi_df.head(25))