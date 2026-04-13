# ============================================================
# Feature-importance pipeline for all buildings and all models
#
# This script:
# - reloads each dataset from disk
# - performs a leakage-safe train/test split
# - fits models on clean training data only
# - computes feature importance on the clean test set only
# - exports one unified feature-importance table across all buildings
#
# Output:sage
#   /kaggle/working/feature_importance_clean_all_buildings_all_models.csv
#
# Supported datasets:
#   - LBNL_DataFDD
#       * RTU
#       * SZCAV
#       * SZVAV
#       * MZVAV_1
#       * MZVAV_2_1
#       * MZVAV_2_2
#   - Nature_LCU_Wang
#       * auditorium
#       * office
#       * hospital
#
# Main properties:
#   - leakage-safe preprocessing:
#       split raw data -> fit scaler on train only -> transform train/test
#   - robust stratified split:
#       singleton classes are forced into the training set
#   - train each model once per building
#   - compute feature importance on clean test data only
#   - save one unified feature-importance table with dataset/building metadata
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
from sklearn.metrics import f1_score

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

# Sequence window settings
SEQ_LEN = 10
SEQ_STRIDE = 1

# Optional caps to control runtime
MAX_TAB_TRAIN = 100000
MAX_SEQ_TRAIN = 250000
MAX_SEQ_TEST = 60000

# Training configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
BATCH = 1024
LR = 8e-4
GRAD_CLIP = 1.0

TAB_SVM_MAX_ITER = 8000
RF_TREES = 300
XGB_TREES = 250

# Feature importance configuration
FI_SEED = 123

# Tabular feature importance: SAGE-style Monte Carlo approximation
TAB_SAGE_PERMS = 6
TAB_MAX_SAMPLES = 1500
TAB_MASKING = "mean"   # "mean" is stable; "sample" is noisier

# Sequence feature importance: permutation-drop importance
SEQ_MAX_SAMPLES = 1200
SEQ_REPEATS = 3
SEQ_BATCH = 1024

OUT_PATH = "./SAGE_out/feature_importance_clean_all_buildings_all_models.csv"


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
    Remove known label or metadata columns to avoid target leakage.
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
    Fail immediately if a likely leakage column is present in the sensor list.
    """
    bad = [
        s for s in sensors
        if ("fault" in s.lower()) or ("label" in s.lower()) or ("truth" in s.lower()) or ("code" in s.lower())
    ]
    assert len(bad) == 0, f"Leakage columns detected in sensors: {bad}"


def clean_fault_name(s):
    """
    Normalize a fault name by removing trailing bracketed details.
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
    Map fine-grained fault labels to broader fault families.
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
    Build fault-family label encoders from a list of raw labels.
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
    Perform a stratified split while safely handling singleton classes.

    Classes with fewer than 2 samples are forced into the training set so that
    sklearn stratified splitting does not fail.
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
    Merge LBNL sensor data with fault intervals and create raw/family labels.
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

    meta = {
        "inv_fault_map": inv,
        "inv_family_map": id2fam
    }
    return df, sensors, meta


def preprocess_wang_building(df, max_sensor_cols=20):
    """
    Preprocess one Wang dataset building and create raw/family labels.
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

    meta = {
        "inv_fault_map": inv_map,
        "inv_family_map": id2fam
    }
    return df, sensors, meta


# ------------------------------------------------------------
# Sequence construction and dataset caps
# ------------------------------------------------------------
def create_sequences(X, y, seq_len=24, stride=1):
    """
    Convert row-wise data into fixed-length sequences.

    The label assigned to each sequence is the label of its final time step.
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


# ------------------------------------------------------------
# Feature-importance helpers
# ------------------------------------------------------------
def coerce_pred_labels(y_pred):
    """
    Convert predictions to a flat integer label array.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.ravel(y_pred)
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = y_pred.astype(int)
    return y_pred


def metric_macro_f1(y_true, y_pred):
    """
    Macro-F1 evaluation metric used for feature-importance scoring.
    """
    y_pred = coerce_pred_labels(y_pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _subsample_xy(X, y, max_n, rng):
    """
    Optional random subsampling used before feature-importance computation.
    """
    if max_n is None or len(X) <= max_n:
        return X, y

    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


# ------------------------------------------------------------
# Tabular feature importance (SAGE-style approximation)
# ------------------------------------------------------------
def _mask_tabular(X, X_train, feat_idx, rng, masking="mean"):
    """
    Mask selected tabular features either by:
    - replacing with training mean
    - replacing with sampled training values
    """
    Xm = X.copy()

    if masking == "mean":
        col_mean = X_train.mean(axis=0)
        for j in feat_idx:
            Xm[:, j] = col_mean[j]
    elif masking == "sample":
        for j in feat_idx:
            src = X_train[:, j]
            Xm[:, j] = rng.choice(src, size=len(Xm), replace=True)
    else:
        raise ValueError("masking must be 'sample' or 'mean'")

    return Xm


def sage_importance_tabular(
    predict_fn,
    X_test,
    y_test,
    X_train,
    metric_fn,
    n_perms=6,
    seed=123,
    max_samples=1500,
    masking="mean"
):
    """
    Approximate global feature importance for tabular models using a
    SAGE-like Monte Carlo procedure.

    Returns mean and standard deviation of importance across permutations.
    """
    rng = np.random.default_rng(seed)
    Xs, ys = _subsample_xy(X_test, y_test, max_samples, rng)

    if len(Xs) == 0:
        return None, None

    F = Xs.shape[1]
    feat_idx = np.arange(F)
    contrib = np.zeros((n_perms, F), dtype=float)

    for p in range(n_perms):
        order = rng.permutation(F)

        Xmasked = _mask_tabular(Xs, X_train, feat_idx, rng, masking=masking)
        pred0 = coerce_pred_labels(predict_fn(Xmasked))
        prev = metric_fn(ys, pred0)

        Xcurr = Xmasked.copy()
        for j in order:
            Xcurr[:, j] = Xs[:, j]
            pred1 = coerce_pred_labels(predict_fn(Xcurr))
            sc1 = metric_fn(ys, pred1)
            contrib[p, j] = sc1 - prev
            prev = sc1

    return contrib.mean(axis=0), contrib.std(axis=0)


# ------------------------------------------------------------
# Sequence feature importance (permutation-drop)
# ------------------------------------------------------------
def torch_predict_labels(model, X, device, batch_size=1024):
    """
    Batched prediction helper for PyTorch sequence models.
    """
    model.eval()
    preds = []

    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size]).float().to(device, non_blocking=True)
            preds.append(model(xb).argmax(1).cpu().numpy())

    return np.concatenate(preds, axis=0)


def permutation_importance_sequence(
    model,
    Xseq_test,
    yseq_test,
    metric_fn,
    seed=123,
    n_repeats=3,
    max_samples=1200,
    device=None,
    batch_size=1024
):
    """
    Compute permutation-drop feature importance for sequence models.

    Each feature is permuted across samples while preserving its temporal shape.
    The importance is measured as the drop in macro-F1.
    """
    rng = np.random.default_rng(seed)
    Xs, ys = _subsample_xy(Xseq_test, yseq_test, max_samples, rng)

    if len(Xs) == 0:
        return None, None

    base_pred = coerce_pred_labels(
        torch_predict_labels(model, Xs, device=device, batch_size=batch_size)
    )
    base_score = metric_fn(ys, base_pred)

    F = Xs.shape[2]
    mean_imp = np.zeros(F, dtype=float)
    std_imp = np.zeros(F, dtype=float)

    for j in range(F):
        drops = []

        for _ in range(n_repeats):
            Xp = Xs.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, :, j] = Xp[perm, :, j]

            pred = coerce_pred_labels(
                torch_predict_labels(model, Xp, device=device, batch_size=batch_size)
            )
            sc = metric_fn(ys, pred)
            drops.append(base_score - sc)

        mean_imp[j] = float(np.mean(drops))
        std_imp[j] = float(np.std(drops))

    return mean_imp, std_imp


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


def train_torch(model, Xtr, ytr, epochs=8, batch_size=1024, lr=8e-4, grad_clip=1.0):
    """
    Train a PyTorch classifier on sequence data.
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
# Building inventory
# ------------------------------------------------------------
def iter_all_buildings():
    """
    Yield metadata for all supported buildings across all supported datasets.
    """
    # LBNL
    RAW_LBNL = Path("./data_FDD/5_lbnl_data_synthesis_inventory/raw/")
    lbnl_map = {
        "RTU":       {"sensor": RAW_LBNL / "RTU.csv",       "faults": RAW_LBNL / "RTU-faults.csv"},
        "SZCAV":     {"sensor": RAW_LBNL / "SZCAV.csv",     "faults": RAW_LBNL / "SZCAV-faults.csv"},
        "SZVAV":     {"sensor": RAW_LBNL / "SZVAV.csv",     "faults": RAW_LBNL / "SZVAV-faults.csv"},
        "MZVAV_1":   {"sensor": RAW_LBNL / "MZVAV-1.csv",   "faults": RAW_LBNL / "MZVAV-1-faults.csv"},
        "MZVAV_2_1": {"sensor": RAW_LBNL / "MZVAV-2-1.csv", "faults": RAW_LBNL / "MZVAV-2-1-faults.csv"},
        "MZVAV_2_2": {"sensor": RAW_LBNL / "MZVAV-2-2.csv", "faults": RAW_LBNL / "MZVAV-2-2-faults.csv"},
    }

    for building, paths in lbnl_map.items():
        yield {
            "dataset": "LBNL_DataFDD",
            "building": building,
            "paths": paths
        }

    # Wang
    ROOT_WANG = Path("./data_FDD/8_nature_lcu_wang/raw/")
    wang_files = {
        "auditorium": ROOT_WANG / "auditorium_scientific_data.csv",
        "office":     ROOT_WANG / "office_scientific_data.csv",
        "hospital":   ROOT_WANG / "hosptial_scientific_data.csv",
    }

    for building, fp in wang_files.items():
        yield {
            "dataset": "Nature_LCU_Wang",
            "building": building,
            "paths": {"csv": fp}
        }


# ------------------------------------------------------------
# Result-row helper
# ------------------------------------------------------------
def add_rows(rows, dataset, building, label_mode, tag, fi_type, model_name, feat_names, mean_imp, std_imp, method):
    """
    Append feature-importance rows to the global output list.
    """
    if mean_imp is None:
        return

    if std_imp is None:
        std_imp = np.zeros_like(mean_imp, dtype=float)

    for f, mu, sd in zip(feat_names, mean_imp, std_imp):
        rows.append({
            "dataset": dataset,
            "building": building,
            "label_mode": label_mode,
            "tag": tag,
            "fi_type": fi_type,
            "model": model_name,
            "feature": str(f),
            "importance_mean": float(mu),
            "importance_std": float(sd),
            "method": method,
            "manipulation": "clean",
            "level": 0.0
        })


# ============================================================
# Main run
# ============================================================
log_header("Feature-importance run across all buildings")
log_kv("Device", DEVICE)
log_kv("Label mode", LABEL_MODE)
log_kv("Test size", TEST_SIZE)
log_kv("SEQ_LEN", SEQ_LEN)
log_kv("MAX_SENSOR_COLS", MAX_SENSOR_COLS)
log_kv("Tabular permutations", TAB_SAGE_PERMS)
log_kv("Sequence repeats", SEQ_REPEATS)

all_rows = []
t_global = time.time()

for item in iter_all_buildings():
    dataset = item["dataset"]
    building = item["building"]
    tag = f"{dataset}__{building}__{LABEL_MODE.upper()}"

    log_header(f"{dataset} | {building}")

    # --------------------------------------------------------
    # Read and preprocess one building
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

        if not fp.exists():
            log_kv("Skip", "missing file")
            continue

        df0 = pd.read_csv(fp)
        df, sensors, meta = preprocess_wang_building(
            df0,
            max_sensor_cols=MAX_SENSOR_COLS
        )

    log_kv("Load seconds", f"{time.time() - t0:.2f}")
    log_kv("Sensors", len(sensors))

    # --------------------------------------------------------
    # Select labels
    # --------------------------------------------------------
    if LABEL_MODE == "family":
        y_col = "FaultFamilyCode"
        inv_map = meta["inv_family_map"]
    else:
        y_col = "FaultCode"
        inv_map = meta["inv_fault_map"]

    X_all_raw = df[sensors].to_numpy().astype(np.float32)
    y_all = df[y_col].to_numpy().astype(int)

    vals, cnts = np.unique(y_all, return_counts=True)
    log_kv("X shape", X_all_raw.shape)
    log_kv("Classes", len(np.unique(y_all)))
    log_kv("Min class count", int(cnts.min()))
    log_kv("Singleton classes", int((cnts < 2).sum()))

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    t0 = time.time()
    Xtr_raw, Xte_raw, ytr_raw, yte_raw = stratified_split_keep_singletons_in_train(
        X_all_raw,
        y_all,
        test_size=TEST_SIZE,
        seed=42
    )
    log_kv("Split seconds", f"{time.time() - t0:.2f}")
    log_kv("Train size", Xtr_raw.shape)
    log_kv("Test size", Xte_raw.shape)

    if len(Xte_raw) < 10 or len(Xtr_raw) < 20:
        log_kv("Skip", "too few samples after split")
        continue

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_m = scaler.transform(Xtr_raw)
    Xte_m = scaler.transform(Xte_raw)

    # Optional cap for tabular model training
    Xtr_fit, ytr_fit = cap(Xtr_m, ytr_raw, MAX_TAB_TRAIN, seed=42)

    # --------------------------------------------------------
    # Build sequence datasets
    # --------------------------------------------------------
    Xtr_seq, ytr_seq = create_sequences(Xtr_m, ytr_raw, seq_len=SEQ_LEN, stride=SEQ_STRIDE)
    Xte_seq, yte_seq = create_sequences(Xte_m, yte_raw, seq_len=SEQ_LEN, stride=SEQ_STRIDE)

    Xtr_seq, ytr_seq = cap(Xtr_seq, ytr_seq, MAX_SEQ_TRAIN, seed=42)
    Xte_seq, yte_seq = cap(Xte_seq, yte_seq, MAX_SEQ_TEST, seed=43)

    # --------------------------------------------------------
    # Train all models once
    # --------------------------------------------------------
    t_train = time.time()

    n_classes = int(max(
        int(ytr_fit.max()) if len(ytr_fit) else 0,
        int(yte_raw.max()) if len(yte_raw) else 0,
        int(ytr_seq.max()) if len(ytr_seq) else 0,
        int(yte_seq.max()) if len(yte_seq) else 0
    ) + 1)
    log_kv("n_classes", n_classes)

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

    # Train tabular models
    for name, mdl in tab_models.items():
        tt = time.time()
        mdl.fit(Xtr_fit, ytr_fit)
        log_kv(f"Train {name} s", f"{time.time() - tt:.2f}")

    # Train XGBoost with remapped labels
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
            objective="multi:softprob",
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

    # Train sequence models
    seq_models = {}
    if len(Xtr_seq) >= 50 and len(Xte_seq) >= 50 and n_classes >= 2:
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

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------
    method_tab = f"sage_tab_mask={TAB_MASKING}_perms={TAB_SAGE_PERMS}_max={TAB_MAX_SAMPLES}"
    method_seq = f"perm_seq_repeats={SEQ_REPEATS}_max={SEQ_MAX_SAMPLES}"

    # Tabular FI
    log_sub("Feature importance: tabular")
    for name, mdl in tab_models.items():
        tfi = time.time()

        if name == "XGBoost":
            # XGBoost is trained on remapped class ids, so the test labels
            # must be filtered and re-encoded accordingly.
            mask = np.isin(yte_raw, le.classes_)
            Xte_xgb = Xte_m[mask]
            yte_xgb = le.transform(yte_raw[mask])

            pred_fn = lambda Z: coerce_pred_labels(mdl.predict(Z))
            mean_imp, std_imp = sage_importance_tabular(
                predict_fn=pred_fn,
                X_test=Xte_xgb,
                y_test=yte_xgb,
                X_train=Xtr_fit,
                metric_fn=metric_macro_f1,
                n_perms=TAB_SAGE_PERMS,
                seed=FI_SEED,
                max_samples=TAB_MAX_SAMPLES,
                masking=TAB_MASKING
            )

            add_rows(
                all_rows,
                dataset,
                building,
                LABEL_MODE,
                tag,
                "sage_tab_encoded",
                name,
                sensors,
                mean_imp,
                std_imp,
                method_tab
            )
        else:
            pred_fn = lambda Z, m=mdl: m.predict(Z)
            mean_imp, std_imp = sage_importance_tabular(
                predict_fn=pred_fn,
                X_test=Xte_m,
                y_test=yte_raw,
                X_train=Xtr_fit,
                metric_fn=metric_macro_f1,
                n_perms=TAB_SAGE_PERMS,
                seed=FI_SEED,
                max_samples=TAB_MAX_SAMPLES,
                masking=TAB_MASKING
            )

            add_rows(
                all_rows,
                dataset,
                building,
                LABEL_MODE,
                tag,
                "sage_tab",
                name,
                sensors,
                mean_imp,
                std_imp,
                method_tab
            )

        log_kv(f"FI {name} s", f"{time.time() - tfi:.2f}")

    # Sequence FI
    log_sub("Feature importance: sequence")
    if len(seq_models) == 0:
        log_kv("Seq FI", "skip")
    else:
        for name, mdl in seq_models.items():
            tfi = time.time()

            mean_imp, std_imp = permutation_importance_sequence(
                mdl,
                Xte_seq,
                yte_seq,
                metric_macro_f1,
                seed=FI_SEED,
                n_repeats=SEQ_REPEATS,
                max_samples=SEQ_MAX_SAMPLES,
                device=DEVICE,
                batch_size=SEQ_BATCH
            )

            add_rows(
                all_rows,
                dataset,
                building,
                LABEL_MODE,
                tag,
                "perm_seq",
                name,
                sensors,
                mean_imp,
                std_imp,
                method_seq
            )

            log_kv(f"FI {name} s", f"{time.time() - tfi:.2f}")


# ============================================================
# Save output
# ============================================================
log_header("Save output")

fi_df = pd.DataFrame(all_rows)
fi_df.to_csv(OUT_PATH, index=False)

log_kv("Saved", OUT_PATH)
log_kv("Rows", len(fi_df))
log_kv("Buildings", fi_df[["dataset", "building"]].drop_duplicates().shape[0] if len(fi_df) else 0)
log_kv("Models", sorted(fi_df["model"].unique()) if len(fi_df) else "none")
log_kv("Total runtime s", f"{time.time() - t_global:.2f}")

display(fi_df.head(20))