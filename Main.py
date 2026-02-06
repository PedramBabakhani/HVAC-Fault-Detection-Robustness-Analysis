# ============================================================
# 🚀 UNIFIED FAULT DETECTION PIPELINE (RAW-FIRST MANIPULATION + NO LEAKAGE + ROBUST XGBOOST)
#
# ✅ Correct order (no leakage + physically meaningful corruptions):
#   1) Build raw X_all, y_all
#   2) Split RAW into train/test
#   3) Fit scaler on CLEAN TRAIN RAW only  ✅
#   4) Apply data manipulation on RAW (train/test depending on CORRUPT_WHERE) ✅
#   5) Normalize manipulated raw via scaler.transform (never refit) ✅
#   6) Train models on (possibly manipulated) normalized train, evaluate on normalized test
#
# ✅ Fixes:
#   - Robust XGBoost: remap train labels per split to contiguous 0..K-1 to avoid "Invalid classes" crash
#   - Heatmap ticks aligned (cell centers)
#   - Predictions coerced to 1D label ints
#
# Outputs:
#   - ./fdd_out/results_all.csv
#   - ./fdd_out/feature_importance_all.csv
#   - ./fdd_out/confmat_all.csv
# ============================================================

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


# -----------------------------
# ✅ CONFIG (EDIT HERE)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("./fdd_out")
OUT.mkdir(exist_ok=True)

SPLIT_MODE = "stratified_time"   # "time" or "stratified" or "stratified_time"
TEST_SIZE  = 0.20
CORRUPT_WHERE = "test"           # "test" (recommended) or "both"
LABEL_MODE = "raw"               # "raw" or "family"

# Sequence config
SEQ_LEN = 10
SEQ_STRIDE = 1
EPOCHS = 15
BATCH  = 1024
LR     = 8e-4
GRAD_CLIP_NORM = 1.0

# Caps for speed
MAX_SEQ_TRAIN = 300000
MAX_SEQ_TEST  = 50000
MAX_TAB_TRAIN = 100000            # set None to disable

# Validation split inside TRAIN (optional)
USE_INTERNAL_VAL = True
VAL_FRACTION = 0.12

# Limit sensor columns (optional)
MAX_SENSOR_COLS = 20

# Plotting
SHOW_HEATMAP_INLINE = True
PLOT_ONLY_CLEAN_AND_WORST = True

# Feature importance (SAGE-style, Monte-Carlo)
COMPUTE_FEATURE_IMPORTANCE = True
SHOW_FI_INLINE = True
FI_TOPK = 25

FI_METRIC = "macro_f1"           # "macro_f1" or "accuracy"
FI_SEED = 123
FI_SAGE_N_PERMS = 20
FI_SAGE_MASKING = "sample"       # "sample" or "mean"
MAX_FI_TEST_SAMPLES_TAB = 5000
MAX_FI_TEST_SAMPLES_SEQ = 3000
MAX_FI_FEATURES = None

# ✅ XGBoost controls (fast + stable)
USE_XGBOOST = True
XGB_TREE_METHOD = "hist"         # "hist" stable; "gpu_hist" optional
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 5
XGB_LR = 0.08

# Include clean baseline + degradations
levels = {
    "clean":    [0],
    "noise":    [0.01, 0.05, 0.10],
    "drift":    [0.01, 0.05, 0.10],
    "bias":     [0.01, 0.05, 0.10],
    "missing":  [0.05, 0.10, 0.20],
    "sampling": [2, 4, 6],
}


# -----------------------------
# 🧾 Fancy logging
# -----------------------------
def log_header(title):
    print("\n" + "="*110)
    print(f"🔵 {title}")
    print("="*110)

def log_sub(title):
    print("\n" + "-"*90)
    print(f"🔸 {title}")
    print("-"*90)

def log_kv(k, v):
    print(f"   • {k}: {v}")


# -----------------------------
# ✅ ALWAYS COERCE PREDICTIONS TO 1D LABELS
# -----------------------------
def coerce_pred_labels(y_pred):
    """
    Accept:
      - (n,) labels
      - (n, C) probabilities/scores
      - (n, 1)
    Return:
      - (n,) int labels
    """
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    y_pred = np.ravel(y_pred)

    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = y_pred.astype(int)

    return y_pred

def log_metrics_block(model_name, manip_name, lvl, y_true, y_pred):
    y_pred = coerce_pred_labels(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("\n" + "."*90)
    print(f"✅ EXPERIMENT | Model={model_name} | manip={manip_name} | level={lvl}")
    print("."*90)
    log_kv("Accuracy", acc)
    log_kv("Macro Precision", p)
    log_kv("Macro Recall", r)
    log_kv("Macro F1", f1m)
    return acc, p, r, f1m

def print_class_balance(y, inv_map, title, topk=999):
    vals, cnts = np.unique(y, return_counts=True)
    pairs = sorted([(int(v), int(c)) for v, c in zip(vals, cnts)], key=lambda x: -x[1])
    print(f"   • {title} class counts:")
    for v, c in pairs[:topk]:
        name = inv_map.get(v, str(v))
        print(f"      - {v:>3} | {c:>8} | {name}")


# -----------------------------
# Helpers
# -----------------------------
def sensor_cols_generic(df, exclude):
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

def sanitize_sensors(df, sensors):
    banned_exact = {
        "FaultCode", "FaultName", "labeling",
        "FaultFamily", "FaultFamilyCode",
        "Fault Detection Ground Truth", "FaultDetectionGroundTruth",
        "Datetime", "timestamp", "Time", "DATE", "AHU name"
    }
    clean = []
    for c in sensors:
        cl = str(c).lower()
        if c in banned_exact:
            continue
        if ("fault" in cl) or ("label" in cl) or ("ground truth" in cl) or ("truth" in cl) or ("code" in cl):
            continue
        clean.append(c)
    return clean

def assert_no_leakage(sensors):
    bad = [s for s in sensors if ("fault" in s.lower()) or ("label" in s.lower()) or ("truth" in s.lower()) or ("code" in s.lower())]
    assert len(bad) == 0, f"🚨 Leakage columns detected in sensors: {bad}"

def create_sequences(X, y, seq_len=24, stride=1):
    Xs, ys = [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    if len(Xs) == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=int)
    return np.stack(Xs), np.array(ys, dtype=int)

def cap_sequences(Xseq, yseq, max_n, seed=42):
    if max_n is None or len(Xseq) <= max_n:
        return Xseq, yseq
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xseq), size=max_n, replace=False)
    return Xseq[idx], yseq[idx]

def cap_tabular(X, y, max_n, seed=42):
    if max_n is None or len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]

def should_plot(manip_name, lvl):
    if not PLOT_ONLY_CLEAN_AND_WORST:
        return True
    if manip_name == "clean":
        return True
    if manip_name in ["noise","drift","bias","missing"]:
        return lvl == max(levels[manip_name])
    if manip_name == "sampling":
        return lvl == max(levels["sampling"])
    return False


# -----------------------------
# ✅ Heatmap tick alignment (force ticks to cell centers)
# -----------------------------
def show_confmat(cm, class_names, title, show_inline=True):
    n = len(class_names)
    assert cm.shape == (n, n), f"cm shape {cm.shape} != ({n},{n})"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True, fmt=".2f",
        cmap="Blues",
        ax=ax,
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False
    )

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names, rotation=0)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    fig.tight_layout()
    if show_inline:
        plt.show()
    plt.close(fig)

def show_feature_importance_signed(feat_names, mean_imp, std_imp, title, topk=25,
                                   show_inline=True, rank_by="abs"):
    if mean_imp is None or feat_names is None or len(feat_names) == 0:
        return
    if std_imp is None:
        std_imp = np.zeros_like(mean_imp, dtype=float)

    df = pd.DataFrame({"feature": feat_names, "mean": mean_imp, "std": std_imp})

    if rank_by == "abs":
        df["rank_key"] = df["mean"].abs()
        df = df.sort_values("rank_key", ascending=False)
    else:
        df = df.sort_values("mean", ascending=False)

    top = df.head(topk).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["mean"], xerr=top["std"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(f"SAGE-style Shapley global importance (mean ± std) on {FI_METRIC}")
    fig.tight_layout()
    if show_inline:
        plt.show()
    plt.close(fig)


def split_train_val(Xtr, ytr, mode="time", val_fraction=0.12, seed=42):
    if (not USE_INTERNAL_VAL) or (val_fraction <= 0.0) or (len(Xtr) < 200):
        return Xtr, ytr, None, None

    n = len(Xtr)
    n_val = int(n * val_fraction)
    if n_val < 50:
        return Xtr, ytr, None, None

    if mode in ["time", "stratified_time"]:
        X_train = Xtr[:-n_val]
        y_train = ytr[:-n_val]
        X_val = Xtr[-n_val:]
        y_val = ytr[-n_val:]
        return X_train, y_train, X_val, y_val

    X_train, X_val, y_train, y_val = train_test_split(
        Xtr, ytr, test_size=val_fraction, stratify=ytr, shuffle=True, random_state=seed
    )
    return X_train, y_train, X_val, y_val


def compute_balanced_weights(y, n_classes, default=1.0):
    y = np.asarray(y, dtype=int)
    present = np.unique(y)
    full = np.full((n_classes,), float(default), dtype=np.float32)
    cw_present = compute_class_weight(class_weight="balanced", classes=present, y=y)
    for c, w in zip(present, cw_present):
        if 0 <= int(c) < n_classes:
            full[int(c)] = float(w)
    return full

def compute_sample_weights(y, class_weights):
    y = np.asarray(y, dtype=int)
    return np.asarray([float(class_weights[int(yi)]) for yi in y], dtype=np.float32)


def stratified_time_split(X, y, test_size=0.2, min_train_per_class=10, min_test_per_class=5):
    y = np.asarray(y, dtype=int)
    idx_all = np.arange(len(y))

    train_idx = []
    test_idx = []

    for c in np.unique(y):
        idx_c = idx_all[y == c]
        m = len(idx_c)
        n_test = int(np.floor(m * test_size))
        n_test = max(n_test, min_test_per_class)

        if m - n_test < min_train_per_class:
            train_idx.extend(idx_c.tolist())
            continue

        split_point = m - n_test
        train_idx.extend(idx_c[:split_point].tolist())
        test_idx.extend(idx_c[split_point:].tolist())

    train_idx = np.array(train_idx, dtype=int)
    test_idx  = np.array(test_idx, dtype=int)
    train_idx.sort()
    test_idx.sort()

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# -----------------------------
# ✅ Aggregators
# -----------------------------
FEATURE_IMPORTANCE_ROWS = []
CONFMAT_ROWS = []

def add_confmat_rows(src, building, label_mode, tag_base, model_name, manip_name, lvl,
                     class_names, cm_raw, cm_norm):
    for norm_flag, cm in [(0, cm_raw), (1, cm_norm)]:
        for i, tname in enumerate(class_names):
            for j, pname in enumerate(class_names):
                CONFMAT_ROWS.append({
                    "source_dataset": src,
                    "building": building,
                    "label_mode": label_mode,
                    "tag": tag_base,
                    "model": model_name,
                    "manipulation": manip_name,
                    "level": lvl,
                    "normalized": norm_flag,
                    "true_label": tname,
                    "pred_label": pname,
                    "value": float(cm[i, j]),
                })

def add_feature_importance_rows(src, building, label_mode, tag_base, model_name, manip_name, lvl,
                               feat_names, mean_imp, std_imp, method):
    if mean_imp is None or feat_names is None:
        return
    if std_imp is None:
        std_imp = np.zeros_like(mean_imp, dtype=float)

    for f, mu, sd in zip(feat_names, mean_imp, std_imp):
        FEATURE_IMPORTANCE_ROWS.append({
            "source_dataset": src,
            "building": building,
            "label_mode": label_mode,
            "tag": tag_base,
            "model": model_name,
            "manipulation": manip_name,
            "level": lvl,
            "feature": str(f),
            "importance_mean": float(mu),
            "importance_std": float(sd),
            "method": method
        })


# -----------------------------
# ✅ Fault family mapping
# -----------------------------
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
    fams = [map_fault_to_family(x) for x in labels]
    ordered = [f for f in FAULT_FAMILY_ORDER if f in set(fams)]
    for f in sorted(set(fams)):
        if f not in ordered:
            ordered.append(f)
    fam2id = {f:i for i,f in enumerate(ordered)}
    id2fam = {i:f for f,i in fam2id.items()}
    return fam2id, id2fam


# -----------------------------
# Degradations (operate on RAW sensors!)
# -----------------------------
def degrade_noise(X, lvl):
    # noise proportional to each feature's std in the provided X
    return X + np.random.normal(0, np.std(X, axis=0, keepdims=True) * lvl, size=X.shape)

def degrade_drift(X, lvl):
    t = np.linspace(0, 1, X.shape[0]).reshape(-1, 1)
    return X + (np.mean(X, axis=0, keepdims=True) * lvl) * t

def degrade_bias(X, lvl):
    return X + (np.mean(X, axis=0, keepdims=True) * lvl)

def degrade_missing(X, lvl):
    Xm = X.copy()
    mask = np.random.rand(*Xm.shape) < lvl
    Xm[mask] = np.nan
    Xm = pd.DataFrame(Xm).ffill().bfill().to_numpy()
    return Xm

def degrade_sampling(X, y, step):
    return X[::step], y[::step]

manips = {
    "noise": degrade_noise,
    "drift": degrade_drift,
    "bias": degrade_bias,
    "missing": degrade_missing,
    "sampling": degrade_sampling,
}


# -----------------------------
# ✅ GPU models
# -----------------------------
class TinyLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TinyCNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, channels=32, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(channels, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = x.transpose(2, 1)
        x = torch.relu(self.cnn(x))
        x = x.transpose(2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TinyInformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.in_proj(x)
        z = self.encoder(x)
        return self.cls(z[:, -1, :])

def train_torch_classifier(model, Xtr, ytr, Xte,
                           epochs=10, batch_size=512, lr=5e-4,
                           class_weights=None, grad_clip_norm=1.0):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr)

    if class_weights is None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        w = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=w)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long()),
        batch_size=batch_size, shuffle=True,
        pin_memory=True
    )

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch_size):
            xb = torch.tensor(Xte[i:i+batch_size]).float().to(DEVICE, non_blocking=True)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)

@torch.no_grad()
def torch_predict_labels(model, X, batch_size=512):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size]).float().to(DEVICE, non_blocking=True)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# -----------------------------
# ✅ SAGE-style Shapley Global Importance
# -----------------------------
def metric_macro_f1(y_true, y_pred):
    y_pred = coerce_pred_labels(y_pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def metric_accuracy(y_true, y_pred):
    y_pred = coerce_pred_labels(y_pred)
    return accuracy_score(y_true, y_pred)

def get_fi_metric_fn():
    return metric_accuracy if FI_METRIC == "accuracy" else metric_macro_f1

def _subsample_xy_tabular(X, y, max_n, rng):
    if max_n is None or len(X) <= max_n:
        return X, y
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]

def _subsample_xy_seq(Xseq, yseq, max_n, rng):
    if max_n is None or len(Xseq) <= max_n:
        return Xseq, yseq
    idx = rng.choice(len(Xseq), size=max_n, replace=False)
    return Xseq[idx], yseq[idx]

def _mask_all_features_tabular(X, X_train, feat_idx, rng, masking="sample"):
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
        raise ValueError("FI_SAGE_MASKING must be one of: 'sample', 'mean'")
    return Xm

def _mask_all_features_sequence(Xseq, X_train_tab, feat_idx, rng, masking="sample"):
    Xm = Xseq.copy()
    if masking == "mean":
        col_mean = X_train_tab.mean(axis=0)
        for j in feat_idx:
            Xm[:, :, j] = col_mean[j]
    elif masking == "sample":
        for j in feat_idx:
            src = X_train_tab[:, j]
            repl = rng.choice(src, size=Xm.shape[0], replace=True)
            Xm[:, :, j] = repl[:, None]
    else:
        raise ValueError("FI_SAGE_MASKING must be one of: 'sample', 'mean'")
    return Xm

def sage_importance_tabular(predict_fn, X_test, y_test, X_train, metric_fn,
                            n_perms=20, seed=123, max_samples=5000, max_features=None,
                            masking="sample"):
    rng = np.random.default_rng(seed)
    Xs, ys = _subsample_xy_tabular(X_test, y_test, max_samples, rng)

    F = Xs.shape[1]
    feat_idx = np.arange(F)
    if max_features is not None:
        feat_idx = feat_idx[:min(max_features, F)]

    contrib = np.zeros((n_perms, len(feat_idx)), dtype=float)

    for p in range(n_perms):
        order = rng.permutation(len(feat_idx))
        Xmasked = _mask_all_features_tabular(Xs, X_train, feat_idx, rng, masking=masking)
        pred0 = coerce_pred_labels(predict_fn(Xmasked))
        prev_score = metric_fn(ys, pred0)

        Xcurr = Xmasked
        for step_pos in order:
            j = feat_idx[step_pos]
            k = np.where(feat_idx == j)[0][0]
            Xcurr[:, j] = Xs[:, j]
            pred1 = coerce_pred_labels(predict_fn(Xcurr))
            score1 = metric_fn(ys, pred1)
            contrib[p, k] = score1 - prev_score
            prev_score = score1

    return contrib.mean(axis=0), contrib.std(axis=0)

def sage_importance_sequence(predict_fn_seq, Xseq_test, yseq_test, X_train_tab, metric_fn,
                             n_perms=20, seed=123, max_samples=3000, max_features=None,
                             masking="sample"):
    rng = np.random.default_rng(seed)
    Xs, ys = _subsample_xy_seq(Xseq_test, yseq_test, max_samples, rng)

    if len(Xs) == 0:
        return np.array([], dtype=int), None, None

    F = Xs.shape[2]
    feat_idx = np.arange(F)
    if max_features is not None:
        feat_idx = feat_idx[:min(max_features, F)]

    contrib = np.zeros((n_perms, len(feat_idx)), dtype=float)

    for p in range(n_perms):
        order = rng.permutation(len(feat_idx))
        Xmasked = _mask_all_features_sequence(Xs, X_train_tab, feat_idx, rng, masking=masking)
        pred0 = coerce_pred_labels(predict_fn_seq(Xmasked))
        prev_score = metric_fn(ys, pred0)

        Xcurr = Xmasked
        for step_pos in order:
            j = feat_idx[step_pos]
            k = np.where(feat_idx == j)[0][0]
            Xcurr[:, :, j] = Xs[:, :, j]
            pred1 = coerce_pred_labels(predict_fn_seq(Xcurr))
            score1 = metric_fn(ys, pred1)
            contrib[p, k] = score1 - prev_score
            prev_score = score1

    return feat_idx, contrib.mean(axis=0), contrib.std(axis=0)

def make_torch_predict_fn(model):
    return lambda X: coerce_pred_labels(torch_predict_labels(model, X, batch_size=BATCH))


# ============================================================
# DATASET A) LBNL DataFDD synthesis inventory
# ============================================================
def clean_fault_name(s):
    s = str(s).strip()
    if "(" in s:
        return s.split("(")[0].strip()
    return s

def merge_lbnl_sensor_fault(sensor_df, fault_df):
    df = sensor_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df["FaultName"] = "Normal"

    fault_df = fault_df.rename(columns=lambda x: x.strip().lower())
    fault_col = next(c for c in fault_df.columns if "fault" in c)
    time_col  = next(c for c in fault_df.columns if "time" in c)

    for _, row in fault_df.iterrows():
        fname = clean_fault_name(row[fault_col])
        t_raw = str(row[time_col])
        t = (t_raw.replace("TO", "to").replace("To", "to")
                    .replace(" - ", " to ").replace("—", " to ").replace("-", " to "))
        parts = [p.strip() for p in t.split("to")]

        if len(parts) == 2:
            start = pd.to_datetime(parts[0], errors="coerce")
            end   = pd.to_datetime(parts[1], errors="coerce")
        else:
            start = pd.to_datetime(parts[0], errors="coerce")
            end   = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

        if pd.isna(start) or pd.isna(end):
            continue

        mask = (df["Datetime"] >= start) & (df["Datetime"] <= end)
        df.loc[mask, "FaultName"] = fname

    uniq = sorted(df["FaultName"].unique())
    fmap = {f:i for i,f in enumerate(uniq)}
    inv  = {i:f for f,i in fmap.items()}
    df["FaultCode"] = df["FaultName"].map(fmap).astype(int)

    fam2id, id2fam = build_family_codec_from_labels(uniq)
    df["FaultFamily"] = df["FaultName"].map(map_fault_to_family)
    df["FaultFamilyCode"] = df["FaultFamily"].map(fam2id).astype(int)

    sensors = sensor_cols_generic(df, exclude=["Datetime","FaultName","FaultCode","FaultFamily","FaultFamilyCode"])
    sensors = sanitize_sensors(df, sensors)
    assert_no_leakage(sensors)

    sensors = sensors[:min(MAX_SENSOR_COLS, len(sensors))]

    df[sensors] = df[sensors].ffill().bfill()
    meta = {"inv_fault_map": inv, "inv_family_map": id2fam}
    return df, sensors, meta

def load_dataset_lbnl():
    RAW  = Path("/kaggle/input/datafdd/5_lbnl_data_synthesis_inventory/raw/")
    file_map = {
        "RTU":       {"sensor": RAW/"RTU.csv",       "faults": RAW/"RTU-faults.csv"},
        "SZCAV":     {"sensor": RAW/"SZCAV.csv",     "faults": RAW/"SZCAV-faults.csv"},
        "SZVAV":     {"sensor": RAW/"SZVAV.csv",     "faults": RAW/"SZVAV-faults.csv"},
        "MZVAV_1":   {"sensor": RAW/"MZVAV-1.csv",   "faults": RAW/"MZVAV-1-faults.csv"},
        "MZVAV_2_1": {"sensor": RAW/"MZVAV-2-1.csv", "faults": RAW/"MZVAV-2-1-faults.csv"},
        "MZVAV_2_2": {"sensor": RAW/"MZVAV-2-2.csv", "faults": RAW/"MZVAV-2-2-faults.csv"},
    }

    items = []
    log_header("Loading DATASET A: LBNL DataFDD synthesis inventory")
    for b, p in file_map.items():
        if not p["sensor"].exists() or not p["faults"].exists():
            log_kv(f"Skip {b}", "missing files")
            continue

        s = pd.read_csv(p["sensor"])
        f = pd.read_csv(p["faults"])
        df, sensors, meta = merge_lbnl_sensor_fault(s, f)

        items.append({
            "source_dataset": "LBNL_DataFDD",
            "building": b,
            "df": df,
            "sensors": sensors,
            "inv_fault_map": meta["inv_fault_map"],
            "inv_family_map": meta["inv_family_map"],
        })

        log_sub(b)
        log_kv("Rows", df.shape[0])
        log_kv("Sensors", len(sensors))
        log_kv("Sensor columns", sensors)
        log_kv("Faults(raw)", list(meta["inv_fault_map"].values())[:10] + (["..."] if len(meta["inv_fault_map"]) > 10 else []))
        log_kv("FaultFamilies", list(meta["inv_family_map"].values()))
    return items


# ============================================================
# DATASET B) Nature LCU Wang
# ============================================================
def preprocess_wang_building(df):
    df = df.copy()

    if "DATE" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce", dayfirst=True
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

    exclude = ["timestamp","Time","DATE","AHU name","labeling","FaultCode","FaultFamily","FaultFamilyCode"]
    sensors = sensor_cols_generic(df, exclude=exclude)
    sensors = sanitize_sensors(df, sensors)
    assert_no_leakage(sensors)

    sensors = sensors[:min(MAX_SENSOR_COLS, len(sensors))]

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df[sensors] = df[sensors].ffill().bfill()
    meta = {"inv_fault_map": inv_map, "inv_family_map": id2fam}
    return df, sensors, meta

def load_dataset_wang():
    ROOT = Path("/kaggle/input/datafdd/8_nature_lcu_wang/raw/")
    log_header("Loading DATASET B: Nature LCU Wang")

    datasets = {
        "auditorium": pd.read_csv(ROOT/"auditorium_scientific_data.csv"),
        "office":     pd.read_csv(ROOT/"office_scientific_data.csv"),
        "hospital":   pd.read_csv(ROOT/"hosptial_scientific_data.csv"),
    }

    items = []
    for b, df0 in datasets.items():
        df, sensors, meta = preprocess_wang_building(df0)

        items.append({
            "source_dataset": "Nature_LCU_Wang",
            "building": b,
            "df": df,
            "sensors": sensors,
            "inv_fault_map": meta["inv_fault_map"],
            "inv_family_map": meta["inv_family_map"],
        })

        log_sub(b)
        log_kv("Rows", df.shape[0])
        log_kv("Sensors", len(sensors))
        log_kv("Sensor columns", sensors)
        log_kv("Faults(raw)", list(meta["inv_fault_map"].values()))
        log_kv("FaultFamilies", list(meta["inv_family_map"].values()))
    return items


# ============================================================
# RUNNER (per building)
# ============================================================
def run_one_building(item):
    src = item["source_dataset"]
    building = item["building"]
    df = item["df"]
    sensors = item["sensors"]

    if LABEL_MODE == "family":
        inv_map = item["inv_family_map"]
        y_col = "FaultFamilyCode"
        tag_base = f"{src}__{building}__FAMILY"
    else:
        inv_map = item["inv_fault_map"]
        y_col = "FaultCode"
        tag_base = f"{src}__{building}__RAW"

    class_names = [inv_map[i] for i in range(len(inv_map))]
    n_classes = len(class_names)

    X_all_raw = df[sensors].to_numpy().astype(np.float32)   # ✅ RAW sensors
    y_all = df[y_col].to_numpy().astype(int)

    # ✅ Split RAW before any scaling (no leakage)
    if SPLIT_MODE == "time":
        split = int(len(X_all_raw) * (1 - TEST_SIZE))
        Xtr_raw, Xte_raw = X_all_raw[:split], X_all_raw[split:]
        ytr_raw, yte_raw = y_all[:split], y_all[split:]

    elif SPLIT_MODE == "stratified":
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
            X_all_raw, y_all, test_size=TEST_SIZE, stratify=y_all, shuffle=True, random_state=42
        )

    elif SPLIT_MODE == "stratified_time":
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = stratified_time_split(
            X_all_raw, y_all, test_size=TEST_SIZE, min_train_per_class=10, min_test_per_class=5
        )
    else:
        raise ValueError("SPLIT_MODE must be one of: 'time', 'stratified', 'stratified_time'")

    log_sub(f"{tag_base} | Split")
    log_kv("Train RAW", Xtr_raw.shape)
    log_kv("Test  RAW", Xte_raw.shape)
    log_kv("n_classes", n_classes)

    print_class_balance(ytr_raw, inv_map, "TRAIN")
    print_class_balance(yte_raw, inv_map, "TEST")

    train_classes = set(np.unique(ytr_raw).tolist())
    test_classes  = set(np.unique(yte_raw).tolist())
    missing_in_train = sorted(list(test_classes - train_classes))
    if len(missing_in_train) > 0:
        log_kv("⚠️ Classes in TEST but missing in TRAIN", [inv_map[i] for i in missing_in_train])

    # ✅ Fit scaler ONLY on CLEAN TRAIN RAW (never on test, never on corrupted)
    scaler = StandardScaler()
    scaler.fit(Xtr_raw)

    rows = []
    metric_fn = get_fi_metric_fn()
    fi_method = f"sage_mc_{FI_SAGE_MASKING}_{FI_METRIC}_mean_std__perms={FI_SAGE_N_PERMS}"

    for manip_name, lvls in levels.items():
        for lvl in lvls:
            log_header(f"{tag_base} | manip={manip_name} | level={lvl}")

            # -----------------------------
            # ✅ RAW-FIRST MANIPULATION
            # -----------------------------
            Xtr_raw_m = Xtr_raw.copy(); ytr_m = ytr_raw.copy()
            Xte_raw_m = Xte_raw.copy(); yte_m = yte_raw.copy()

            if manip_name == "clean":
                pass

            elif manip_name == "sampling":
                if CORRUPT_WHERE == "both":
                    Xtr_raw_m, ytr_m = degrade_sampling(Xtr_raw_m, ytr_m, lvl)
                Xte_raw_m, yte_m = degrade_sampling(Xte_raw_m, yte_m, lvl)

            else:
                if CORRUPT_WHERE == "both":
                    Xtr_raw_m = manips[manip_name](Xtr_raw_m, lvl)
                Xte_raw_m = manips[manip_name](Xte_raw_m, lvl)

            # Ensure any NaNs are resolved (esp. after missing)
            Xtr_raw_m = pd.DataFrame(Xtr_raw_m).ffill().bfill().to_numpy()
            Xte_raw_m = pd.DataFrame(Xte_raw_m).ffill().bfill().to_numpy()

            # ✅ Normalize using scaler fitted on clean train raw
            Xtr_m = scaler.transform(Xtr_raw_m)
            Xte_m = scaler.transform(Xte_raw_m)

            log_kv("Train after corrupt (RAW)", Xtr_raw_m.shape)
            log_kv("Test  after corrupt (RAW)", Xte_raw_m.shape)
            log_kv("Train after scale", Xtr_m.shape)
            log_kv("Test  after scale", Xte_m.shape)

            # ---------------- TABULAR ----------------
            Xtr_tab, ytr_tab = cap_tabular(Xtr_m, ytr_m, MAX_TAB_TRAIN, seed=42)
            if len(Xtr_tab) != len(Xtr_m):
                log_kv("Tabular cap", f"{len(Xtr_m)} -> {len(Xtr_tab)}")

            Xtr_fit, ytr_fit, Xval_fit, yval_fit = split_train_val(
                Xtr_tab, ytr_tab, mode=SPLIT_MODE, val_fraction=VAL_FRACTION, seed=42
            )

            cw_fit = compute_balanced_weights(ytr_fit, n_classes)
            sw_fit = compute_sample_weights(ytr_fit, cw_fit)

            # ✅ LinearSVM
            t0 = time.time()
            svm = LinearSVC(C=1.0, dual=False, max_iter=8000, class_weight="balanced")
            svm.fit(Xtr_fit, ytr_fit)
            pred = coerce_pred_labels(svm.predict(Xte_m))
            rt = time.time() - t0

            acc, mp, mr, mf1 = log_metrics_block("LinearSVM", manip_name, lvl, yte_m, pred)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
            add_confmat_rows(src, building, LABEL_MODE, tag_base, "LinearSVM", manip_name, lvl, class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(cm_norm, class_names, title=f"{tag_base} | LinearSVM | {manip_name}={lvl}", show_inline=True)

            if COMPUTE_FEATURE_IMPORTANCE:
                mean_imp, std_imp = sage_importance_tabular(
                    predict_fn=lambda Z: coerce_pred_labels(svm.predict(Z)),
                    X_test=Xte_m, y_test=yte_m,
                    X_train=Xtr_fit,
                    metric_fn=metric_fn,
                    n_perms=FI_SAGE_N_PERMS,
                    seed=FI_SEED,
                    max_samples=MAX_FI_TEST_SAMPLES_TAB,
                    max_features=MAX_FI_FEATURES,
                    masking=FI_SAGE_MASKING
                )
                used_names = sensors[:min(MAX_FI_FEATURES, len(sensors))] if MAX_FI_FEATURES is not None else sensors
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "LinearSVM", manip_name, lvl, used_names, mean_imp, std_imp, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        used_names, mean_imp, std_imp,
                        title=f"{tag_base} | LinearSVM | FI ({manip_name}={lvl})",
                        topk=FI_TOPK, show_inline=True
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl, "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "LinearSVM", "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
                "runtime_s": rt
            })

            # ✅ RF
            t0 = time.time()
            rf = RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced_subsample"
            )
            rf.fit(Xtr_fit, ytr_fit)
            pred = coerce_pred_labels(rf.predict(Xte_m))
            rt = time.time() - t0

            acc, mp, mr, mf1 = log_metrics_block("RF", manip_name, lvl, yte_m, pred)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
            add_confmat_rows(src, building, LABEL_MODE, tag_base, "RF", manip_name, lvl, class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(cm_norm, class_names, title=f"{tag_base} | RF | {manip_name}={lvl}", show_inline=True)

            if COMPUTE_FEATURE_IMPORTANCE:
                mean_imp, std_imp = sage_importance_tabular(
                    predict_fn=lambda Z: coerce_pred_labels(rf.predict(Z)),
                    X_test=Xte_m, y_test=yte_m,
                    X_train=Xtr_fit,
                    metric_fn=metric_fn,
                    n_perms=FI_SAGE_N_PERMS,
                    seed=FI_SEED,
                    max_samples=MAX_FI_TEST_SAMPLES_TAB,
                    max_features=MAX_FI_FEATURES,
                    masking=FI_SAGE_MASKING
                )
                used_names = sensors[:min(MAX_FI_FEATURES, len(sensors))] if MAX_FI_FEATURES is not None else sensors
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "RF", manip_name, lvl, used_names, mean_imp, std_imp, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        used_names, mean_imp, std_imp,
                        title=f"{tag_base} | RF | FI ({manip_name}={lvl})",
                        topk=FI_TOPK, show_inline=True
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl, "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "RF", "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
                "runtime_s": rt
            })

            # ✅ XGBoost (ROBUST LABEL REMAP PER TRAIN SPLIT)
            if USE_XGBOOST:
                t0 = time.time()

                tree_method = "hist"
                if XGB_TREE_METHOD == "gpu_hist" and torch.cuda.is_available():
                    tree_method = "gpu_hist"

                le = LabelEncoder()
                ytr_xgb = le.fit_transform(ytr_fit)  # contiguous 0..K-1
                K = len(le.classes_)

                if K < 2:
                    log_kv("XGBoost", f"skip (only {K} class in train after split/cap)")
                else:
                    xgb_model = XGBClassifier(
                        tree_method=tree_method,
                        max_depth=int(XGB_MAX_DEPTH),
                        n_estimators=int(XGB_N_ESTIMATORS),
                        learning_rate=float(XGB_LR),
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=1.0,
                        min_child_weight=1.0,
                        random_state=42,
                        objective="multi:softprob",
                        num_class=K,
                        eval_metric="mlogloss",
                        n_jobs=-1,
                        max_bin=256
                    )

                    xgb_model.fit(Xtr_fit, ytr_xgb, sample_weight=sw_fit, verbose=False)

                    pred_local = coerce_pred_labels(xgb_model.predict(Xte_m))
                    pred = le.inverse_transform(pred_local)  # back to original label IDs

                    rt = time.time() - t0
                    acc, mp, mr, mf1 = log_metrics_block("XGBoost", manip_name, lvl, yte_m, pred)
                    log_kv("Runtime (s)", rt)

                    cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
                    cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
                    add_confmat_rows(src, building, LABEL_MODE, tag_base, "XGBoost", manip_name, lvl, class_names, cm_raw, cm_norm)

                    if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                        show_confmat(cm_norm, class_names, title=f"{tag_base} | XGBoost | {manip_name}={lvl}", show_inline=True)

                    if COMPUTE_FEATURE_IMPORTANCE:
                        # keep predict_fn returning original-label IDs
                        mean_imp, std_imp = sage_importance_tabular(
                            predict_fn=lambda Z: le.inverse_transform(coerce_pred_labels(xgb_model.predict(Z))),
                            X_test=Xte_m, y_test=yte_m,
                            X_train=Xtr_fit,
                            metric_fn=metric_fn,
                            n_perms=FI_SAGE_N_PERMS,
                            seed=FI_SEED,
                            max_samples=MAX_FI_TEST_SAMPLES_TAB,
                            max_features=MAX_FI_FEATURES,
                            masking=FI_SAGE_MASKING
                        )
                        used_names = sensors[:min(MAX_FI_FEATURES, len(sensors))] if MAX_FI_FEATURES is not None else sensors
                        add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "XGBoost", manip_name, lvl, used_names, mean_imp, std_imp, method=fi_method)

                        if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                            show_feature_importance_signed(
                                used_names, mean_imp, std_imp,
                                title=f"{tag_base} | XGBoost | FI ({manip_name}={lvl})",
                                topk=FI_TOPK, show_inline=True
                            )

                    rows.append({
                        "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                        "manipulation": manip_name, "level": lvl, "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                        "model": "XGBoost", "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
                        "runtime_s": rt
                    })

            # ---------------- SEQUENCE ----------------
            Xtr_seq, ytr_seq = create_sequences(Xtr_m, ytr_m, seq_len=SEQ_LEN, stride=SEQ_STRIDE)
            Xte_seq, yte_seq = create_sequences(Xte_m, yte_m, seq_len=SEQ_LEN, stride=SEQ_STRIDE)

            log_sub("SEQUENCE | Build sequences")
            log_kv("Train seq (raw)", Xtr_seq.shape)
            log_kv("Test seq (raw)", Xte_seq.shape)

            if len(Xtr_seq) < 50 or len(Xte_seq) < 50:
                log_kv("Skipping seq models", f"too few sequences (train={len(Xtr_seq)}, test={len(Xte_seq)})")
                continue

            Xtr_seq, ytr_seq = cap_sequences(Xtr_seq, ytr_seq, MAX_SEQ_TRAIN, seed=42)
            Xte_seq, yte_seq = cap_sequences(Xte_seq, yte_seq, MAX_SEQ_TEST, seed=43)

            log_sub("SEQUENCE | Sequences (capped)")
            log_kv("Train seq", Xtr_seq.shape)
            log_kv("Test seq", Xte_seq.shape)

            cw_seq = compute_balanced_weights(ytr_seq, n_classes)

            # LSTM
            t0 = time.time()
            m = TinyLSTM(input_dim=Xtr_seq.shape[2], num_classes=n_classes)
            pred = train_torch_classifier(
                m, Xtr_seq, ytr_seq, Xte_seq,
                epochs=EPOCHS, batch_size=BATCH, lr=LR,
                class_weights=cw_seq, grad_clip_norm=GRAD_CLIP_NORM
            )
            pred = coerce_pred_labels(pred)
            rt = time.time() - t0

            acc, mp, mr, mf1 = log_metrics_block("LSTM", manip_name, lvl, yte_seq, pred)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
            add_confmat_rows(src, building, LABEL_MODE, tag_base, "LSTM", manip_name, lvl, class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(cm_norm, class_names, title=f"{tag_base} | LSTM | {manip_name}={lvl}", show_inline=True)

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl, "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "LSTM", "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
                "runtime_s": rt
            })

            # CNN-LSTM
            t0 = time.time()
            m = TinyCNNLSTM(input_dim=Xtr_seq.shape[2], num_classes=n_classes)
            pred = train_torch_classifier(
                m, Xtr_seq, ytr_seq, Xte_seq,
                epochs=EPOCHS, batch_size=BATCH, lr=LR,
                class_weights=cw_seq, grad_clip_norm=GRAD_CLIP_NORM
            )
            pred = coerce_pred_labels(pred)
            rt = time.time() - t0

            acc, mp, mr, mf1 = log_metrics_block("CNN-LSTM", manip_name, lvl, yte_seq, pred)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
            add_confmat_rows(src, building, LABEL_MODE, tag_base, "CNN-LSTM", manip_name, lvl, class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(cm_norm, class_names, title=f"{tag_base} | CNN-LSTM | {manip_name}={lvl}", show_inline=True)

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl, "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "CNN-LSTM", "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
                "runtime_s": rt
            })

            # Informer
            t0 = time.time()
            m = TinyInformerClassifier(input_dim=Xtr_seq.shape[2], num_classes=n_classes)
            pred = train_torch_classifier(
                m, Xtr_seq, ytr_seq, Xte_seq,
                epochs=EPOCHS, batch_size=BATCH, lr=LR,
                class_weights=cw_seq, grad_clip_norm=GRAD_CLIP_NORM
            )
            pred = coerce_pred_labels(pred)
            rt = time.time() - t0

            acc, mp, mr, mf1 = log_metrics_block("Informer", manip_name, lvl, yte_seq, pred)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
            add_confmat_rows(src, building, LABEL_MODE, tag_base, "Informer", manip_name, lvl, class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(cm_norm, class_names, title=f"{tag_base} | Informer | {manip_name}={lvl}", show_inline=True)

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl, "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "Informer", "accuracy": acc, "macro_precision": mp, "macro_recall": mr, "macro_f1": mf1,
                "runtime_s": rt
            })

    return rows


# ============================================================
# ✅ RUN ALL
# ============================================================
log_header("CONFIG SUMMARY")
log_kv("Device", DEVICE)
log_kv("Split mode", SPLIT_MODE)
log_kv("Test size", TEST_SIZE)
log_kv("Corrupt where", CORRUPT_WHERE)
log_kv("LABEL_MODE", LABEL_MODE)
log_kv("SEQ_LEN", SEQ_LEN)
log_kv("SEQ_STRIDE", SEQ_STRIDE)
log_kv("Epochs", EPOCHS)
log_kv("Batch", BATCH)
log_kv("LR", LR)
log_kv("GRAD_CLIP_NORM", GRAD_CLIP_NORM)
log_kv("MAX_TAB_TRAIN", MAX_TAB_TRAIN)
log_kv("MAX_SEQ_TRAIN", MAX_SEQ_TRAIN)
log_kv("MAX_SEQ_TEST", MAX_SEQ_TEST)
log_kv("USE_INTERNAL_VAL", USE_INTERNAL_VAL)
log_kv("VAL_FRACTION", VAL_FRACTION)
log_kv("USE_XGBOOST", USE_XGBOOST)
log_kv("XGB_TREE_METHOD", XGB_TREE_METHOD)
log_kv("XGB_N_ESTIMATORS", XGB_N_ESTIMATORS)
log_kv("SHOW_HEATMAP_INLINE", SHOW_HEATMAP_INLINE)
log_kv("SHOW_FI_INLINE", SHOW_FI_INLINE)
log_kv("Only plot clean+worst", PLOT_ONLY_CLEAN_AND_WORST)
log_kv("FI metric", FI_METRIC)
log_kv("FI seed", FI_SEED)
log_kv("FI_SAGE_N_PERMS", FI_SAGE_N_PERMS)
log_kv("FI_SAGE_MASKING", FI_SAGE_MASKING)
log_kv("MAX_FI_TEST_SAMPLES_TAB", MAX_FI_TEST_SAMPLES_TAB)
log_kv("MAX_FI_TEST_SAMPLES_SEQ", MAX_FI_TEST_SAMPLES_SEQ)
log_kv("MAX_FI_FEATURES", MAX_FI_FEATURES)
log_kv("MAX_SENSOR_COLS", MAX_SENSOR_COLS)

items = []
items += load_dataset_lbnl()
items += load_dataset_wang()

log_header(f"TOTAL BUILDINGS TO RUN: {len(items)}")

all_results = []
for item in items:
    all_results.extend(run_one_building(item))

res_df = pd.DataFrame(all_results)
res_path = OUT / "results_all.csv"
res_df.to_csv(res_path, index=False)

fi_all = pd.DataFrame(FEATURE_IMPORTANCE_ROWS)
fi_path = OUT / "feature_importance_all.csv"
fi_all.to_csv(fi_path, index=False)

cm_all = pd.DataFrame(CONFMAT_ROWS)
cm_path = OUT / "confmat_all.csv"
cm_all.to_csv(cm_path, index=False)

log_header("✅ FINAL OUTPUTS SAVED")
log_kv("Results rows", len(res_df))
log_kv("Saved results_all.csv", str(res_path))
log_kv("FI rows", len(fi_all))
log_kv("Saved feature_importance_all.csv", str(fi_path))
log_kv("Confmat rows", len(cm_all))
log_kv("Saved confmat_all.csv", str(cm_path))

if len(res_df):
    log_kv("Datasets", sorted(res_df["source_dataset"].unique()))
    log_kv("Buildings", sorted(res_df["building"].unique()))
    log_kv("Models", sorted(res_df["model"].unique()))
    log_kv("Manipulations", sorted(res_df["manipulation"].unique()))

print("\n📁 Output folder contents:")
for p in sorted(OUT.glob("*")):
    print(" -", p.name)
