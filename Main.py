# ============================================================
# 🚀 UNIFIED FAULT DETECTION NOTEBOOK CELL (FAST-ish + GPU + CLEAN BASELINE + NO LEAKAGE)
# DATASETS:
#   A) LBNL DataFDD synthesis inventory (sensor+fault intervals)
#   B) Nature LCU Wang dataset (auditorium/office/hospital with labeling)
#
# MODELS:
#   - Sequence (GPU): TinyLSTM, TinyCNNLSTM, TinyInformerClassifier
#   - Tabular (CPU): FAST LinearSVM, RF, XGBoost
#
# ROBUSTNESS:
#   - clean + 5 degradations × 3 levels: noise, drift, bias, missing, sampling
#   - CORRUPT_WHERE: "test" (default) or "both"
#   - SPLIT_MODE: "time" (default) or "stratified"
#
# SPEED KNOBS:
#   - SEQ_STRIDE > 1 reduces sequences
#   - MAX_SEQ_TRAIN / MAX_SEQ_TEST caps seq samples
#   - MAX_TAB_TRAIN caps tabular training samples (keeps SVM fast)
#   - PLOT_ONLY_CLEAN_AND_WORST reduces plotting overhead
#
# SCIENTIFIC FEATURE IMPORTANCE (ONE METHOD FOR ALL):
#   ✅ Permutation Importance on SAME held-out test set
#   ✅ SAME metric for ALL models (macro-F1 by default)
#   ✅ NO CLIPPING (negatives preserved)
#   ✅ mean ± std over N repeats (stability + honesty)
#   ✅ signed plot around zero line
#
# OUTPUTS:
#   ✅ ./fdd_out/results_all.csv
#   ✅ ./fdd_out/feature_importance_all.csv  (signed mean/std)
#   ✅ ./fdd_out/confmat_all.csv
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# ✅ CONFIG (EDIT HERE)
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("./fdd_out")
OUT.mkdir(exist_ok=True)

SPLIT_MODE = "time"          # "time" (recommended) or "stratified"
TEST_SIZE  = 0.20
CORRUPT_WHERE = "test"       # "test" (recommended) or "both"

# ✅ raw vs unified fault families
LABEL_MODE = "raw"        # "raw" or "family"

# Sequence config
SEQ_LEN = 10
SEQ_STRIDE = 1
EPOCHS = 10
BATCH  = 1024
LR     = 5e-4

# Caps for speed
MAX_SEQ_TRAIN = 100000
MAX_SEQ_TEST  = 20000
MAX_TAB_TRAIN = 80000     # set None to disable

# Plotting: show heatmaps inline
SHOW_HEATMAP_INLINE = True
PLOT_ONLY_CLEAN_AND_WORST = True

# Feature importance
COMPUTE_FEATURE_IMPORTANCE = True
SHOW_FI_INLINE = True
FI_TOPK = 25

# ✅ ONE FAIR METHOD FOR ALL: Permutation importance
FI_METRIC = "macro_f1"       # "macro_f1" or "accuracy" (choose one and keep it fixed)
FI_N_REPEATS = 5             # increase to 10+ for more stability if runtime allows
FI_SEED = 123

# For sequence FI speed
MAX_SEQ_IMPORTANCE_SAMPLES = 3000
MAX_SEQ_FEATURES_FOR_FI = None  # None = all, or e.g. 30

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
# 🧾 Fancy logging (YOUR STYLE)
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

def show_confmat(cm, class_names, title, show_inline=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    if show_inline:
        plt.show()
    plt.close(fig)

def show_feature_importance_signed(feat_names, mean_drop, std_drop, title, topk=25,
                                   show_inline=True, rank_by="abs"):
    """
    Scientifically honest permutation importance:
      - shows signed mean drop (can be negative)
      - error bars (std over repeats)
      - ranks by abs(mean) by default so strong negatives are not hidden
    """
    if mean_drop is None or feat_names is None or len(feat_names) == 0:
        return
    if std_drop is None:
        std_drop = np.zeros_like(mean_drop, dtype=float)

    df = pd.DataFrame({
        "feature": feat_names,
        "mean_drop": mean_drop,
        "std_drop": std_drop
    })

    if rank_by == "abs":
        df["rank_key"] = df["mean_drop"].abs()
        df = df.sort_values("rank_key", ascending=False)
    else:
        df = df.sort_values("mean_drop", ascending=False)

    top = df.head(topk).iloc[::-1]  # reverse for barh

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["mean_drop"], xerr=top["std_drop"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(f"Permutation importance: baseline {FI_METRIC} − permuted {FI_METRIC} (mean ± std)")
    fig.tight_layout()
    if show_inline:
        plt.show()
    plt.close(fig)

# -----------------------------
# ✅ Aggregators (ONE CSV at end)
# -----------------------------
FEATURE_IMPORTANCE_ROWS = []   # long-form
CONFMAT_ROWS = []              # long-form

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
                               feat_names, mean_drop, std_drop, method):
    """
    Stores signed mean_drop (can be negative) + std. No clipping. Scientific.
    """
    if mean_drop is None or feat_names is None:
        return
    if std_drop is None:
        std_drop = np.zeros_like(mean_drop, dtype=float)

    for f, mu, sd in zip(feat_names, mean_drop, std_drop):
        FEATURE_IMPORTANCE_ROWS.append({
            "source_dataset": src,
            "building": building,
            "label_mode": label_mode,
            "tag": tag_base,
            "model": model_name,
            "manipulation": manip_name,
            "level": lvl,
            "feature": str(f),
            "importance_mean": float(mu),   # signed
            "importance_std": float(sd),
            "method": method
        })

# -----------------------------
# ✅ Fault family mapping (unified)
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
# Degradations (operate on scaled arrays)
# -----------------------------
def degrade_noise(X, lvl):
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
# ✅ Smaller GPU models
# -----------------------------
class TinyLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TinyCNNLSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        x = x.transpose(2, 1)
        x = torch.relu(self.cnn(x))
        x = x.transpose(2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TinyInformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.in_proj(x)
        z = self.encoder(x)
        return self.cls(z[:, -1, :])

def train_torch_classifier(model, Xtr, ytr, Xte, epochs=10, batch_size=512, lr=5e-4):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long()),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch_size):
            xb = torch.tensor(Xte[i:i+batch_size]).float().to(DEVICE)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)

@torch.no_grad()
def torch_predict_labels(model, X, batch_size=512):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size]).float().to(DEVICE)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)

# -----------------------------
# ✅ ONE FAIR METHOD: Permutation Importance for ALL models
# -----------------------------
def metric_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def metric_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_fi_metric_fn():
    if FI_METRIC == "accuracy":
        return metric_accuracy
    return metric_macro_f1

def perm_importance_tabular(predict_fn, X, y, metric_fn, n_repeats=5, seed=123):
    """
    Permutation importance for TABULAR models.
    Returns mean_drop, std_drop per feature (signed).
    """
    rng = np.random.default_rng(seed)
    base_pred = predict_fn(X)
    base_score = metric_fn(y, base_pred)

    drops = np.zeros((n_repeats, X.shape[1]), dtype=float)
    for r in range(n_repeats):
        for j in range(X.shape[1]):
            Xp = X.copy()
            idx = rng.permutation(len(Xp))
            Xp[:, j] = Xp[idx, j]
            predp = predict_fn(Xp)
            scorep = metric_fn(y, predp)
            drops[r, j] = base_score - scorep
    return drops.mean(axis=0), drops.std(axis=0)

def perm_importance_sequence(predict_fn_seq, Xseq, yseq, metric_fn,
                            n_repeats=5, seed=123, max_samples=3000, max_features=None):
    """
    Permutation importance for SEQUENCE models (X: N,T,F).
    Shuffles each feature across SAMPLES (keeps within-window temporal structure intact).
    Returns used_feature_indices, mean_drop, std_drop (signed).
    """
    if len(Xseq) == 0:
        return np.array([], dtype=int), None, None

    rng = np.random.default_rng(seed)

    if max_samples is not None and len(Xseq) > max_samples:
        idx = rng.choice(len(Xseq), size=max_samples, replace=False)
        Xuse = Xseq[idx].copy()
        yuse = yseq[idx].copy()
    else:
        Xuse = Xseq.copy()
        yuse = yseq.copy()

    base_pred = predict_fn_seq(Xuse)
    base_score = metric_fn(yuse, base_pred)

    F = Xuse.shape[2]
    feat_idx = np.arange(F)
    if max_features is not None:
        feat_idx = feat_idx[:min(max_features, F)]

    drops = np.zeros((n_repeats, len(feat_idx)), dtype=float)

    for r in range(n_repeats):
        for k, j in enumerate(feat_idx):
            Xp = Xuse.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, :, j] = Xp[perm, :, j]
            predp = predict_fn_seq(Xp)
            scorep = metric_fn(yuse, predp)
            drops[r, k] = base_score - scorep

    return feat_idx, drops.mean(axis=0), drops.std(axis=0)

def make_torch_predict_fn(model):
    return lambda X: torch_predict_labels(model, X, batch_size=BATCH)

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

    df[sensors] = df[sensors].ffill().bfill()
    meta = {"inv_fault_map": inv, "inv_family_map": id2fam}
    return df, sensors, meta

def load_dataset_lbnl():
    RAW  = Path("/kaggle/input/datafdd/5_lbnl_data_synthesis_inventory/raw/")
    file_map = {
        "MZVAV_1":   {"sensor": RAW/"MZVAV-1.csv",   "faults": RAW/"MZVAV-1-faults.csv"},
        "MZVAV_2_1": {"sensor": RAW/"MZVAV-2-1.csv", "faults": RAW/"MZVAV-2-1-faults.csv"},
        "MZVAV_2_2": {"sensor": RAW/"MZVAV-2-2.csv", "faults": RAW/"MZVAV-2-2-faults.csv"},
        "RTU":       {"sensor": RAW/"RTU.csv",       "faults": RAW/"RTU-faults.csv"},
        "SZCAV":     {"sensor": RAW/"SZCAV.csv",     "faults": RAW/"SZCAV-faults.csv"},
        "SZVAV":     {"sensor": RAW/"SZVAV.csv",     "faults": RAW/"SZVAV-faults.csv"},
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
        log_kv("Faults(raw)", list(meta["inv_fault_map"].values())[:10] + (["..."] if len(meta["inv_fault_map"]) > 10 else []))
        log_kv("FaultFamilies", list(meta["inv_family_map"].values()))
    return items

# ============================================================
# DATASET B) Nature LCU Wang
# ============================================================
def preprocess_wang_building(df):
    df = df.copy()

    if "DATE" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["Time"].astype(str),
                                         errors="coerce", dayfirst=True)
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
        log_kv("Faults(raw)", list(meta["inv_fault_map"].values()))
        log_kv("FaultFamilies", list(meta["inv_family_map"].values()))
    return items

# ============================================================
# RUNNER
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

    X_all = df[sensors].to_numpy().astype(np.float32)
    y_all = df[y_col].to_numpy().astype(int)

    # split before scaling
    if SPLIT_MODE == "time":
        split = int(len(X_all) * (1 - TEST_SIZE))
        Xtr_raw, Xte_raw = X_all[:split], X_all[split:]
        ytr_raw, yte_raw = y_all[:split], y_all[split:]
    else:
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, stratify=y_all, shuffle=True, random_state=42
        )

    log_sub(f"{tag_base} | Split")
    log_kv("Train raw", Xtr_raw.shape)
    log_kv("Test raw", Xte_raw.shape)
    log_kv("Classes", class_names)

    # scale on train only (no leakage)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    rows = []
    metric_fn = get_fi_metric_fn()
    fi_method = f"permutation_drop_{FI_METRIC}_mean_std"

    for manip_name, lvls in levels.items():
        for lvl in lvls:
            log_header(f"{tag_base} | manip={manip_name} | level={lvl}")

            Xtr_m = Xtr.copy(); ytr_m = ytr_raw.copy()
            Xte_m = Xte.copy(); yte_m = yte_raw.copy()

            # apply corruption (default test-only)
            if manip_name == "clean":
                pass
            elif manip_name == "sampling":
                if CORRUPT_WHERE == "both":
                    Xtr_m, ytr_m = degrade_sampling(Xtr_m, ytr_m, lvl)
                Xte_m, yte_m = degrade_sampling(Xte_m, yte_m, lvl)
            else:
                if CORRUPT_WHERE == "both":
                    Xtr_m = manips[manip_name](Xtr_m, lvl)
                Xte_m = manips[manip_name](Xte_m, lvl)

            log_kv("Train after corrupt", Xtr_m.shape)
            log_kv("Test after corrupt", Xte_m.shape)

            # ---------------- TABULAR (CPU) ----------------
            Xtr_tab, ytr_tab = cap_tabular(Xtr_m, ytr_m, MAX_TAB_TRAIN, seed=42)
            if len(Xtr_tab) != len(Xtr_m):
                log_kv("Tabular cap", f"{len(Xtr_m)} -> {len(Xtr_tab)}")

            # ✅ FAST LinearSVM
            t0 = time.time()
            svm = LinearSVC(C=1.0, dual=False, max_iter=4000)
            svm.fit(Xtr_tab, ytr_tab)
            pred = svm.predict(Xte_m)
            rt = time.time() - t0

            acc = accuracy_score(yte_m, pred)
            f1m = f1_score(yte_m, pred, average="macro", zero_division=0)
            f1w = f1_score(yte_m, pred, average="weighted", zero_division=0)

            log_sub("TABULAR | LinearSVM")
            log_kv("Accuracy", acc)
            log_kv("Macro F1", f1m)
            log_kv("Weighted F1", f1w)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

            add_confmat_rows(src, building, LABEL_MODE, tag_base, "LinearSVM", manip_name, lvl,
                             class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(
                    cm_norm, class_names,
                    title=f"{tag_base} | LinearSVM | {manip_name}={lvl}",
                    show_inline=True
                )

            # ✅ ONE METHOD FI: permutation (signed, mean±std)
            if COMPUTE_FEATURE_IMPORTANCE:
                mean_drop, std_drop = perm_importance_tabular(
                    predict_fn=lambda Z: svm.predict(Z),
                    X=Xte_m, y=yte_m,
                    metric_fn=metric_fn,
                    n_repeats=FI_N_REPEATS,
                    seed=FI_SEED
                )
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "LinearSVM", manip_name, lvl,
                                            sensors, mean_drop, std_drop, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        sensors, mean_drop, std_drop,
                        title=f"Permutation FI | {tag_base} | LinearSVM | {manip_name}={lvl} | metric={FI_METRIC}",
                        topk=FI_TOPK,
                        show_inline=True,
                        rank_by="abs"
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl,
                "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "LinearSVM", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
                "runtime_s": rt
            })

            # RF
            t0 = time.time()
            rf = RandomForestClassifier(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42)
            rf.fit(Xtr_tab, ytr_tab)
            pred = rf.predict(Xte_m)
            rt = time.time() - t0

            acc = accuracy_score(yte_m, pred)
            f1m = f1_score(yte_m, pred, average="macro", zero_division=0)
            f1w = f1_score(yte_m, pred, average="weighted", zero_division=0)

            log_sub("TABULAR | RF")
            log_kv("Accuracy", acc)
            log_kv("Macro F1", f1m)
            log_kv("Weighted F1", f1w)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

            add_confmat_rows(src, building, LABEL_MODE, tag_base, "RF", manip_name, lvl,
                             class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(
                    cm_norm, class_names,
                    title=f"{tag_base} | RF | {manip_name}={lvl}",
                    show_inline=True
                )

            # ✅ permutation FI
            if COMPUTE_FEATURE_IMPORTANCE:
                mean_drop, std_drop = perm_importance_tabular(
                    predict_fn=lambda Z: rf.predict(Z),
                    X=Xte_m, y=yte_m,
                    metric_fn=metric_fn,
                    n_repeats=FI_N_REPEATS,
                    seed=FI_SEED
                )
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "RF", manip_name, lvl,
                                            sensors, mean_drop, std_drop, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        sensors, mean_drop, std_drop,
                        title=f"Permutation FI | {tag_base} | RF | {manip_name}={lvl} | metric={FI_METRIC}",
                        topk=FI_TOPK,
                        show_inline=True,
                        rank_by="abs"
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl,
                "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "RF", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
                "runtime_s": rt
            })

            # XGBoost
            t0 = time.time()
            xgb = XGBClassifier(
                tree_method="hist",
                max_depth=5,
                n_estimators=120,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42
            )
            xgb.fit(Xtr_tab, ytr_tab)
            pred = xgb.predict(Xte_m)
            rt = time.time() - t0

            acc = accuracy_score(yte_m, pred)
            f1m = f1_score(yte_m, pred, average="macro", zero_division=0)
            f1w = f1_score(yte_m, pred, average="weighted", zero_division=0)

            log_sub("TABULAR | XGBoost")
            log_kv("Accuracy", acc)
            log_kv("Macro F1", f1m)
            log_kv("Weighted F1", f1w)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

            add_confmat_rows(src, building, LABEL_MODE, tag_base, "XGBoost", manip_name, lvl,
                             class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(
                    cm_norm, class_names,
                    title=f"{tag_base} | XGBoost | {manip_name}={lvl}",
                    show_inline=True
                )

            # ✅ permutation FI
            if COMPUTE_FEATURE_IMPORTANCE:
                mean_drop, std_drop = perm_importance_tabular(
                    predict_fn=lambda Z: xgb.predict(Z),
                    X=Xte_m, y=yte_m,
                    metric_fn=metric_fn,
                    n_repeats=FI_N_REPEATS,
                    seed=FI_SEED
                )
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "XGBoost", manip_name, lvl,
                                            sensors, mean_drop, std_drop, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        sensors, mean_drop, std_drop,
                        title=f"Permutation FI | {tag_base} | XGBoost | {manip_name}={lvl} | metric={FI_METRIC}",
                        topk=FI_TOPK,
                        show_inline=True,
                        rank_by="abs"
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl,
                "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "XGBoost", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
                "runtime_s": rt
            })

            # ---------------- SEQUENCE (GPU) ----------------
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

            # LSTM
            t0 = time.time()
            m = TinyLSTM(input_dim=Xtr_seq.shape[2], num_classes=n_classes)
            pred = train_torch_classifier(m, Xtr_seq, ytr_seq, Xte_seq, epochs=EPOCHS, batch_size=BATCH, lr=LR)
            rt = time.time() - t0

            acc = accuracy_score(yte_seq, pred)
            f1m = f1_score(yte_seq, pred, average="macro", zero_division=0)
            f1w = f1_score(yte_seq, pred, average="weighted", zero_division=0)

            log_sub("SEQUENCE | LSTM")
            log_kv("Accuracy", acc)
            log_kv("Macro F1", f1m)
            log_kv("Weighted F1", f1w)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

            add_confmat_rows(src, building, LABEL_MODE, tag_base, "LSTM", manip_name, lvl,
                             class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(
                    cm_norm, class_names,
                    title=f"{tag_base} | LSTM | {manip_name}={lvl}",
                    show_inline=True
                )

            # ✅ permutation FI (sequence)
            if COMPUTE_FEATURE_IMPORTANCE:
                pred_fn = make_torch_predict_fn(m.to(DEVICE))
                feat_idx, mean_drop, std_drop = perm_importance_sequence(
                    predict_fn_seq=pred_fn,
                    Xseq=Xte_seq, yseq=yte_seq,
                    metric_fn=metric_fn,
                    n_repeats=FI_N_REPEATS,
                    seed=FI_SEED + 999,
                    max_samples=MAX_SEQ_IMPORTANCE_SAMPLES,
                    max_features=MAX_SEQ_FEATURES_FOR_FI
                )
                used_names = [sensors[j] for j in feat_idx]
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "LSTM", manip_name, lvl,
                                            used_names, mean_drop, std_drop, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        used_names, mean_drop, std_drop,
                        title=f"Permutation FI | {tag_base} | LSTM | {manip_name}={lvl} | metric={FI_METRIC}",
                        topk=FI_TOPK,
                        show_inline=True,
                        rank_by="abs"
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl,
                "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "LSTM", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
                "runtime_s": rt
            })

            # CNN-LSTM
            t0 = time.time()
            m = TinyCNNLSTM(input_dim=Xtr_seq.shape[2], num_classes=n_classes)
            pred = train_torch_classifier(m, Xtr_seq, ytr_seq, Xte_seq, epochs=EPOCHS, batch_size=BATCH, lr=LR)
            rt = time.time() - t0

            acc = accuracy_score(yte_seq, pred)
            f1m = f1_score(yte_seq, pred, average="macro", zero_division=0)
            f1w = f1_score(yte_seq, pred, average="weighted", zero_division=0)

            log_sub("SEQUENCE | CNN-LSTM")
            log_kv("Accuracy", acc)
            log_kv("Macro F1", f1m)
            log_kv("Weighted F1", f1w)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

            add_confmat_rows(src, building, LABEL_MODE, tag_base, "CNN-LSTM", manip_name, lvl,
                             class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(
                    cm_norm, class_names,
                    title=f"{tag_base} | CNN-LSTM | {manip_name}={lvl}",
                    show_inline=True
                )

            # ✅ permutation FI
            if COMPUTE_FEATURE_IMPORTANCE:
                pred_fn = make_torch_predict_fn(m.to(DEVICE))
                feat_idx, mean_drop, std_drop = perm_importance_sequence(
                    predict_fn_seq=pred_fn,
                    Xseq=Xte_seq, yseq=yte_seq,
                    metric_fn=metric_fn,
                    n_repeats=FI_N_REPEATS,
                    seed=FI_SEED + 1999,
                    max_samples=MAX_SEQ_IMPORTANCE_SAMPLES,
                    max_features=MAX_SEQ_FEATURES_FOR_FI
                )
                used_names = [sensors[j] for j in feat_idx]
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "CNN-LSTM", manip_name, lvl,
                                            used_names, mean_drop, std_drop, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        used_names, mean_drop, std_drop,
                        title=f"Permutation FI | {tag_base} | CNN-LSTM | {manip_name}={lvl} | metric={FI_METRIC}",
                        topk=FI_TOPK,
                        show_inline=True,
                        rank_by="abs"
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl,
                "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "CNN-LSTM", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
                "runtime_s": rt
            })

            # Informer
            t0 = time.time()
            m = TinyInformerClassifier(input_dim=Xtr_seq.shape[2], num_classes=n_classes, d_model=32, nhead=2, num_layers=1)
            pred = train_torch_classifier(m, Xtr_seq, ytr_seq, Xte_seq, epochs=EPOCHS, batch_size=BATCH, lr=LR)
            rt = time.time() - t0

            acc = accuracy_score(yte_seq, pred)
            f1m = f1_score(yte_seq, pred, average="macro", zero_division=0)
            f1w = f1_score(yte_seq, pred, average="weighted", zero_division=0)

            log_sub("SEQUENCE | Informer")
            log_kv("Accuracy", acc)
            log_kv("Macro F1", f1m)
            log_kv("Weighted F1", f1w)
            log_kv("Runtime (s)", rt)

            cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
            cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)

            add_confmat_rows(src, building, LABEL_MODE, tag_base, "Informer", manip_name, lvl,
                             class_names, cm_raw, cm_norm)

            if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                show_confmat(
                    cm_norm, class_names,
                    title=f"{tag_base} | Informer | {manip_name}={lvl}",
                    show_inline=True
                )

            # ✅ permutation FI
            if COMPUTE_FEATURE_IMPORTANCE:
                pred_fn = make_torch_predict_fn(m.to(DEVICE))
                feat_idx, mean_drop, std_drop = perm_importance_sequence(
                    predict_fn_seq=pred_fn,
                    Xseq=Xte_seq, yseq=yte_seq,
                    metric_fn=metric_fn,
                    n_repeats=FI_N_REPEATS,
                    seed=FI_SEED + 2999,
                    max_samples=MAX_SEQ_IMPORTANCE_SAMPLES,
                    max_features=MAX_SEQ_FEATURES_FOR_FI
                )
                used_names = [sensors[j] for j in feat_idx]
                add_feature_importance_rows(src, building, LABEL_MODE, tag_base, "Informer", manip_name, lvl,
                                            used_names, mean_drop, std_drop, method=fi_method)

                if should_plot(manip_name, lvl) and SHOW_FI_INLINE:
                    show_feature_importance_signed(
                        used_names, mean_drop, std_drop,
                        title=f"Permutation FI | {tag_base} | Informer | {manip_name}={lvl} | metric={FI_METRIC}",
                        topk=FI_TOPK,
                        show_inline=True,
                        rank_by="abs"
                    )

            rows.append({
                "source_dataset": src, "building": building, "label_mode": LABEL_MODE,
                "manipulation": manip_name, "level": lvl,
                "corrupt_where": CORRUPT_WHERE, "split_mode": SPLIT_MODE,
                "model": "Informer", "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
                "runtime_s": rt
            })

    return rows

# ============================================================
# ✅ RUN ALL
# ============================================================
log_header("CONFIG SUMMARY")
log_kv("Device", DEVICE)
log_kv("Split mode", SPLIT_MODE)
log_kv("Corrupt where", CORRUPT_WHERE)
log_kv("LABEL_MODE", LABEL_MODE)
log_kv("SEQ_LEN", SEQ_LEN)
log_kv("SEQ_STRIDE", SEQ_STRIDE)
log_kv("Epochs", EPOCHS)
log_kv("Batch", BATCH)
log_kv("MAX_TAB_TRAIN", MAX_TAB_TRAIN)
log_kv("MAX_SEQ_TRAIN", MAX_SEQ_TRAIN)
log_kv("MAX_SEQ_TEST", MAX_SEQ_TEST)
log_kv("SHOW_HEATMAP_INLINE", SHOW_HEATMAP_INLINE)
log_kv("SHOW_FI_INLINE", SHOW_FI_INLINE)
log_kv("Only plot clean+worst", PLOT_ONLY_CLEAN_AND_WORST)
log_kv("FI metric", FI_METRIC)
log_kv("FI repeats", FI_N_REPEATS)
log_kv("FI seed", FI_SEED)

items = []
items += load_dataset_lbnl()
items += load_dataset_wang()

log_header(f"TOTAL BUILDINGS TO RUN: {len(items)}")

all_results = []
for item in items:
    all_results.extend(run_one_building(item))

# main results
res_df = pd.DataFrame(all_results)
res_path = OUT / "results_all.csv"
res_df.to_csv(res_path, index=False)

# ✅ one combined file: feature importance (signed mean/std, no clipping)
fi_all = pd.DataFrame(FEATURE_IMPORTANCE_ROWS)
fi_path = OUT / "feature_importance_all.csv"
fi_all.to_csv(fi_path, index=False)

# ✅ one combined file: confusion matrices (true vs pred heatmap values)
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
