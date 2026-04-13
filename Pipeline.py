# ============================================================
# Unified fault detection pipeline with model statistics export
#
# This script:
# - loads and preprocesses two fault detection datasets
# - supports multiple split strategies, including episode-based temporal splitting
# - trains tabular and sequence models once on the training split
# - evaluates the trained models on clean and degraded test data
# - exports prediction metrics, confusion matrices, feature importance, and model statistics
#
# Output files:
#   ./fdd_out/results_all.csv
#   ./fdd_out/feature_importance_all.csv
#   ./fdd_out/confmat_all.csv
#   ./fdd_out/model_stats_all.csv
#   ./fdd_out/model_table_rows.tex
# ============================================================

import os
import time
import warnings
import io
import pickle
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


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("./fdd_out")
OUT.mkdir(exist_ok=True)

# Available split modes:
# - "time": strict chronological split
# - "stratified": random row-wise stratified split
# - "stratified_time": class-wise chronological split
# - "episode_stratified_time": split contiguous label episodes in time
SPLIT_MODE = "stratified_time"

TEST_SIZE = 0.20
CORRUPT_WHERE = "test"      # this pipeline assumes train-once, evaluate on test
LABEL_MODE = "raw"          # "raw" or "family"

# Sequence model settings
SEQ_LEN = 10
SEQ_STRIDE = 1
EPOCHS = 15
BATCH = 1024
LR = 8e-4
GRAD_CLIP_NORM = 1.0

# Optional caps to reduce runtime and memory usage
MAX_SEQ_TRAIN = 300000
MAX_SEQ_TEST = 50000
MAX_TAB_TRAIN = 100000      # set to None to disable

# Optional validation split inside the training set
USE_INTERNAL_VAL = True
VAL_FRACTION = 0.12

# Optional sensor column cap
MAX_SENSOR_COLS = 19

# Plotting configuration
SHOW_HEATMAP_INLINE = True
PLOT_ONLY_CLEAN_AND_WORST = True

# Global feature importance configuration (SAGE-style Monte Carlo)
COMPUTE_FEATURE_IMPORTANCE = False
SHOW_FI_INLINE = True
FI_TOPK = 25

FI_METRIC = "macro_f1"      # "macro_f1" or "accuracy"
FI_SEED = 123
FI_SAGE_N_PERMS = 20
FI_SAGE_MASKING = "sample"  # "sample" or "mean"
MAX_FI_TEST_SAMPLES_TAB = 5000
MAX_FI_TEST_SAMPLES_SEQ = 3000
MAX_FI_FEATURES = None

# XGBoost configuration
USE_XGBOOST = True
XGB_TREE_METHOD = "hist"    # "hist" is stable; "gpu_hist" if desired and available
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 5
XGB_LR = 0.08

# Episode split configuration
EPISODE_MIN_TRAIN_EPISODES_PER_CLASS = 1
EPISODE_MIN_TEST_EPISODES_PER_CLASS = 1
SEQUENCE_GAP_ROWS = SEQ_LEN - 1   # purge boundary rows for sequence safety

# Test degradations
levels = {
    "clean":    [0],
    "noise":    [0.01, 0.05, 0.10],
    "drift":    [0.01, 0.05, 0.10],
    "bias":     [0.01, 0.05, 0.10],
    "missing":  [0.05, 0.10, 0.20],
    "sampling": [2, 5, 10],
}

assert CORRUPT_WHERE == "test", "This pipeline assumes CORRUPT_WHERE='test'."


# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------
def log_header(title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def log_sub(title):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)


def log_kv(k, v):
    print(f"   • {k}: {v}")


# ------------------------------------------------------------
# Prediction handling
# ------------------------------------------------------------
def coerce_pred_labels(y_pred):
    """
    Convert predictions to a flat integer label vector.

    This function handles:
    - class probability / logits arrays of shape (N, C)
    - already predicted class arrays of shape (N,)
    - floating arrays that should represent label ids
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.ravel(y_pred)
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = y_pred.astype(int)
    return y_pred


def log_metrics_block(model_name, manip_name, lvl, y_true, y_pred):
    """
    Compute and print the main classification metrics for one experiment.
    """
    y_pred = coerce_pred_labels(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "." * 90)
    print(f"EXPERIMENT | Model={model_name} | manipulation={manip_name} | level={lvl}")
    print("." * 90)
    log_kv("Accuracy", acc)
    log_kv("Macro Precision", p)
    log_kv("Macro Recall", r)
    log_kv("Macro F1", f1m)

    return acc, p, r, f1m


def print_class_balance(y, inv_map, title, topk=999):
    """
    Print class counts for a label vector, using inverse label mapping when available.
    """
    vals, cnts = np.unique(y, return_counts=True)
    pairs = sorted([(int(v), int(c)) for v, c in zip(vals, cnts)], key=lambda x: -x[1])
    print(f"   • {title} class counts:")
    for v, c in pairs[:topk]:
        name = inv_map.get(v, str(v))
        print(f"      - {v:>3} | {c:>8} | {name}")


# ------------------------------------------------------------
# Stratified splitting helpers
# ------------------------------------------------------------
def stratified_split_keep_singletons_in_train(X, y, test_size=0.2, seed=42, tag=""):
    """
    Perform a stratified train/test split while forcing singleton classes into train.

    This prevents sklearn stratification errors when a class appears only once.
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n < 5:
        split = max(1, int(n * (1 - test_size)))
        return X[:split], X[split:], y[:split], y[split:]

    vals, cnts = np.unique(y, return_counts=True)
    cnt_map = {int(v): int(c) for v, c in zip(vals, cnts)}

    singleton_classes = [c for c, k in cnt_map.items() if k < 2]
    if len(singleton_classes) == 0:
        return train_test_split(
            X, y, test_size=test_size, stratify=y, shuffle=True, random_state=seed
        )

    idx_all = np.arange(n)
    idx_single = idx_all[np.isin(y, singleton_classes)]
    idx_rest = idx_all[~np.isin(y, singleton_classes)]

    if len(idx_rest) < 5:
        if tag:
            print(f"Warning [{tag}]: Too few non-singleton samples after removing singleton classes. Using all rows for training.")
        Xtr, ytr = X, y
        Xte, yte = X[:0], y[:0]
        return Xtr, Xte, ytr, yte

    X_rest = X[idx_rest]
    y_rest = y[idx_rest]

    vals2, cnts2 = np.unique(y_rest, return_counts=True)
    if cnts2.min() < 2:
        if tag:
            print(
                f"Warning [{tag}]: After removing singletons, stratification is still impossible "
                f"(minimum class count < 2). Falling back to non-stratified random split."
            )
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_rest, y_rest, test_size=test_size, shuffle=True, random_state=seed, stratify=None
        )
    else:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_rest, y_rest, test_size=test_size, shuffle=True, random_state=seed, stratify=y_rest
        )

    Xtr = np.concatenate([Xtr_r, X[idx_single]], axis=0)
    ytr = np.concatenate([ytr_r, y[idx_single]], axis=0)
    Xte = Xte_r
    yte = yte_r

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[perm], ytr[perm]

    if tag:
        print("\n" + "!" * 110)
        print(f"[{tag}] Stratified split adjusted: singleton classes forced into train: {singleton_classes}")
        print(f"   - train size: {len(ytr)}, test size: {len(yte)}")
        print("!" * 110)

    return Xtr, Xte, ytr, yte


def stratified_train_val_keep_singletons_in_train(Xtr, ytr, val_fraction=0.12, seed=42, tag=""):
    """
    Create an internal validation split from training data, again protecting singleton classes.
    """
    if (not USE_INTERNAL_VAL) or (val_fraction <= 0.0) or (len(Xtr) < 200):
        return Xtr, ytr, None, None

    X_train, X_val, y_train, y_val = stratified_split_keep_singletons_in_train(
        Xtr, ytr, test_size=val_fraction, seed=seed, tag=tag
    )
    return X_train, y_train, X_val, y_val


def stratified_time_split(X, y, test_size=0.2, min_train_per_class=10, min_test_per_class=5):
    """
    Perform a time-respecting split within each class:
    older samples for training, later samples for testing.
    """
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
    test_idx = np.array(test_idx, dtype=int)
    train_idx.sort()
    test_idx.sort()

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ------------------------------------------------------------
# Episode-based temporal split
# ------------------------------------------------------------
def build_episodes_from_labels(df, y_col, time_col):
    """
    Group contiguous rows with the same label into episodes.

    Each episode stores:
    - label
    - start/end row indices
    - start/end timestamps
    - episode length
    """
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    y = df[y_col].to_numpy()

    episodes = []
    if len(df) == 0:
        return df, episodes

    start = 0
    for i in range(1, len(df)):
        if y[i] != y[i - 1]:
            episodes.append({
                "label": int(y[start]),
                "start_idx": int(start),
                "end_idx": int(i - 1),
                "start_time": df.loc[start, time_col],
                "end_time": df.loc[i - 1, time_col],
                "length": int(i - start),
            })
            start = i

    episodes.append({
        "label": int(y[start]),
        "start_idx": int(start),
        "end_idx": int(len(df) - 1),
        "start_time": df.loc[start, time_col],
        "end_time": df.loc[len(df) - 1, time_col],
        "length": int(len(df) - start),
    })

    return df, episodes


def stratified_episode_time_split(
    df,
    y_col,
    time_col,
    test_size=0.20,
    min_train_episodes_per_class=1,
    min_test_episodes_per_class=1,
    gap_rows=0,
    tag=""
):
    """
    Split data by label episodes while keeping temporal order.

    For each class:
    - earlier episodes go to training
    - later episodes go to testing

    Optionally purge rows around the train/test boundary to avoid
    leakage for sequence windows.
    """
    df, episodes = build_episodes_from_labels(df, y_col=y_col, time_col=time_col)

    if len(episodes) == 0:
        return df.iloc[:0].copy(), df.iloc[:0].copy(), episodes, {}

    by_label = {}
    for ep_id, ep in enumerate(episodes):
        by_label.setdefault(ep["label"], []).append((ep_id, ep))

    train_episode_ids = set()
    test_episode_ids = set()

    for label, eps in by_label.items():
        n_eps = len(eps)

        if n_eps <= min_train_episodes_per_class:
            for ep_id, _ in eps:
                train_episode_ids.add(ep_id)
            continue

        n_test = max(min_test_episodes_per_class, int(round(n_eps * test_size)))
        max_allowed_test = n_eps - min_train_episodes_per_class
        n_test = min(n_test, max_allowed_test)

        if n_test <= 0:
            for ep_id, _ in eps:
                train_episode_ids.add(ep_id)
            continue

        split_pos = n_eps - n_test

        for ep_id, _ in eps[:split_pos]:
            train_episode_ids.add(ep_id)
        for ep_id, _ in eps[split_pos:]:
            test_episode_ids.add(ep_id)

    train_idx = []
    test_idx = []

    for ep_id, ep in enumerate(episodes):
        s, e = ep["start_idx"], ep["end_idx"]
        if ep_id in train_episode_ids:
            train_idx.extend(range(s, e + 1))
        elif ep_id in test_episode_ids:
            test_idx.extend(range(s, e + 1))

    train_idx = np.array(sorted(train_idx), dtype=int)
    test_idx = np.array(sorted(test_idx), dtype=int)

    # Purge rows near the partition boundary to protect sequence windows
    if gap_rows > 0 and len(train_idx) > 0 and len(test_idx) > 0:
        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())

        keep_train = np.ones(len(train_idx), dtype=bool)
        keep_test = np.ones(len(test_idx), dtype=bool)

        for i, idx in enumerate(train_idx):
            for k in range(1, gap_rows + 1):
                if (idx - k) in test_set or (idx + k) in test_set:
                    keep_train[i] = False
                    break

        for i, idx in enumerate(test_idx):
            for k in range(1, gap_rows + 1):
                if (idx - k) in train_set or (idx + k) in train_set:
                    keep_test[i] = False
                    break

        train_idx = train_idx[keep_train]
        test_idx = test_idx[keep_test]

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    if tag:
        print("\n" + "=" * 110)
        print(f"[{tag}] Episode-based temporal split")
        print(f"   - total rows      : {len(df)}")
        print(f"   - total episodes  : {len(episodes)}")
        print(f"   - train rows      : {len(df_train)}")
        print(f"   - test rows       : {len(df_test)}")
        print(f"   - train episodes  : {len(train_episode_ids)}")
        print(f"   - test episodes   : {len(test_episode_ids)}")
        print("=" * 110)

    return df_train, df_test, episodes, by_label


def print_episode_split_summary(df_train, df_test, y_col, inv_map, name=""):
    """
    Summarize which classes appear in train, test, or both partitions.
    """
    tr_classes = sorted(np.unique(df_train[y_col]).tolist()) if len(df_train) else []
    te_classes = sorted(np.unique(df_test[y_col]).tolist()) if len(df_test) else []

    tr_only = sorted(set(tr_classes) - set(te_classes))
    te_only = sorted(set(te_classes) - set(tr_classes))
    both = sorted(set(tr_classes) & set(te_classes))

    print("\n" + "=" * 100)
    print(f"EPISODE SPLIT SUMMARY | {name}")
    print("=" * 100)
    print("Train rows:", len(df_train))
    print("Test rows :", len(df_test))
    print("Classes in both:", [inv_map[i] for i in both])
    print("Train only      :", [inv_map[i] for i in tr_only])
    print("Test only       :", [inv_map[i] for i in te_only])


def exact_overlap_count(Xtr, Xte, decimals=6):
    """
    Count exact row overlap between train and test after rounding.

    This is a basic leakage diagnostic for tabular feature matrices.
    """
    if len(Xtr) == 0 or len(Xte) == 0:
        return 0
    tr = pd.DataFrame(np.round(Xtr, decimals))
    te = pd.DataFrame(np.round(Xte, decimals))
    htr = set(pd.util.hash_pandas_object(tr, index=False).values.tolist())
    hte = set(pd.util.hash_pandas_object(te, index=False).values.tolist())
    return len(htr & hte)


# ------------------------------------------------------------
# General preprocessing helpers
# ------------------------------------------------------------
def sensor_cols_generic(df, exclude):
    """
    Return numeric columns that are not in the exclude list.
    """
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


def sanitize_sensors(df, sensors):
    """
    Remove columns that may leak fault labels or metadata into the model.
    """
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
    """
    Hard check that no obvious target-leakage columns remain.
    """
    bad = [
        s for s in sensors
        if ("fault" in s.lower()) or ("label" in s.lower()) or ("truth" in s.lower()) or ("code" in s.lower())
    ]
    assert len(bad) == 0, f"Leakage columns detected in sensors: {bad}"


def create_sequences(X, y, seq_len=24, stride=1):
    """
    Convert tabular time-ordered samples into fixed-length sequences.

    The sequence label is the label of the last time step in the window.
    """
    Xs, ys = [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len - 1])

    if len(Xs) == 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,), dtype=int)

    return np.stack(Xs), np.array(ys, dtype=int)


def cap_sequences(Xseq, yseq, max_n, seed=42):
    """
    Randomly subsample sequence data if it exceeds a given cap.
    """
    if max_n is None or len(Xseq) <= max_n:
        return Xseq, yseq
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xseq), size=max_n, replace=False)
    return Xseq[idx], yseq[idx]


def cap_tabular(X, y, max_n, seed=42):
    """
    Randomly subsample tabular data if it exceeds a given cap.
    """
    if max_n is None or len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


def should_plot(manip_name, lvl):
    """
    Control whether plots should be shown for a given corruption setting.

    If PLOT_ONLY_CLEAN_AND_WORST is enabled, only:
    - clean
    - the most severe level of each corruption
    are plotted.
    """
    if not PLOT_ONLY_CLEAN_AND_WORST:
        return True
    if manip_name == "clean":
        return True
    if manip_name in ["noise", "drift", "bias", "missing"]:
        return lvl == max(levels[manip_name])
    if manip_name == "sampling":
        return lvl == max(levels["sampling"])
    return False


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def show_confmat(cm, class_names, title, show_inline=True):
    """
    Plot a confusion matrix heatmap with aligned tick labels.
    """
    n = len(class_names)
    assert cm.shape == (n, n), f"cm shape {cm.shape} != ({n}, {n})"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
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
    """
    Plot signed global feature importance values with optional error bars.
    """
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


# ------------------------------------------------------------
# Class weighting helpers
# ------------------------------------------------------------
def compute_balanced_weights(y, n_classes, default=1.0):
    """
    Compute balanced class weights for all class ids from 0..n_classes-1.

    Classes missing in the input keep the default weight.
    """
    y = np.asarray(y, dtype=int)
    present = np.unique(y)
    full = np.full((n_classes,), float(default), dtype=np.float32)
    cw_present = compute_class_weight(class_weight="balanced", classes=present, y=y)
    for c, w in zip(present, cw_present):
        if 0 <= int(c) < n_classes:
            full[int(c)] = float(w)
    return full


def compute_sample_weights(y, class_weights):
    """
    Convert class weights into per-sample weights.
    """
    y = np.asarray(y, dtype=int)
    return np.asarray([float(class_weights[int(yi)]) for yi in y], dtype=np.float32)


# ------------------------------------------------------------
# Result aggregators
# ------------------------------------------------------------
FEATURE_IMPORTANCE_ROWS = []
CONFMAT_ROWS = []
MODEL_STATS_ROWS = []


def add_confmat_rows(src, building, label_mode, tag_base, model_name, manip_name, lvl,
                     class_names, cm_raw, cm_norm):
    """
    Store raw and normalized confusion matrix entries in long-table format.
    """
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
    """
    Store feature importance values in long-table format.
    """
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


def bytes_to_human(n_bytes):
    """
    Convert a byte count to a readable size string.
    """
    n = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.2f} {unit}"
        n /= 1024.0


def sklearn_model_size_bytes(model):
    """
    Estimate serialized size of a scikit-learn model using pickle.
    """
    return int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))


def torch_model_size_bytes(model):
    """
    Estimate serialized size of a PyTorch model using state_dict.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return int(buf.getbuffer().nbytes)


def torch_trainable_params(model):
    """
    Count trainable parameters in a PyTorch model.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def linear_svm_param_count(model):
    """
    Count learned weights and intercepts of a linear SVM.
    """
    return int(model.coef_.size + model.intercept_.size)


def rf_total_nodes(model):
    """
    Count total nodes across all trees in a random forest.
    """
    return int(sum(est.tree_.node_count for est in model.estimators_))


def rf_total_leaves(model):
    """
    Count total leaves across all trees in a random forest.
    """
    return int(sum(est.tree_.n_leaves for est in model.estimators_))


def rf_total_trees(model):
    """
    Count trees in a random forest.
    """
    return int(len(model.estimators_))


def xgb_tree_dataframe(model):
    """
    Return XGBoost tree structure as a dataframe.
    """
    return model.get_booster().trees_to_dataframe()


def xgb_total_nodes(model):
    """
    Count total nodes in an XGBoost model.
    """
    df = xgb_tree_dataframe(model)
    return int(len(df))


def xgb_total_leaves(model):
    """
    Count leaf nodes in an XGBoost model.
    """
    df = xgb_tree_dataframe(model)
    if "Feature" in df.columns:
        return int((df["Feature"] == "Leaf").sum())
    return np.nan


def xgb_total_trees(model):
    """
    Count trees in an XGBoost model.
    """
    df = xgb_tree_dataframe(model)
    return int(df["Tree"].nunique())


def add_model_stats_row(
    src, building, label_mode, tag_base,
    model_name, family, architecture, hidden_col,
    param_count=None, node_count=None, leaf_count=None, tree_count=None,
    model_size_bytes=None, table_size_text=None, notes=None
):
    """
    Append one row to the exported model statistics table.
    """
    MODEL_STATS_ROWS.append({
        "source_dataset": src,
        "building": building,
        "label_mode": label_mode,
        "tag": tag_base,
        "model": model_name,
        "family": family,
        "architecture": architecture,
        "hidden_col": hidden_col,
        "param_count": (int(param_count) if param_count is not None and not pd.isna(param_count) else np.nan),
        "node_count": (int(node_count) if node_count is not None and not pd.isna(node_count) else np.nan),
        "leaf_count": (int(leaf_count) if leaf_count is not None and not pd.isna(leaf_count) else np.nan),
        "tree_count": (int(tree_count) if tree_count is not None and not pd.isna(tree_count) else np.nan),
        "model_size_bytes": (int(model_size_bytes) if model_size_bytes is not None and not pd.isna(model_size_bytes) else np.nan),
        "model_size_human": (bytes_to_human(model_size_bytes) if model_size_bytes is not None and not pd.isna(model_size_bytes) else ""),
        "table_size_text": table_size_text,
        "notes": notes
    })


def collect_model_stats_for_building(src, building, label_mode, tag_base, trained, n_features, n_classes):
    """
    Collect size and architecture metadata for all trained models of one building.
    """
    if "LinearSVM" in trained:
        m = trained["LinearSVM"]["model"]
        p = linear_svm_param_count(m)
        sz = sklearn_model_size_bytes(m)
        add_model_stats_row(
            src=src,
            building=building,
            label_mode=label_mode,
            tag_base=tag_base,
            model_name="LinearSVM",
            family="Tabular",
            architecture="Linear classifier with C=1.0 and class-balanced weighting.",
            hidden_col="--",
            param_count=p,
            model_size_bytes=sz,
            table_size_text=f"{p:,} weights",
            notes=f"coef_shape={tuple(m.coef_.shape)}, intercept_shape={tuple(m.intercept_.shape)}, n_features={n_features}, n_classes={n_classes}"
        )

    if "RF" in trained:
        m = trained["RF"]["model"]
        nodes = rf_total_nodes(m)
        leaves = rf_total_leaves(m)
        trees = rf_total_trees(m)
        sz = sklearn_model_size_bytes(m)
        add_model_stats_row(
            src=src,
            building=building,
            label_mode=label_mode,
            tag_base=tag_base,
            model_name="RF",
            family="Tabular",
            architecture="400 decision trees; minimum leaf size 2; unrestricted depth; class-balanced subsampling.",
            hidden_col="--",
            node_count=nodes,
            leaf_count=leaves,
            tree_count=trees,
            model_size_bytes=sz,
            table_size_text=f"{nodes:,} nodes",
            notes=f"trees={trees}, leaves={leaves}, n_features={n_features}, n_classes={n_classes}"
        )

    if "XGBoost" in trained:
        m = trained["XGBoost"]["model"]
        nodes = xgb_total_nodes(m)
        leaves = xgb_total_leaves(m)
        trees = xgb_total_trees(m)
        sz = sklearn_model_size_bytes(m)
        add_model_stats_row(
            src=src,
            building=building,
            label_mode=label_mode,
            tag_base=tag_base,
            model_name="XGBoost",
            family="Tabular",
            architecture=f"300 boosted trees; maximum depth {XGB_MAX_DEPTH}; learning rate {XGB_LR}; row/column subsampling 0.85.",
            hidden_col="--",
            node_count=nodes,
            leaf_count=leaves,
            tree_count=trees,
            model_size_bytes=sz,
            table_size_text=f"{nodes:,} nodes",
            notes=f"trees={trees}, leaves={leaves}, n_features={n_features}, n_classes={n_classes}"
        )

    if "LSTM" in trained:
        m = trained["LSTM"]["model"]
        p = torch_trainable_params(m)
        sz = torch_model_size_bytes(m)
        add_model_stats_row(
            src=src,
            building=building,
            label_mode=label_mode,
            tag_base=tag_base,
            model_name="LSTM",
            family="Sequence",
            architecture="2×LSTM + linear classifier (last step to logits); dropout 0.2.",
            hidden_col=str(m.lstm.hidden_size),
            param_count=p,
            model_size_bytes=sz,
            table_size_text=f"{p:,} params",
            notes=f"input_dim={m.lstm.input_size}, hidden={m.lstm.hidden_size}, layers={m.lstm.num_layers}, n_classes={n_classes}"
        )

    if "CNN-LSTM" in trained:
        m = trained["CNN-LSTM"]["model"]
        p = torch_trainable_params(m)
        sz = torch_model_size_bytes(m)
        add_model_stats_row(
            src=src,
            building=building,
            label_mode=label_mode,
            tag_base=tag_base,
            model_name="CNN-LSTM",
            family="Sequence",
            architecture="Conv1D (32 channels, kernel size 3) + 1×LSTM + linear classifier.",
            hidden_col=str(m.lstm.hidden_size),
            param_count=p,
            model_size_bytes=sz,
            table_size_text=f"{p:,} params",
            notes=f"input_dim={m.cnn.in_channels}, conv_channels={m.cnn.out_channels}, hidden={m.lstm.hidden_size}, n_classes={n_classes}"
        )

    if "Informer" in trained:
        m = trained["Informer"]["model"]
        p = torch_trainable_params(m)
        sz = torch_model_size_bytes(m)
        d_model = m.in_proj.out_features
        num_layers = len(m.encoder.layers)
        nhead = m.encoder.layers[0].self_attn.num_heads
        ff_dim = m.encoder.layers[0].linear1.out_features
        add_model_stats_row(
            src=src,
            building=building,
            label_mode=label_mode,
            tag_base=tag_base,
            model_name="Informer",
            family="Sequence",
            architecture=f"Input projection to d={d_model} + {num_layers}×Transformer encoder ({nhead} heads) + linear classifier.",
            hidden_col=str(d_model),
            param_count=p,
            model_size_bytes=sz,
            table_size_text=f"{p:,} params",
            notes=f"input_dim={m.in_proj.in_features}, d_model={d_model}, nhead={nhead}, layers={num_layers}, ff_dim={ff_dim}, n_classes={n_classes}"
        )


def export_latex_rows_for_building(stats_df, source_dataset, building, out_path=None):
    """
    Export formatted LaTeX table rows for one building.
    """
    order = ["LinearSVM", "RF", "XGBoost", "LSTM", "CNN-LSTM", "Informer"]
    display_name = {
        "LinearSVM": "Linear SVM",
        "RF": "Random Forest",
        "XGBoost": "XGBoost",
        "LSTM": "LSTM",
        "CNN-LSTM": "CNN--LSTM",
        "Informer": "Informer-style encoder",
    }

    dfx = stats_df[
        (stats_df["source_dataset"] == source_dataset) &
        (stats_df["building"] == building)
    ].copy()

    lines = []
    for m in order:
        sub = dfx[dfx["model"] == m]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        line = (
            f'{display_name[m]} & {r["family"]} & {r["architecture"]} & '
            f'{r["hidden_col"]} & {r["table_size_text"]} \\\\'
        )
        lines.append(line)

    latex_rows = "\n".join(lines)

    if out_path is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex_rows)

    return latex_rows


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
    Build family-to-id and id-to-family mappings from a set of labels.
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
# Test-set degradations
# ------------------------------------------------------------
def degrade_noise(X, lvl):
    """
    Add Gaussian noise scaled by the per-feature standard deviation.
    """
    return X + np.random.normal(0, np.std(X, axis=0, keepdims=True) * lvl, size=X.shape)


def degrade_drift(X, lvl):
    """
    Add a linear drift over time, scaled by the per-feature mean.
    """
    t = np.linspace(0, 1, X.shape[0]).reshape(-1, 1)
    return X + (np.mean(X, axis=0, keepdims=True) * lvl) * t


def degrade_bias(X, lvl):
    """
    Add a constant bias offset to every row.
    """
    return X + (np.mean(X, axis=0, keepdims=True) * lvl)


def degrade_missing(X, lvl):
    """
    Randomly mask values as missing, then forward/backward fill them.
    """
    Xm = X.copy()
    mask = np.random.rand(*Xm.shape) < lvl
    Xm[mask] = np.nan
    Xm = pd.DataFrame(Xm).ffill().bfill().to_numpy()
    return Xm


def degrade_sampling(X, y, step):
    """
    Simulate reduced sampling by taking every n-th row.
    """
    return X[::step], y[::step]


manips = {
    "noise": degrade_noise,
    "drift": degrade_drift,
    "bias": degrade_bias,
    "missing": degrade_missing,
    "sampling": degrade_sampling,
}


# ------------------------------------------------------------
# Sequence models
# ------------------------------------------------------------
class TinyLSTM(nn.Module):
    """
    Lightweight LSTM classifier using the final time step output.
    """
    def __init__(self, input_dim, num_classes, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, dropout=dropout, batch_first=True)
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
            channels, hidden, num_layers=layers,
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
    Lightweight Transformer-encoder classifier in an Informer-style setup.

    This version uses:
    - linear input projection
    - transformer encoder blocks
    - final-step classification
    """
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


@torch.no_grad()
def torch_predict_labels(model, X, batch_size=512):
    """
    Predict class labels for a sequence model in batches.
    """
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size]).float().to(DEVICE, non_blocking=True)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_torch_classifier_fit(
    model,
    Xtr, ytr,
    Xval=None, yval=None,
    epochs=10, batch_size=512, lr=5e-4,
    class_weights=None, grad_clip_norm=1.0
):
    """
    Train a PyTorch classifier with optional validation selection by macro-F1.
    """
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr)

    if class_weights is None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        w = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=w)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long()),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    best_state = None
    best_score = -1e9

    for ep in range(epochs):
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

        if (Xval is not None) and (yval is not None) and (len(Xval) > 0):
            pred_val = torch_predict_labels(model, Xval, batch_size=batch_size)
            score = f1_score(yval, coerce_pred_labels(pred_val), average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model


# ------------------------------------------------------------
# SAGE-style Shapley global feature importance
# ------------------------------------------------------------
def metric_macro_f1(y_true, y_pred):
    """
    Macro-F1 scoring function for feature importance evaluation.
    """
    y_pred = coerce_pred_labels(y_pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def metric_accuracy(y_true, y_pred):
    """
    Accuracy scoring function for feature importance evaluation.
    """
    y_pred = coerce_pred_labels(y_pred)
    return accuracy_score(y_true, y_pred)


def get_fi_metric_fn():
    """
    Return the configured evaluation metric for feature importance.
    """
    return metric_accuracy if FI_METRIC == "accuracy" else metric_macro_f1


def _subsample_xy_tabular(X, y, max_n, rng):
    """
    Optional random subsampling for tabular importance computation.
    """
    if max_n is None or len(X) <= max_n:
        return X, y
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


def _subsample_xy_seq(Xseq, yseq, max_n, rng):
    """
    Optional random subsampling for sequence importance computation.
    """
    if max_n is None or len(Xseq) <= max_n:
        return Xseq, yseq
    idx = rng.choice(len(Xseq), size=max_n, replace=False)
    return Xseq[idx], yseq[idx]


def _mask_all_features_tabular(X, X_train, feat_idx, rng, masking="sample"):
    """
    Mask all selected tabular features either by:
    - replacing with training-set column mean
    - replacing with sampled values from the training distribution
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
        raise ValueError("FI_SAGE_MASKING must be one of: 'sample', 'mean'")
    return Xm


def _mask_all_features_sequence(Xseq, X_train_tab, feat_idx, rng, masking="sample"):
    """
    Mask all selected sequence features across all time steps either by:
    - replacing with training-set column mean
    - replacing with sampled values from the training distribution
    """
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
    """
    Monte Carlo SAGE-style global feature importance for tabular models.
    """
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
    """
    Monte Carlo SAGE-style global feature importance for sequence models.
    """
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


# ============================================================
# Dataset A: LBNL DataFDD synthesis inventory
# ============================================================
def clean_fault_name(s):
    """
    Normalize fault names by removing trailing bracketed descriptions.
    """
    s = str(s).strip()
    if "(" in s:
        return s.split("(")[0].strip()
    return s


def merge_lbnl_sensor_fault(sensor_df, fault_df):
    """
    Merge sensor measurements with fault intervals for the LBNL dataset.

    Each sensor row is assigned a raw fault label according to the fault time ranges.
    Additional family labels are also derived.
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
            t_raw.replace("TO", "to").replace("To", "to")
            .replace(" - ", " to ").replace("—", " to ").replace("-", " to ")
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
    sensors = sensors[:min(MAX_SENSOR_COLS, len(sensors))]

    df[sensors] = df[sensors].ffill().bfill()
    meta = {"inv_fault_map": inv, "inv_family_map": id2fam}
    return df, sensors, meta


def load_dataset_lbnl():
    """
    Load all configured LBNL buildings and return them as pipeline items.
    """
    RAW = Path("./data_FDD/5_lbnl_data_synthesis_inventory/raw/")
    file_map = {
        "RTU":       {"sensor": RAW / "RTU.csv",       "faults": RAW / "RTU-faults.csv"},
        "SZCAV":     {"sensor": RAW / "SZCAV.csv",     "faults": RAW / "SZCAV-faults.csv"},
        "SZVAV":     {"sensor": RAW / "SZVAV.csv",     "faults": RAW / "SZVAV-faults.csv"},
        "MZVAV_1":   {"sensor": RAW / "MZVAV-1.csv",   "faults": RAW / "MZVAV-1-faults.csv"},
        "MZVAV_2_1": {"sensor": RAW / "MZVAV-2-1.csv", "faults": RAW / "MZVAV-2-1-faults.csv"},
        "MZVAV_2_2": {"sensor": RAW / "MZVAV-2-2.csv", "faults": RAW / "MZVAV-2-2-faults.csv"},
    }

    items = []
    log_header("Loading dataset A: LBNL DataFDD synthesis inventory")

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
        log_kv(
            "Faults(raw)",
            list(meta["inv_fault_map"].values())[:10] + (["..."] if len(meta["inv_fault_map"]) > 10 else [])
        )
        log_kv("Fault families", list(meta["inv_family_map"].values()))

    return items


# ============================================================
# Dataset B: Nature LCU Wang
# ============================================================
def preprocess_wang_building(df):
    """
    Preprocess one Wang dataset building:
    - create timestamp
    - encode raw fault labels
    - derive fault families
    - select safe numeric sensor columns
    """
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

    exclude = ["timestamp", "Time", "DATE", "AHU name", "labeling", "FaultCode", "FaultFamily", "FaultFamilyCode"]
    sensors = sensor_cols_generic(df, exclude=exclude)
    sensors = sanitize_sensors(df, sensors)
    assert_no_leakage(sensors)
    sensors = sensors[:min(MAX_SENSOR_COLS, len(sensors))]

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df[sensors] = df[sensors].ffill().bfill()

    meta = {"inv_fault_map": inv_map, "inv_family_map": id2fam}
    return df, sensors, meta


def load_dataset_wang():
    """
    Load the Wang dataset buildings and return them as pipeline items.
    """
    ROOT = Path("./data_FDD/8_nature_lcu_wang/raw/")
    log_header("Loading dataset B: Nature LCU Wang")

    datasets = {
        "auditorium": pd.read_csv(ROOT / "auditorium_scientific_data.csv"),
        "office":     pd.read_csv(ROOT / "office_scientific_data.csv"),
        "hospital":   pd.read_csv(ROOT / "hosptial_scientific_data.csv"),
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
        log_kv("Fault families", list(meta["inv_family_map"].values()))

    return items


# ============================================================
# Per-building runner: train once, test on many corruptions
# ============================================================
def run_one_building_train_once(item):
    """
    Run the complete train-once / test-many evaluation for one building.

    Workflow:
    - choose raw or family labels
    - split train/test
    - scale features using training data only
    - train tabular models
    - train sequence models
    - evaluate on clean and corrupted test variants
    - store metrics, confusion matrices, and optional feature importance
    """
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

    X_all_raw = df[sensors].to_numpy().astype(np.float32)
    y_all = df[y_col].to_numpy().astype(int)

    if "Datetime" in df.columns:
        time_col = "Datetime"
    elif "timestamp" in df.columns:
        time_col = "timestamp"
    else:
        raise ValueError("No time column found for temporal split.")

    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------
    if SPLIT_MODE == "time":
        df = df.sort_values(time_col).reset_index(drop=True)
        split = int(len(df) * (1 - TEST_SIZE))

        left_end = max(0, split - SEQUENCE_GAP_ROWS)
        right_start = min(len(df), split + SEQUENCE_GAP_ROWS)

        df_train = df.iloc[:left_end].copy().reset_index(drop=True)
        df_test = df.iloc[right_start:].copy().reset_index(drop=True)

        Xtr_raw = df_train[sensors].to_numpy().astype(np.float32)
        ytr_raw = df_train[y_col].to_numpy().astype(int)
        Xte_raw = df_test[sensors].to_numpy().astype(np.float32)
        yte_raw = df_test[y_col].to_numpy().astype(int)

    elif SPLIT_MODE == "stratified":
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = stratified_split_keep_singletons_in_train(
            X_all_raw, y_all, test_size=TEST_SIZE, seed=42, tag=f"{tag_base}__RAW_SPLIT"
        )
        df_train = None
        df_test = None

    elif SPLIT_MODE == "stratified_time":
        Xtr_raw, Xte_raw, ytr_raw, yte_raw = stratified_time_split(
            X_all_raw, y_all, test_size=TEST_SIZE, min_train_per_class=10, min_test_per_class=5
        )
        df_train = None
        df_test = None

    elif SPLIT_MODE == "episode_stratified_time":
        df_train, df_test, episodes, by_label = stratified_episode_time_split(
            df=df,
            y_col=y_col,
            time_col=time_col,
            test_size=TEST_SIZE,
            min_train_episodes_per_class=EPISODE_MIN_TRAIN_EPISODES_PER_CLASS,
            min_test_episodes_per_class=EPISODE_MIN_TEST_EPISODES_PER_CLASS,
            gap_rows=SEQUENCE_GAP_ROWS,
            tag=f"{tag_base}__EPISODE_SPLIT"
        )

        Xtr_raw = df_train[sensors].to_numpy().astype(np.float32)
        ytr_raw = df_train[y_col].to_numpy().astype(int)
        Xte_raw = df_test[sensors].to_numpy().astype(np.float32)
        yte_raw = df_test[y_col].to_numpy().astype(int)

        print_episode_split_summary(df_train, df_test, y_col, inv_map, name=tag_base)

    else:
        raise ValueError("SPLIT_MODE must be one of: 'time', 'stratified', 'stratified_time', 'episode_stratified_time'")

    if len(Xtr_raw) == 0 or len(Xte_raw) == 0:
        raise ValueError(f"Empty train/test split after applying {SPLIT_MODE} for {tag_base}")

    log_sub(f"{tag_base} | Split")
    log_kv("Train RAW", Xtr_raw.shape)
    log_kv("Test  RAW", Xte_raw.shape)
    log_kv("n_classes", n_classes)
    log_kv("Exact rounded row overlap", exact_overlap_count(Xtr_raw, Xte_raw))

    if df_train is not None and df_test is not None:
        log_kv("Train time range", f"{df_train[time_col].min()} -> {df_train[time_col].max()}")
        log_kv("Test time range", f"{df_test[time_col].min()} -> {df_test[time_col].max()}")

    print_class_balance(ytr_raw, inv_map, "TRAIN")
    print_class_balance(yte_raw, inv_map, "TEST")

    train_classes = set(np.unique(ytr_raw).tolist())
    test_classes = set(np.unique(yte_raw).tolist())
    missing_in_train = sorted(list(test_classes - train_classes))
    if len(missing_in_train) > 0:
        log_kv("Classes in TEST but missing in TRAIN", [inv_map[i] for i in missing_in_train])

    scaler = StandardScaler()
    scaler.fit(Xtr_raw)

    Xtr_clean = pd.DataFrame(Xtr_raw).ffill().bfill().to_numpy()
    Xtr_m = scaler.transform(Xtr_clean)

    Xtr_tab, ytr_tab = cap_tabular(Xtr_m, ytr_raw, MAX_TAB_TRAIN, seed=42)
    if len(Xtr_tab) != len(Xtr_m):
        log_kv("Tabular cap", f"{len(Xtr_m)} -> {len(Xtr_tab)}")

    # --------------------------------------------------------
    # Internal validation split
    # --------------------------------------------------------
    # For row-wise stratified splitting, use another stratified random split.
    # For time-aware splitting, keep temporal order and use the tail of train as validation.
    if SPLIT_MODE == "stratified":
        Xtr_fit, ytr_fit, Xval_fit, yval_fit = stratified_train_val_keep_singletons_in_train(
            Xtr_tab, ytr_tab, val_fraction=VAL_FRACTION, seed=42, tag=f"{tag_base}__TAB_VAL"
        )
    else:
        n_val = int(len(Xtr_tab) * VAL_FRACTION)
        if (not USE_INTERNAL_VAL) or n_val < 50:
            Xtr_fit, ytr_fit, Xval_fit, yval_fit = Xtr_tab, ytr_tab, None, None
        else:
            Xtr_fit, ytr_fit = Xtr_tab[:-n_val], ytr_tab[:-n_val]
            Xval_fit, yval_fit = Xtr_tab[-n_val:], ytr_tab[-n_val:]

    cw_fit = compute_balanced_weights(ytr_fit, n_classes)
    sw_fit = compute_sample_weights(ytr_fit, cw_fit)

    trained = {}

    # --------------------------------------------------------
    # Train tabular models
    # --------------------------------------------------------
    log_sub(f"{tag_base} | Train once (tabular)")

    t0 = time.time()
    svm = LinearSVC(C=1.0, dual=False, max_iter=8000, class_weight="balanced")
    svm.fit(Xtr_fit, ytr_fit)
    trained["LinearSVM"] = {"model": svm}
    log_kv("Trained LinearSVM (s)", time.time() - t0)

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
    trained["RF"] = {"model": rf}
    log_kv("Trained RF (s)", time.time() - t0)

    if USE_XGBOOST:
        t0 = time.time()
        tree_method = "hist"
        if XGB_TREE_METHOD == "gpu_hist" and torch.cuda.is_available():
            tree_method = "gpu_hist"

        le = LabelEncoder()
        ytr_xgb = le.fit_transform(ytr_fit)
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
            trained["XGBoost"] = {"model": xgb_model, "label_encoder": le, "K": K}
            log_kv("Trained XGBoost (s)", time.time() - t0)

    # --------------------------------------------------------
    # Train sequence models
    # --------------------------------------------------------
    log_sub(f"{tag_base} | Train once (sequence)")
    Xtr_seq, ytr_seq = create_sequences(Xtr_m, ytr_raw, seq_len=SEQ_LEN, stride=SEQ_STRIDE)

    if len(Xtr_seq) < 50:
        log_kv("Skipping sequence models", f"too few train sequences (train={len(Xtr_seq)})")
        seq_ok = False
    else:
        seq_ok = True
        Xtr_seq, ytr_seq = cap_sequences(Xtr_seq, ytr_seq, MAX_SEQ_TRAIN, seed=42)
        cw_seq = compute_balanced_weights(ytr_seq, n_classes)

        if SPLIT_MODE == "stratified":
            Xtr_seq_fit, ytr_seq_fit, Xval_seq, yval_seq = stratified_train_val_keep_singletons_in_train(
                Xtr_seq, ytr_seq, val_fraction=VAL_FRACTION, seed=42, tag=f"{tag_base}__SEQ_VAL"
            )
        else:
            n_val = int(len(Xtr_seq) * VAL_FRACTION)
            if (not USE_INTERNAL_VAL) or n_val < 50:
                Xtr_seq_fit, ytr_seq_fit, Xval_seq, yval_seq = Xtr_seq, ytr_seq, None, None
            else:
                Xtr_seq_fit, ytr_seq_fit = Xtr_seq[:-n_val], ytr_seq[:-n_val]
                Xval_seq, yval_seq = Xtr_seq[-n_val:], ytr_seq[-n_val:]

        t0 = time.time()
        m = TinyLSTM(input_dim=Xtr_seq_fit.shape[2], num_classes=n_classes)
        m = train_torch_classifier_fit(
            m, Xtr_seq_fit, ytr_seq_fit,
            Xval=Xval_seq, yval=yval_seq,
            epochs=EPOCHS, batch_size=BATCH, lr=LR,
            class_weights=cw_seq, grad_clip_norm=GRAD_CLIP_NORM
        )
        trained["LSTM"] = {"model": m}
        log_kv("Trained LSTM (s)", time.time() - t0)

        t0 = time.time()
        m = TinyCNNLSTM(input_dim=Xtr_seq_fit.shape[2], num_classes=n_classes)
        m = train_torch_classifier_fit(
            m, Xtr_seq_fit, ytr_seq_fit,
            Xval=Xval_seq, yval=yval_seq,
            epochs=EPOCHS, batch_size=BATCH, lr=LR,
            class_weights=cw_seq, grad_clip_norm=GRAD_CLIP_NORM
        )
        trained["CNN-LSTM"] = {"model": m}
        log_kv("Trained CNN-LSTM (s)", time.time() - t0)

        t0 = time.time()
        m = TinyInformerClassifier(input_dim=Xtr_seq_fit.shape[2], num_classes=n_classes)
        m = train_torch_classifier_fit(
            m, Xtr_seq_fit, ytr_seq_fit,
            Xval=Xval_seq, yval=yval_seq,
            epochs=EPOCHS, batch_size=BATCH, lr=LR,
            class_weights=cw_seq, grad_clip_norm=GRAD_CLIP_NORM
        )
        trained["Informer"] = {"model": m}
        log_kv("Trained Informer (s)", time.time() - t0)

    # --------------------------------------------------------
    # Collect model statistics
    # --------------------------------------------------------
    collect_model_stats_for_building(
        src=src,
        building=building,
        label_mode=LABEL_MODE,
        tag_base=tag_base,
        trained=trained,
        n_features=len(sensors),
        n_classes=n_classes
    )

    # --------------------------------------------------------
    # Evaluate on clean and degraded test sets
    # --------------------------------------------------------
    rows = []
    metric_fn = get_fi_metric_fn()
    fi_method = f"sage_mc_{FI_SAGE_MASKING}_{FI_METRIC}_mean_std__perms={FI_SAGE_N_PERMS}"

    X_train_for_fi_tab = Xtr_fit

    for manip_name, lvls in levels.items():
        for lvl in lvls:
            log_header(f"{tag_base} | Evaluation | manipulation={manip_name} | level={lvl}")

            Xte_raw_m = Xte_raw.copy()
            yte_m = yte_raw.copy()

            if manip_name == "clean":
                pass
            elif manip_name == "sampling":
                Xte_raw_m, yte_m = degrade_sampling(Xte_raw_m, yte_m, lvl)
            else:
                Xte_raw_m = manips[manip_name](Xte_raw_m, lvl)

            Xte_raw_m = pd.DataFrame(Xte_raw_m).ffill().bfill().to_numpy()
            Xte_m = scaler.transform(Xte_raw_m)

            log_kv("Test after corruption (RAW)", Xte_raw_m.shape)
            log_kv("Test after scaling", Xte_m.shape)

            # ------------------------------------------------
            # LinearSVM
            # ------------------------------------------------
            t0 = time.time()
            svm = trained["LinearSVM"]["model"]
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
                    X_train=X_train_for_fi_tab,
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
                        title=f"{tag_base} | LinearSVM | Feature importance ({manip_name}={lvl})",
                        topk=FI_TOPK, show_inline=True
                    )

            rows.append({
                "source_dataset": src,
                "building": building,
                "label_mode": LABEL_MODE,
                "manipulation": manip_name,
                "level": lvl,
                "corrupt_where": CORRUPT_WHERE,
                "split_mode": SPLIT_MODE,
                "model": "LinearSVM",
                "accuracy": acc,
                "macro_precision": mp,
                "macro_recall": mr,
                "macro_f1": mf1,
                "runtime_s": rt
            })

            # ------------------------------------------------
            # Random Forest
            # ------------------------------------------------
            t0 = time.time()
            rf = trained["RF"]["model"]
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
                    X_train=X_train_for_fi_tab,
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
                        title=f"{tag_base} | RF | Feature importance ({manip_name}={lvl})",
                        topk=FI_TOPK, show_inline=True
                    )

            rows.append({
                "source_dataset": src,
                "building": building,
                "label_mode": LABEL_MODE,
                "manipulation": manip_name,
                "level": lvl,
                "corrupt_where": CORRUPT_WHERE,
                "split_mode": SPLIT_MODE,
                "model": "RF",
                "accuracy": acc,
                "macro_precision": mp,
                "macro_recall": mr,
                "macro_f1": mf1,
                "runtime_s": rt
            })

            # ------------------------------------------------
            # XGBoost
            # ------------------------------------------------
            if USE_XGBOOST and ("XGBoost" in trained):
                t0 = time.time()
                xgb_model = trained["XGBoost"]["model"]
                le = trained["XGBoost"]["label_encoder"]

                pred_local = coerce_pred_labels(xgb_model.predict(Xte_m))
                pred = le.inverse_transform(pred_local)

                rt = time.time() - t0
                acc, mp, mr, mf1 = log_metrics_block("XGBoost", manip_name, lvl, yte_m, pred)
                log_kv("Runtime (s)", rt)

                cm_raw = confusion_matrix(yte_m, pred, labels=np.arange(n_classes))
                cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
                add_confmat_rows(src, building, LABEL_MODE, tag_base, "XGBoost", manip_name, lvl, class_names, cm_raw, cm_norm)
                if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                    show_confmat(cm_norm, class_names, title=f"{tag_base} | XGBoost | {manip_name}={lvl}", show_inline=True)

                if COMPUTE_FEATURE_IMPORTANCE:
                    mean_imp, std_imp = sage_importance_tabular(
                        predict_fn=lambda Z: le.inverse_transform(coerce_pred_labels(xgb_model.predict(Z))),
                        X_test=Xte_m, y_test=yte_m,
                        X_train=X_train_for_fi_tab,
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
                            title=f"{tag_base} | XGBoost | Feature importance ({manip_name}={lvl})",
                            topk=FI_TOPK, show_inline=True
                        )

                rows.append({
                    "source_dataset": src,
                    "building": building,
                    "label_mode": LABEL_MODE,
                    "manipulation": manip_name,
                    "level": lvl,
                    "corrupt_where": CORRUPT_WHERE,
                    "split_mode": SPLIT_MODE,
                    "model": "XGBoost",
                    "accuracy": acc,
                    "macro_precision": mp,
                    "macro_recall": mr,
                    "macro_f1": mf1,
                    "runtime_s": rt
                })

            # ------------------------------------------------
            # Sequence models
            # ------------------------------------------------
            if seq_ok and ("LSTM" in trained):
                Xte_seq, yte_seq = create_sequences(Xte_m, yte_m, seq_len=SEQ_LEN, stride=SEQ_STRIDE)

                if len(Xte_seq) < 50:
                    log_kv("Skipping sequence evaluation", f"too few test sequences (test={len(Xte_seq)})")
                else:
                    Xte_seq, yte_seq = cap_sequences(Xte_seq, yte_seq, MAX_SEQ_TEST, seed=43)

                    # LSTM
                    t0 = time.time()
                    m = trained["LSTM"]["model"]
                    pred = coerce_pred_labels(torch_predict_labels(m, Xte_seq, batch_size=BATCH))
                    rt = time.time() - t0
                    acc, mp, mr, mf1 = log_metrics_block("LSTM", manip_name, lvl, yte_seq, pred)
                    log_kv("Runtime (s)", rt)

                    cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
                    cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
                    add_confmat_rows(src, building, LABEL_MODE, tag_base, "LSTM", manip_name, lvl, class_names, cm_raw, cm_norm)
                    if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                        show_confmat(cm_norm, class_names, title=f"{tag_base} | LSTM | {manip_name}={lvl}", show_inline=True)

                    rows.append({
                        "source_dataset": src,
                        "building": building,
                        "label_mode": LABEL_MODE,
                        "manipulation": manip_name,
                        "level": lvl,
                        "corrupt_where": CORRUPT_WHERE,
                        "split_mode": SPLIT_MODE,
                        "model": "LSTM",
                        "accuracy": acc,
                        "macro_precision": mp,
                        "macro_recall": mr,
                        "macro_f1": mf1,
                        "runtime_s": rt
                    })

                    # CNN-LSTM
                    t0 = time.time()
                    m = trained["CNN-LSTM"]["model"]
                    pred = coerce_pred_labels(torch_predict_labels(m, Xte_seq, batch_size=BATCH))
                    rt = time.time() - t0
                    acc, mp, mr, mf1 = log_metrics_block("CNN-LSTM", manip_name, lvl, yte_seq, pred)
                    log_kv("Runtime (s)", rt)

                    cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
                    cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
                    add_confmat_rows(src, building, LABEL_MODE, tag_base, "CNN-LSTM", manip_name, lvl, class_names, cm_raw, cm_norm)
                    if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                        show_confmat(cm_norm, class_names, title=f"{tag_base} | CNN-LSTM | {manip_name}={lvl}", show_inline=True)

                    rows.append({
                        "source_dataset": src,
                        "building": building,
                        "label_mode": LABEL_MODE,
                        "manipulation": manip_name,
                        "level": lvl,
                        "corrupt_where": CORRUPT_WHERE,
                        "split_mode": SPLIT_MODE,
                        "model": "CNN-LSTM",
                        "accuracy": acc,
                        "macro_precision": mp,
                        "macro_recall": mr,
                        "macro_f1": mf1,
                        "runtime_s": rt
                    })

                    # Informer
                    t0 = time.time()
                    m = trained["Informer"]["model"]
                    pred = coerce_pred_labels(torch_predict_labels(m, Xte_seq, batch_size=BATCH))
                    rt = time.time() - t0
                    acc, mp, mr, mf1 = log_metrics_block("Informer", manip_name, lvl, yte_seq, pred)
                    log_kv("Runtime (s)", rt)

                    cm_raw = confusion_matrix(yte_seq, pred, labels=np.arange(n_classes))
                    cm_norm = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1)
                    add_confmat_rows(src, building, LABEL_MODE, tag_base, "Informer", manip_name, lvl, class_names, cm_raw, cm_norm)
                    if should_plot(manip_name, lvl) and SHOW_HEATMAP_INLINE:
                        show_confmat(cm_norm, class_names, title=f"{tag_base} | Informer | {manip_name}={lvl}", show_inline=True)

                    rows.append({
                        "source_dataset": src,
                        "building": building,
                        "label_mode": LABEL_MODE,
                        "manipulation": manip_name,
                        "level": lvl,
                        "corrupt_where": CORRUPT_WHERE,
                        "split_mode": SPLIT_MODE,
                        "model": "Informer",
                        "accuracy": acc,
                        "macro_precision": mp,
                        "macro_recall": mr,
                        "macro_f1": mf1,
                        "runtime_s": rt
                    })

    return rows


# ============================================================
# Run all buildings
# ============================================================
log_header("Configuration summary")
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
log_kv("EPISODE_MIN_TRAIN_EPISODES_PER_CLASS", EPISODE_MIN_TRAIN_EPISODES_PER_CLASS)
log_kv("EPISODE_MIN_TEST_EPISODES_PER_CLASS", EPISODE_MIN_TEST_EPISODES_PER_CLASS)
log_kv("SEQUENCE_GAP_ROWS", SEQUENCE_GAP_ROWS)

items = []
items += load_dataset_lbnl()
items += load_dataset_wang()

log_header(f"Total buildings to run: {len(items)}")

all_results = []

for bi, item in enumerate(items, start=1):
    tag = f"{item['source_dataset']}__{item['building']}"
    log_header(f"Run building {bi}/{len(items)} | {tag}")

    try:
        rows = run_one_building_train_once(item)
        all_results.extend(rows)

    except Exception as e:
        print("\n" + "!" * 110)
        print(f"FAILED building: {tag}")
        print(f"Exception: {type(e).__name__}: {e}")
        print("Continuing with next building...")
        print("!" * 110)

    # Incremental save after each building
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUT / "results_all.csv", index=False)

    fi_all = pd.DataFrame(FEATURE_IMPORTANCE_ROWS)
    fi_all.to_csv(OUT / "feature_importance_all.csv", index=False)

    cm_all = pd.DataFrame(CONFMAT_ROWS)
    cm_all.to_csv(OUT / "confmat_all.csv", index=False)

    stats_all = pd.DataFrame(MODEL_STATS_ROWS)
    stats_all.to_csv(OUT / "model_stats_all.csv", index=False)

log_header("Final outputs saved")
log_kv("Results rows", len(pd.DataFrame(all_results)))
log_kv("Saved results_all.csv", str(OUT / "results_all.csv"))
log_kv("Feature importance rows", len(pd.DataFrame(FEATURE_IMPORTANCE_ROWS)))
log_kv("Saved feature_importance_all.csv", str(OUT / "feature_importance_all.csv"))
log_kv("Confusion matrix rows", len(pd.DataFrame(CONFMAT_ROWS)))
log_kv("Saved confmat_all.csv", str(OUT / "confmat_all.csv"))
log_kv("Model stats rows", len(pd.DataFrame(MODEL_STATS_ROWS)))
log_kv("Saved model_stats_all.csv", str(OUT / "model_stats_all.csv"))

if len(all_results):
    res_df = pd.DataFrame(all_results)
    log_kv("Datasets", sorted(res_df["source_dataset"].unique()))
    log_kv("Buildings", sorted(res_df["building"].unique()))
    log_kv("Models", sorted(res_df["model"].unique()))
    log_kv("Manipulations", sorted(res_df["manipulation"].unique()))

stats_df = pd.DataFrame(MODEL_STATS_ROWS)
if len(stats_df):
    print("\nModel statistics preview:")
    cols_show = [
        "source_dataset", "building", "model",
        "param_count", "node_count", "leaf_count", "tree_count",
        "model_size_human", "table_size_text"
    ]
    print(stats_df[cols_show].head(30).to_string(index=False))

    one_src = stats_df.iloc[0]["source_dataset"]
    one_bld = stats_df.iloc[0]["building"]
    latex_rows = export_latex_rows_for_building(
        stats_df,
        source_dataset=one_src,
        building=one_bld,
        out_path=OUT / "model_table_rows.tex"
    )
    print("\nSaved LaTeX rows for one building to:", OUT / "model_table_rows.tex")
    print(latex_rows)

print("\nOutput folder contents:")
for p in sorted(OUT.glob("*")):
    print(" -", p.name)