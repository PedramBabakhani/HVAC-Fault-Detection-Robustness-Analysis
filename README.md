```markdown
# Unified Fault Detection Notebook Cell — README (Big, Detailed, and Practical)

This README documents a **single notebook cell** that runs a **unified fault detection benchmark** across **two public HVAC fault datasets**, multiple **tabular + sequence models**, multiple **robustness degradations**, and a **single scientific feature-importance method for all models**:  
✅ **SAGE-style Shapley Global Importance (Monte Carlo)**.

The goals are:

- **No data leakage**
- **Same evaluation protocol for every model**
- **Same feature-importance method for every model**
- **Time-series “honesty”** (we do **not** shuffle timesteps)
- **Repeatable + scalable runtime** using caps/knobs

---

## Table of Contents

1. [What this code does](#what-this-code-does)
2. [Datasets](#datasets)
   - [Dataset A: LBNL DataFDD synthesis inventory](#dataset-a-lbnl-datafdd-synthesis-inventory)
   - [Dataset B: Nature LCU Wang](#dataset-b-nature-lcu-wang)
3. [Labels and fault-family unification](#labels-and-fault-family-unification)
4. [Models](#models)
   - [Tabular models (CPU)](#tabular-models-cpu)
   - [Sequence models (GPU)](#sequence-models-gpu)
5. [Train/test split strategy (no leakage)](#traintest-split-strategy-no-leakage)
6. [Scaling and preprocessing](#scaling-and-preprocessing)
7. [Robustness benchmark (degradations)](#robustness-benchmark-degradations)
8. [Universal feature importance: SAGE-style Shapley global importance](#universal-feature-importance-sage-style-shapley-global-importance)
   - [What it measures](#what-it-measures)
   - [Why it’s “fair” across all models](#why-its-fair-across-all-models)
   - [How “missing feature” is defined](#how-missing-feature-is-defined)
   - [Sequence safety: no shuffling time](#sequence-safety-no-shuffling-time)
   - [Why importance can be negative](#why-importance-can-be-negative)
   - [Uncertainty: mean ± std](#uncertainty-mean--std)
9. [Outputs](#outputs)
   - [results_all.csv](#results_allcsv)
   - [feature_importance_all.csv](#feature_importance_allcsv)
   - [confmat_all.csv](#confmat_allcsv)
10. [Plots shown inline](#plots-shown-inline)
11. [Config guide (all knobs explained)](#config-guide-all-knobs-explained)
12. [Runtime guidance](#runtime-guidance)
13. [Reproducibility and seeds](#reproducibility-and-seeds)
14. [Common pitfalls and troubleshooting](#common-pitfalls-and-troubleshooting)
15. [Interpreting the results correctly](#interpreting-the-results-correctly)
16. [Extending the pipeline](#extending-the-pipeline)
17. [FAQ](#faq)

---

## What this code does

This notebook cell builds a **consistent evaluation harness** for fault classification from HVAC sensor data.

For each building/stream in each dataset, it:

1. Loads the data and merges sensors with fault intervals (LBNL) or uses labeled records (Wang).
2. Selects sensor columns, aggressively **removing leakage columns**.
3. Splits the time series into train/test.
4. Fits the scaler on train only and transforms train/test.
5. For each robustness manipulation (clean/noise/drift/bias/missing/sampling) and each level:
   - Corrupts **test only** by default (or train+test if configured).
   - Trains and evaluates **tabular models** on per-timestep data.
   - Trains and evaluates **sequence models** on rolling windows (length = `SEQ_LEN`).
   - Logs accuracy and F1.
   - Produces confusion matrices.
   - Computes feature importance with one universal method (**SAGE-style Shapley global importance**).
6. Aggregates everything into 3 CSVs in `./fdd_out`.

---

## Datasets

### Dataset A: LBNL DataFDD synthesis inventory

**Path expected:**
```

/kaggle/input/datafdd/5_lbnl_data_synthesis_inventory/raw/

```

**Files expected:**
- `MZVAV-1.csv`, `MZVAV-1-faults.csv`
- `MZVAV-2-1.csv`, `MZVAV-2-1-faults.csv`
- `MZVAV-2-2.csv`, `MZVAV-2-2-faults.csv`
- `RTU.csv`, `RTU-faults.csv`
- `SZCAV.csv`, `SZCAV-faults.csv`
- `SZVAV.csv`, `SZVAV-faults.csv`

**How labeling is built:**
- Sensor CSV has a `Datetime` column.
- Fault CSV lists fault name and time interval (string that is parsed into start/end).
- The code initializes all timesteps as `"Normal"`, then overlays fault intervals.

**Important detail:**
- Time parsing is robust to variations (`TO`, hyphens, etc.).
- Fault intervals that fail parsing are skipped.

---

### Dataset B: Nature LCU Wang

**Path expected:**
```

/kaggle/input/datafdd/8_nature_lcu_wang/raw/

```

**Files expected:**
- `auditorium_scientific_data.csv`
- `office_scientific_data.csv`
- `hosptial_scientific_data.csv` (note spelling)

**How labeling is built:**
- Uses the dataset’s `labeling` column as the raw label.
- Builds categorical codes.
- Adds a fault-family mapping (optional target).

**Timestamp parsing:**
- If the file has both `DATE` and `Time`, it merges them into `timestamp`.
- Else if only `Time`, it parses `timestamp` from `Time`.
- Rows without valid timestamps are dropped, and data is time-sorted.

---

## Labels and fault-family unification

You can switch between:

- `LABEL_MODE = "raw"`  
  Use dataset-specific fault labels (many classes possible)

- `LABEL_MODE = "family"`  
  Map faults into a smaller set of **unified fault families**:

```

NORMAL
SENSOR_BIAS
TEMP_FAULT
VALVE_FAULT
DAMPER_VENT_FAULT
FAN_PUMP_FAULT
EQUIPMENT_HEATX_FAULT
SCHEDULING_SETBACK_FAULT
OTHER

```

### Why do this?
Raw labels may be inconsistent across datasets. Family-mode provides:
- fewer classes
- more comparability across sources
- easier confusion matrix interpretation

### The mapping is rule-based
`map_fault_to_family(label)` uses substring logic:
- “bias” + “sensor/thermostat” → SENSOR_BIAS
- “valve” → VALVE_FAULT
- “damper” / “infiltration” → DAMPER_VENT_FAULT
- etc.

---

## Models

### Tabular models (CPU)

Trained on **individual time points** (rows), using sensor vector `X[t, :]`.

1. **LinearSVM** (`sklearn.svm.LinearSVC`)
   - fast linear classifier
   - no probability outputs used here; only hard predictions
2. **RandomForestClassifier**
   - tree ensemble, handles nonlinearity
3. **XGBClassifier**
   - boosted trees, often strong baseline

Tabular models are capped by:
- `MAX_TAB_TRAIN` to keep runtime stable

---

### Sequence models (GPU)

Trained on rolling windows of sensors:

- input sequence: `X[t-SEQ_LEN+1 : t, :]`
- target label: `y[t]` (label at the last timestep in the window)

Sequence models implemented:

1. **TinyLSTM**
   - 1 LSTM layer, hidden size 32
2. **TinyCNNLSTM**
   - 1D conv over time (feature channels) then LSTM
3. **TinyInformerClassifier** (Transformer encoder)
   - linear projection → transformer encoder → classify last step embedding

Sequence sampling is controlled by:
- `SEQ_LEN` (window length)
- `SEQ_STRIDE` (stride between windows)
- `MAX_SEQ_TRAIN`, `MAX_SEQ_TEST` (caps)

---

## Train/test split strategy (no leakage)

Two split modes:

### 1) `SPLIT_MODE="time"` (recommended)
- First `(1 - TEST_SIZE)` portion is train
- Last `TEST_SIZE` portion is test
- Prevents future data leaking into training

### 2) `SPLIT_MODE="stratified"`
- random split with `stratify=y`
- may mix time (not ideal for time series)
- but can be used if your dataset is not time-causal

---

## Scaling and preprocessing

Scaling is done with `StandardScaler`:

- `scaler.fit(X_train_raw)`
- `X_train = scaler.transform(...)`
- `X_test = scaler.transform(...)`

✅ This is correct leakage-free scaling.

Missing values:
- The code uses `ffill().bfill()` at dataset load time (before scaling).
- Degradation “missing” introduces NaNs and then uses forward/backfill too.

---

## Robustness benchmark (degradations)

This pipeline tests performance under 6 conditions:

1. **clean**
2. **noise**: add Gaussian noise proportional to feature std
3. **drift**: add linear drift over time
4. **bias**: add constant offset based on mean magnitude
5. **missing**: randomly drop values → ffill/bfill imputation
6. **sampling**: downsample the series (take every k-th point)

Control where corruption is applied:
- `CORRUPT_WHERE="test"` (default): only degrade test, measure generalization under stress
- `CORRUPT_WHERE="both"`: degrade both train and test, measure adaptation under degraded training

---

## Universal feature importance: SAGE-style Shapley global importance

### What it measures

For each feature (sensor), it estimates:

> **How much this feature contributes to the model’s predictive performance**, on a held-out test set, **averaged over all possible feature coalitions**.

Unlike “feature importance” from trees or coefficients from linear models, this is:
- model-agnostic
- performance-based
- directly comparable across models

### Why it’s fair across all models

Every model is evaluated with the same game:

- pick a set of features “available”
- treat the rest as “missing” using a consistent masking operator
- measure performance change using the same metric (`macro_f1` or `accuracy`)
- compute Shapley-style marginal contributions

So your CNN-LSTM and your XGBoost are graded using the **exact same definition** of “importance”.

---

### How “missing feature” is defined

This is the most critical scientific detail.

When features are “missing”, the code replaces them using the **training distribution**:

Two options via `FI_SAGE_MASKING`:

#### 1) `"sample"` (recommended)
- For each missing feature j, impute by sampling values from `X_train[:, j]`
- This approximates a marginal distribution baseline
- Avoids forcing everything to the mean
- Produces more realistic masking

#### 2) `"mean"`
- Replace missing feature j with the training mean
- Fast and simple
- Can be overly “clean” and sometimes optimistic/pessimistic depending on feature scaling

---

### Sequence safety: no shuffling time

Time series models are sensitive to temporal order.

This code **does not permute timesteps**.

For sequences:
- “masking a feature” means replacing the **entire feature trajectory** across all timesteps in the window:
```

X[:, :, j] = imputed_value

```
- That preserves temporal structure completely.

This is the correct concept of “missing sensor stream” rather than “random time scrambling”.

---

### Why importance can be negative

SAGE returns **signed contributions**.

A negative mean contribution for feature j means:

> On average, adding feature j makes the model perform worse (under this masking/imputation protocol).

This can happen if:
- feature is noisy and model overfits it
- feature is redundant and causes instability
- feature is correlated with a spurious signal
- imputation distribution interacts with model in non-intuitive ways
- data is small and variance is high

Negative values are not “wrong”; they’re a signal.

---

### Uncertainty: mean ± std

Because you approximate Shapley values using Monte Carlo permutations:

- `FI_SAGE_N_PERMS` = number of random feature orderings
- you get a distribution of contributions per feature
- you report:
- mean contribution
- std deviation across permutations

This is why your feature importance output is honest and scientific:
- you can see which importances are stable (small std)
- you can see which ones are uncertain (large std)

---

## Outputs

All outputs go to:
```

./fdd_out/

```

### `results_all.csv`

One row per:
- dataset
- building
- label_mode
- manipulation
- level
- model

Includes:
- accuracy
- macro_f1
- weighted_f1
- runtime

Use this file for overall benchmarking and plotting.

---

### `feature_importance_all.csv`

Long format table, one row per:
- dataset/building/model/manip/level
- feature

Includes:
- `importance_mean` (signed Shapley contribution)
- `importance_std` (uncertainty)
- `method` (string describing configuration)

This is your universal feature importance output.

---

### `confmat_all.csv`

Long format confusion matrix storage:
- stores both raw and normalized confusion matrices
- each row corresponds to a single cell (true_label, pred_label)

Fields:
- `normalized = 0` raw counts
- `normalized = 1` row-normalized proportions

---

## Plots shown inline

Depending on configuration:

- confusion matrix heatmaps (`SHOW_HEATMAP_INLINE`)
- feature importance bar plots (`SHOW_FI_INLINE`)

Plot reduction option:
- `PLOT_ONLY_CLEAN_AND_WORST=True`
  - plots `clean`
  - and only the maximum degradation level for each manipulation

This prevents drowning the notebook in hundreds of plots.

---

## Config guide (all knobs explained)

### Core runtime
- `DEVICE`: uses GPU if available
- `EPOCHS`, `BATCH`, `LR`: sequence model training
- `MAX_TAB_TRAIN`: cap training rows for tabular models
- `MAX_SEQ_TRAIN`, `MAX_SEQ_TEST`: cap sequences (windows)
- `SEQ_LEN`: window length
- `SEQ_STRIDE`: step between windows

### Evaluation
- `TEST_SIZE`: fraction of data used as test
- `SPLIT_MODE`: `"time"` (recommended) vs `"stratified"`
- `LABEL_MODE`: `"raw"` vs `"family"`

### Robustness
- `CORRUPT_WHERE`: `"test"` vs `"both"`
- `levels`: manipulation → list of intensities

### Feature importance (SAGE)
- `COMPUTE_FEATURE_IMPORTANCE`: toggle importance computation
- `FI_METRIC`: `"macro_f1"` (recommended for class imbalance) vs `"accuracy"`
- `FI_SAGE_N_PERMS`: MC permutations (20–50 usually ok)
- `FI_SAGE_MASKING`: `"sample"` recommended
- `MAX_FI_TEST_SAMPLES_TAB`: cap samples for importance
- `MAX_FI_TEST_SAMPLES_SEQ`: cap sequence windows for importance
- `MAX_FI_FEATURES`: reduce number of evaluated features for speed (optional)

---

## Runtime guidance

SAGE importance is expensive because it evaluates the model many times:

Roughly:

- Tabular SAGE cost per run:
```

O(n_perms * n_features * model_predict_cost)

```
- Sequence SAGE cost per run is larger because prediction cost is larger.

To keep it sane:
- keep `FI_SAGE_N_PERMS` around 20
- cap samples (`MAX_FI_TEST_SAMPLES_*`)
- optionally cap features (`MAX_FI_FEATURES`)

If it’s still heavy:
- set `COMPUTE_FEATURE_IMPORTANCE=False` for benchmarking
- then rerun only for clean + worst settings (or a single building)

---

## Reproducibility and seeds

- `FI_SEED` controls the permutation RNG for feature importance
- There are additional seeds used for sequence FI offsets (+999, +1999, +2999)
- Train/test split in `"time"` mode is deterministic
- `"stratified"` uses `random_state=42`

If you want strict reproducibility for Torch training:
- set torch seeds and deterministic flags (not included by default because it slows GPU)

---

## Common pitfalls and troubleshooting

### 1) “Too few sequences” and sequence models skip
If your dataset is small or sampling degradation reduces points, you may get:
- train windows < 50 or test windows < 50

Then sequence models are skipped for that condition.

Fix:
- reduce `SEQ_LEN`
- reduce `sampling` level max
- set `SEQ_STRIDE=1`
- increase dataset length

---

### 2) Missing files in Kaggle paths
If file paths don’t exist, buildings are skipped.

Check:
- dataset mount name
- folder structure
- file names (especially hospital spelling)

---

### 3) Feature importance seems weird / negative
That can be real. But verify:
- you’re using `FI_SAGE_MASKING="sample"` (more realistic)
- `FI_SAGE_N_PERMS` is large enough (20+)
- caps aren’t too small (e.g., only 100 samples gives noisy importance)

---

### 4) Macro-F1 vs accuracy confusion
If classes are imbalanced (typical in fault data), accuracy can look good even if minority faults fail.

Macro-F1:
- treats each class equally
- is a better “fault coverage” metric

---

## Interpreting the results correctly

### When robustness drops
If performance drops heavily under:
- bias/drift: model is not invariant to calibration shifts
- missing: model depends on a few critical sensors
- sampling: model needs high temporal resolution

Use feature importance + confusion matrices together:
- confusion matrix shows which faults become confused
- SAGE importance shows which sensors are driving performance

---

## Extending the pipeline

Ideas that fit this design:

1. Add calibration / probability-based metrics:
 - log loss, Brier score (requires probability outputs)
2. Add class-weighted loss for sequence models
3. Add early stopping for speed
4. Add temporal cross-validation per building
5. Add per-family transfer evaluation across datasets
6. Add per-manipulation importance comparison (feature drift sensitivity)

---

## FAQ

### Q1: Why not use SHAP instead of SAGE?
SHAP can be universal, but:
- “correct” SHAP for sequences is tricky
- KernelSHAP is expensive
- TreeSHAP won’t apply to LSTM/Transformer
SAGE-style Shapley on performance is the simplest universal method that stays honest.

---

### Q2: Why is the masking done from the training distribution?
Because “missing feature” must be defined consistently:
- train distribution is the only fair source
- using test distribution would leak information
- using a constant zero/mean can be unrealistic

---

### Q3: Why doesn’t the code permute timesteps for sequence importance?
Because that destroys temporal structure and tests something else (temporal sensitivity), not feature importance.
This code aims at sensor importance while preserving time order.

---

### Q4: Can I compare importance across models directly?
Yes — that’s the point of using a single definition:
- same held-out test set
- same metric
- same masking operator
- same Monte Carlo Shapley procedure

Still, compare within the same dataset/building/manip setting.

---

### Q5: What does “importance_mean” represent numerically?
It’s a **performance contribution** in units of the chosen metric:
- If `FI_METRIC="macro_f1"`, contribution is in macro-F1 points.
- Example: `+0.03` means adding that sensor contributes +0.03 macro-F1 on average.

---

## Quick Start Checklist

1. Ensure Kaggle datasets are mounted at the expected paths
2. Start with:
 - `LABEL_MODE="family"`
 - `SPLIT_MODE="time"`
 - `CORRUPT_WHERE="test"`
 - `FI_SAGE_N_PERMS=20`
 - `FI_SAGE_MASKING="sample"`
3. Run cell
4. Inspect:
 - `results_all.csv` for model ranking
 - `confmat_all.csv` for fault confusion
 - `feature_importance_all.csv` for sensor drivers

---

## Output Directory Example

After a successful run, you should see:

```

fdd_out/
results_all.csv
feature_importance_all.csv
confmat_all.csv

```

---

## Summary

This notebook cell provides a **full, reproducible fault-detection benchmark** with:

- multiple datasets
- multiple models (tabular + time-series)
- multiple degradations
- strict leakage prevention
- **one universal, scientific, model-agnostic feature importance method**
  ✅ SAGE-style Shapley global importance on performance  
  ✅ time order preserved  
  ✅ signed values + uncertainty  

If you want, I can also generate:
- a “how to plot” companion notebook section (performance vs degradation curves, per-building ranking, FI stability plots)
- a “methods” section you can paste into a paper/report describing the benchmark protocol and importance method
```
