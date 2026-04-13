
# Fault Detection Pipeline for HVAC Systems

A reproducible machine learning pipeline for **fault detection and diagnosis (FDD)** in building and HVAC systems.
This repository provides a unified benchmarking framework for training and evaluating both **tabular** and **sequence-based** models under **clean and degraded sensor conditions**.

---

## Overview

This repository implements a complete experimental pipeline for **HVAC fault detection**, designed to support:

* structured model comparison
* reproducible experiments
* robustness evaluation
* realistic time-aware validation

The pipeline integrates:

* multiple datasets
* multiple model families
* several split strategies
* sensor degradation testing
* standardized outputs for reporting and analysis

This framework is suitable for:

* academic research
* thesis work
* benchmarking experiments
* robustness evaluation of machine learning models

---

## Key Features

* Supports **raw fault labels** and **fault family labels**
* Multiple **data split strategies**
* Train-once / evaluate-many design
* Robustness testing using sensor degradations
* Classical and deep learning models
* GPU acceleration support (PyTorch)
* Structured result export
* LaTeX-ready model tables

---

## Implemented Models

### Tabular Models

* Linear SVM
* Random Forest
* XGBoost

### Sequence Models

* LSTM
* CNN-LSTM
* Informer-style Transformer encoder

These models allow comparison between classical machine learning approaches and deep sequence models.

---

## Supported Datasets

This pipeline currently supports:

### Dataset A — LBNL DataFDD Synthesis Inventory

Synthetic building fault detection dataset including:

* RTU systems
* VAV systems
* multi-zone configurations

### Dataset B — Nature LCU Wang Dataset

Realistic building-level dataset containing:

* multiple building types
* labeled operational faults
* sensor time-series data

The pipeline structure allows easy extension to additional datasets.

---

## Repository Structure

A recommended repository structure:

```
.
├── README.md
├── requirements.txt
├── fdd_pipeline.py
├── visualization.py
├── fdd_out/
│   ├── results_all.csv
│   ├── feature_importance_all.csv
│   ├── confmat_all.csv
│   ├── model_stats_all.csv
│   └── model_table_rows.tex
└── notebooks/
```

---

## Methodology

The pipeline follows a **train-once, test-many** methodology.

### Workflow

1. Load dataset
2. Preprocess sensor data
3. Encode fault labels
4. Split data into training and testing
5. Train models once
6. Evaluate on:

   * clean test data
   * corrupted test data
7. Export evaluation results

This design allows:

* fair comparison
* consistent evaluation
* realistic robustness testing

---

## Data Split Modes

The pipeline supports multiple splitting strategies.

---

### `time`

Strict chronological split.

* earlier data → training
* later data → testing

Best suited for:

* realistic deployment simulation
* temporal forecasting scenarios

---

### `stratified`

Random stratified split.

* maintains class balance
* ignores time ordering

Best suited for:

* quick baselines
* debugging experiments

---

### `stratified_time`

Class-wise temporal split.

Each class is split chronologically.

Useful when:

* preserving temporal structure
* maintaining class distribution

---

### `episode_stratified_time`

Episode-based temporal split.

Fault segments are grouped into episodes and split over time.

Best suited for:

* realistic fault progression studies
* sequence-based evaluation

---

## Test-Time Sensor Degradations

To simulate real-world sensor issues, the pipeline applies controlled corruptions to the test data.

---

### Noise

Additive Gaussian noise.

Simulates:

* measurement uncertainty
* sensor fluctuations

---

### Drift

Gradual temporal shift.

Simulates:

* calibration drift
* sensor aging

---

### Bias

Constant offset added to values.

Simulates:

* sensor miscalibration

---

### Missing Values

Random missing entries.

Recovered using:

* forward fill
* backward fill

Simulates:

* communication failures
* packet loss

---

### Sampling Reduction

Subsampling of time-series.

Simulates:

* reduced sensor frequency
* bandwidth limitations

---

## Output Files

All outputs are written to:

```
./fdd_out/
```

### Generated Files

| File                         | Description                  |
| ---------------------------- | ---------------------------- |
| `results_all.csv`            | Main evaluation metrics      |
| `feature_importance_all.csv` | Feature importance results   |
| `confmat_all.csv`            | Confusion matrices           |
| `model_stats_all.csv`        | Model complexity statistics  |
| `model_table_rows.tex`       | LaTeX-ready model table rows |

---

## Installation

### Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### Create Virtual Environment

Linux / macOS:

```
python -m venv .venv
source .venv/bin/activate
```

Windows:

```
python -m venv .venv
.venv\Scripts\activate
```

---

### Install Dependencies

```
pip install -r requirements.txt
```

Example minimal `requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
torch
```

---

## Usage

Run the main pipeline:

```
python Pipeline.py
```

This will:

* load datasets
* train models
* evaluate corrupted test sets
* save all outputs

---

Run the feature importance pipeline:

```
python Feature_importance.py
```

This will:

* load datasets
* train models
* compute the permutation feature importance


## Configuration

Key configuration parameters are defined at the top of the script.

Example:

```python
SPLIT_MODE = "stratified_time"

TEST_SIZE = 0.20

LABEL_MODE = "raw"

SEQ_LEN = 10
SEQ_STRIDE = 1

EPOCHS = 15
BATCH = 1024
LR = 8e-4

USE_XGBOOST = True

COMPUTE_FEATURE_IMPORTANCE = False
```

Adjust these depending on:

* dataset size
* GPU availability
* runtime limits

---

## Feature Importance

Optional global feature importance is computed using:

**SAGE-style Monte Carlo Shapley estimation**

Supports:

* tabular models
* sequence models

Outputs:

```
feature_importance_all.csv
```

---

## Model Statistics

Model complexity metrics include:

* parameter count
* tree counts
* node counts
* model memory size

These statistics support:

* computational comparison
* memory footprint analysis
* architecture reporting

---

## Reproducibility Notes

For consistent experiments:

* Training statistics are fit on training data only
* Test corruption is applied after splitting
* Random seeds are configurable
* Internal validation is optional

For strict reproducibility:

Set seeds for:

* NumPy
* PyTorch
* Python random module

---

## Example Research Questions

This repository can support experiments such as:

* How do tabular models compare to sequence models?
* How robust are models to missing data?
* Which split strategy is most realistic?
* Which sensors contribute most to predictions?
* How does performance degrade under noise?

---

## Recommended Extensions

Possible future improvements:

* Additional datasets
* Hyperparameter optimization
* Transformer architecture variants
* Cross-building transfer learning
* Experiment tracking integration
* Visualization dashboards

---

## Citation

If this repository supports academic work, include a citation.

Example:

```
@misc{yourname_fdd_pipeline_2026,
  title        = {Pipeline of Robustness analysis of Data-Driven Fault Detection methods for HVAC Systems},
  author       = {Your Name},
  year         = {2026},
  howpublished = {GitHub repository}
}
```

---

## License

Recommended licenses:

* MIT License
* Apache 2.0
* GPL v3

Example:

```
MIT License
```

Add a separate `LICENSE` file to the repository.

---

## Contact

If this project is part of academic or research work:

```
Your Name  
TU Berlin, Hermann Rietschel Institute (HRI)  
pedram.babakhani@tu-berlin.de
```

---

## Notes

This repository provides a **modular and extensible framework** for building fault detection experiments.

It is intended for:

* research
* benchmarking
* model comparison
* robustness evaluation

The structure supports both **academic research workflows** and **industrial experimentation**.
