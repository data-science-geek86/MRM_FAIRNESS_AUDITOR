# ⚖️ Bias Audit & Fairness Assessment Engine

Welcome to the **Bias Audit & Fairness Assessment Engine** documentation. This library delivers an automated, data-driven framework designed to audit, quantify, and visualize statistical disparities and potential biases across demographic subgroups within datasets or model predictions.

By leveraging the **Group Stability Index (GSI)** alongside customizable statistical binning/partitioning strategies, this engine surfaces structural imbalances relative to a designated baseline reference group.


## 📦 Installation Guide: Bias Audit & Fairness Engine

Follow step-by-step instructions for installing the **Bias Audit & Fairness Assessment Engine** and its necessary dependencies. 

---

### 📋 Prerequisites

Before installing, ensure you have the following requirements met:
* **Python**: Version `3.8` or higher is recommended.
* **pip**: Python package installer (updated to the latest version).

The core dependencies that will be installed automatically include:
* `numpy` 
* `pandas` 
* `plotly` 
* `scikit-learn`

---

### 🚀 1. Standard Installation via PyPI

For most users, the standard and stable version can be installed directly from PyPI using `pip`. Run the following command in your terminal or command prompt:

```bash
pip install mrm_fairness_auditor
```

### 🛠️ 2. Installation from GitHub (Development Version)
If you wish to work with the latest features, contribute to development, or run the engine locally from source, you can install directly from the version control repository.

#### Install Directly via pip and Git
You can point pip directly to the remote repository without downloading it manually:

``` bash

pip install git+https://github.com/data-science-geek86/MRM_FAIRNESS.git#subdirectory=mrm_fairness

```

---

## 🔬 Core Methodology (GSI Calculation)

The engine utilizes the **Group Stability Index (GSI)** to quantify representation disparities. Originally designed to monitor population shifts over time, it is adapted here to measure systemic bias by tracking distributional differences between a designated **Reference Group** ($R$) and a **Comparison Group** ($C$).

For any given attribute divided into $k$ discrete bins, the index contribution for an individual bin $i$ is calculated as:


$$\text{GSI}_i = (C_i - R_i) \times \ln\left(\frac{C_i}{R_i}\right)$$


The **Global GSI** for that feature is the sum of all bin-level contributions:


$$\text{GSI} = \sum_{i=1}^{k} \left( (C_i - R_i) \times \ln\left(\frac{C_i}{R_i}\right) \right)$$


### 🚦 Thresholds and Actions

The calculated global score is mapped to actionable fairness tiers:

| GSI Score Range | Fairness Status | Recommended Action |
| :--- | :--- | :--- |
| **$\text{GSI} < 0.10$** | 🟢 Fair / Minimal Disparity | Acceptable stability profile. No adjustments required. |
| **$0.10 \le \text{GSI} < 0.25$** | 🟡 Mild Bias / Moderate Disparity | Monitor carefully over time. Minor deviations detected. |
| **$\text{GSI} \ge 0.25$** | 🔴 Significant Bias / High Disparity | **Action Required!** Mitigate underlying bias dependencies. |

---

## 📦 1. Data Binning Strategies

To build valid probability distributions for continuous numerical features, the library provides specialized binning models through the `BinningStrategy` interface.

### `EqualWidthBinning`
Divides the entire range of data into $k$ intervals of completely equal sizing bounds.
* **Best for:** Uniformly distributed continuous features.

### `EqualFrequencyBinning`
Partitions data using mathematical quantiles so that every bucket holds an equal number of raw records.
* **Best for:** Heavily skewed distributions or data containing severe outliers.

### `KMeansBinning`
Applies a 1D Unsupervised K-Means clustering routine to discover natural groupings. Boundary edges are defined as the midpoints between cluster centroids.
* **Best for:** Uncovering hidden sub-populations or multi-modal attributes.

### `DecisionTreeBinning`
An adaptive, supervised partitioning scheme that fits a shallow decision tree against a target label. It optimizes boundaries to maximize mutual information relative to the outcome.
* **Best for:** Finding bias boundaries that directly correlate with downstream predictions or label outcomes.

---

## ⚙️ 2. The Core Engine (`BiasAuditor`)

The `BiasAuditor` class serves as the central orchestration hub. It performs data validation, isolates demographic groups, runs the selected binning strategies, and applies mathematical smoothing to prevent division-by-zero errors.

### Key Class Methods

#### `__init__(df, protected_attribute, reference_group, epsilon=1e-4)`
* Initializes the auditing pipeline.
* Validates that the sensitive attribute exists and contains the specified baseline group.
* Uses `epsilon` as an anti-zero adjustment factor to handle empty bins gracefully.

#### `calculate_gsi(target_cols, binning_method='equal_frequency', num_bins=10, dt_target_col=None)`
* Executes the multi-group disparity calculations.
* **Returns:** A `Tuple` containing two pandas DataFrames:
    1. **Summary Table:** A high-level overview mapping each target column to its global GSI, fairness tier, and recommended action.
    2. **Granular Table:** A comprehensive bin-by-bin breakdown showing raw probability weights and individual index contributions.

---

## 📊 3. Interactive Visualizations (`BiasVisualizer`)

Raw audit metrics can be difficult to fully comprehend without visual context. The `BiasVisualizer` class integrates directly with Plotly to render diagnostic subplots.

### `plot_granular_distribution(granular_df, target_feature)`
* Accepts the granular data frame generated directly by `BiasAuditor`.
* Automatically renders aligned, multi-row bar charts contrasting the baseline reference distribution against each comparison cohort side-by-side.

---

## 🚀 4. Quickstart Guide & Example

The following self-contained script shows how to implement a complete bias assessment pipeline using synthetic credit data.

``` python
import numpy as np
import pandas as pd

# 1. Generate dummy audit data
np.random.seed(42)
n_samples = 1000

mock_data = {
    'demographic_group': np.random.choice(['Group_A', 'Group_B', 'Group_C'], size=n_samples, p=[0.5, 0.3, 0.2]),
    'credit_score': np.random.normal(loc=650, scale=50, size=n_samples),
    'approved': np.random.choice([0, 1], size=n_samples)
}
df = pd.DataFrame(mock_data)

# Inject synthetic disparity into Group_C for test validation
df.loc[df['demographic_group'] == 'Group_C', 'credit_score'] -= 45

# 2. Instantiate the auditor specifying Group_A as the baseline
auditor = BiasAuditor(df=df, protected_attribute='demographic_group', reference_group='Group_A')

# 3. Compute metrics using KMeans partitioning
summary_df, granular_df = auditor.calculate_gsi(
    target_cols=['credit_score'],
    binning_method='kmeans',
    num_bins=5
)

# 4. Display high-level fairness findings
print("=== GLOBAL FAIRNESS RESULTS ===")
print(summary_df[['Comparison Group', 'Target Feature/Prediction', 'GSI Value', 'Fairness Status']])

# 5. Render interactive distribution plots
BiasVisualizer.plot_granular_distribution(granular_df, target_feature='credit_score')

```