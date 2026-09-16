# Imbalanced Tabular Data Resampling Pipeline

A high-performance, modular, and defensive Python pipeline engineered to resolve severe class imbalance in tabular numerical datasets. Built upon GoF design patterns (Strategy, Factory), strict data schema validation, standardized POSIX exit codes, and automated path resolution, this engine serves both as an independent command-line utility and as a decoupled backend engine for desktop GUIs (e.g., C++ / Qt node systems) and automated orchestration pipelines.

---

## Table of Contents

- [System Architecture & Design Patterns](#system-architecture--design-patterns)
  - [SOLID Principles Compliance](#solid-principles-compliance)
  - [Pipeline Flow Diagram](#pipeline-flow-diagram)
  - [Directory Layout](#directory-layout)
- [Algorithmic Foundations & Mathematical Mechanics](#algorithmic-foundations--mathematical-mechanics)
  - [1. SMOTE (Synthetic Minority Over-sampling Technique)](#1-smote-synthetic-minority-over-sampling-technique)
  - [2. ADASYN (Adaptive Synthetic Sampling)](#2-adasyn-adaptive-synthetic-sampling)
  - [3. SMOTE-TOMEK (Hybrid Over-sampling & Boundary Cleaning)](#3-smote-tomek-hybrid-over-sampling--boundary-cleaning)
- [Defensive Data Validation Contract](#defensive-data-validation-contract)
- [Automated Path Resolution & File Prefixing](#automated-path-resolution--file-prefixing)
- [Installation](#installation)
- [Command-Line Interface (CLI) Reference](#command-line-interface-cli-reference)
  - [Parameter Contracts](#parameter-contracts)
  - [Execution Examples](#execution-examples)
- [Process Exit Codes Contract](#process-exit-codes-contract)
- [External Process & IPC Integration (C++ / Qt)](#external-process--ipc-integration-c--qt)

---

## System Architecture & Design Patterns

The pipeline is decoupled into discrete abstraction layers to eliminate monolithic procedural execution and ensure strict separation of concerns.

```
+─────────────────────────────────────────────────────────────────────────────+
|                             Entry Layer (CLI / IPC)                         |
|                    main.py (ExitCode & Unhandled Boundary)                  |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                         Configuration & Contract Layer                      |
|          src/config.py (PipelineConfig, Argparse, Prefix Resolution)        |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                     Ingestion & Defensive Validation Layer                  |
|          src/io_handler.py (CSV Parsing, NaN Checks, k-NN Feasibility)      |
+──────────────────────────────────────┬──────────────────────────────────────+
                                       │
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                       Strategy Selection (Factory Layer)                    |
|             src/resamplers/factory.py (ResamplerFactory)                    |
+───────────────────┬──────────────────┼───────────────────┬──────────────────+
                    │                  │                   │
                    ▼                  ▼                   ▼
            ┌──────────────┐   ┌───────────────┐   ┌───────────────┐
            │ SMOTEResampler│   │ADASYNResampler│   │SMOTETomekRes. │
            └──────────────┘   └───────────────┘   └───────────────┘
                    ▲                  ▲                   ▲
                    └──────────────────┴───────────────────┘
                                       │
                           implements BaseResampler (ABC)
                                       │
                                       ▼
+─────────────────────────────────────────────────────────────────────────────+
|                          Data Export & Logging Layer                        |
|   src/io_handler.py (Disk Persistence), src/logger.py (Console Logging),    |
|       & src/report_generator.py (Detailed ASCII Execution Reports)          |
+─────────────────────────────────────────────────────────────────────────────+
```

### SOLID Principles Compliance

- **Single Responsibility Principle (SRP):**
  - `IOHandler` exclusively governs disk I/O, tabular parsing, and schema invariants.
  - `PipelineConfig` strictly handles parameter validation, normalization, and path inference.
  - `ExecutionReporter` formats and writes comprehensive profiling benchmarks, comparative class distributions, and failure audits.
  - `ResamplingPipeline` orchestrates execution lifecycle without coupling to specific algorithmic details.
  - Concrete resamplers (`SMOTEResampler`, `ADASYNResampler`, `SMOTETomekResampler`) encapsulate only their respective mathematical transformations.
- **Open/Closed Principle (OCP):**
  - New resampling methodologies (e.g., Borderline-SMOTE, SMOTE-ENN) can be integrated by extending `BaseResampler` and registering them in `ResamplerFactory` without altering existing pipeline or entrypoint logic.
- **Liskov Substitution Principle (LSP):**
  - Every concrete resampler strictly satisfies the `BaseResampler` contract: `resample(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]`. Calling code interacts exclusively via the abstract interface.
- **Interface Segregation Principle (ISP):**
  - `BaseResampler` defines the minimal necessary interface required for dataset balancing, preventing interface pollution.
- **Dependency Inversion Principle (DIP):**
  - High-level orchestration (`ResamplingPipeline`) depends on abstractions (`BaseResampler`, `ResamplerFactory`), never on concrete low-level algorithmic modules.

### Directory Layout

```
imbalanced-data-resampling/
├── main.py                  # CLI Entrypoint, exit code mapping, top-level error isolation
├── requirements.txt         # Production dependencies
└── src/
    ├── __init__.py          # Package exposure
    ├── config.py            # Typed PipelineConfig dataclass, CLI parser, prefix resolver
    ├── exceptions.py        # Domain exception hierarchy (ResamplingError) & ExitCode enum
    ├── io_handler.py        # Defensive data loader, k-NN feasibility check, CSV persistence
    ├── logger.py            # Real-time stdout console logging utility
    ├── pipeline.py          # Orchestration engine managing lifecycle and metrics
    ├── report_generator.py  # Structured comparative benchmark & failure audit reporter
    └── resamplers/
        ├── __init__.py      # Subpackage exports
        ├── base.py          # Abstract BaseResampler (ABC) definition
        ├── smote_resampler.py        # Vectorized SMOTE implementation
        ├── adasyn_resampler.py       # Density-weighted ADASYN implementation
        ├── smote_tomek_resampler.py  # Two-stage SMOTE + TomekLinks hybrid resampler
        └── factory.py                # ResamplerFactory strategy selector
```

---

## Algorithmic Foundations & Mathematical Mechanics

Class imbalance poses a fundamental risk to statistical learning algorithms: standard objective functions minimize global error, leading models to favor majority distributions and treat minority signals as uninformative outliers. This pipeline provides three mathematically rigorous sampling strategies.

```
     Raw Feature Space                    SMOTE                     SMOTE-TOMEK
  (Severe Class Imbalance)        (Linear Interpolation)         (Boundary Cleaned)

  ○  ○  ○  ○  ○  ○  ○  ○           ○  ○  ○  ○  ○  ○  ○  ○         ○  ○  ○  ○  ○  ○  ○  ○
  ○  ○  ○  ○  ○  ○  ○  ○           ○  ○  ○  ○  ○  ○  ○  ○         ○  ○  ○  ○  ○  ○  ○  ○
  ○  ○  ○  ○  ○  ○  ○  ○           ○  ○  ○  ○  ○  ○  ○  ○         ○  ○  ○  ○  ○  ○  ○  ○
               ▲                                ▲                            ▲
               │ Decision Boundary              │ Overlapping Synthetic      │ Sharpened Boundary
  ─────────────┼───────────────    ─────────────┼───────────────  ───────────┼───────────
               │                                │                            │
        ●    ● │                        ●   *●* │ *●*                     ●  │
          ●    │                          ●   *●*  ●                        ●│
                                                                             │
  Legend: ○ Majority Class    ● Minority Class    *●* Synthetic Point (Noise cleaned in Tomek)
```

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
- **Mathematical Mechanics:**
  For each minority instance $x_i \in X_{min}$, the algorithm identifies its $k$-nearest neighbors within $X_{min}$ using Euclidean metric space:
  $$d(x_i, x_j) = \sqrt{\sum_{f=1}^{D} (x_{i,f} - x_{j,f})^2}$$
  A neighbor $x_{zi}$ is sampled randomly from the $k$-NN set, and a synthetic instance $x_{new}$ is generated along the line segment joining $x_i$ and $x_{zi}$:
  $$x_{new} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
- **Theoretical Characteristics:**
  - Expands the minority convex hull uniformly without considering majority class proximity.
  - **When to Use:** Baseline oversampling where minority instances form coherent clusters and boundary overlap is minimal.

### 2. ADASYN (Adaptive Synthetic Sampling)
- **Mathematical Mechanics:**
  Unlike SMOTE, ADASYN weights minority instances adaptively according to their learning difficulty. For each $x_i \in X_{min}$, its $k$-nearest neighbors are computed across the **entire dataset** ($X_{all} = X_{min} \cup X_{maj}$). The ratio of majority samples in its neighborhood is calculated:
  $$r_i = \frac{\Delta_i}{k}, \quad \Delta_i \in \{0, \dots, k\}$$
  Where $\Delta_i$ is the number of majority instances in the $k$-NN of $x_i$. The ratios are normalized into a discrete probability density:
  $$\hat{r}_i = \frac{r_i}{\sum_{j=1}^{|X_{min}|} r_j}$$
  Given the total synthetic samples required $G = (|X_{maj}| - |X_{min}|) \times \beta$, instance $x_i$ is allocated a generation quota:
  $$g_i = \hat{r}_i \times G$$
- **Defensive Safeguard:**
  If a dataset has isolated minority instances with zero majority neighbors within $k$-NN ($\sum r_i = 0$), standard ADASYN induces a division-by-zero runtime failure. The pipeline intercepts this condition defensively and raises an informative `ResamplingExecutionError` suggesting parameter or algorithmic adjustments.
- **When to Use:** Datasets where decision boundaries are critical, and complex minority instances near boundaries require enhanced density.

### 3. SMOTE-TOMEK (Hybrid Over-sampling & Boundary Cleaning)
- **Mathematical Mechanics:**
  A two-stage pipeline combining generative synthesis with topological boundary pruning:
  1. **Stage 1 (Generative Oversampling):** SMOTE generates synthetic minority instances to balance class proportions.
  2. **Stage 2 (Tomek Links Identification):** A pair of instances $(x_i, x_j)$ forms a Tomek link if $y_i \neq y_j$ and:
     $$\forall x_k: d(x_i, x_j) < d(x_i, x_k) \quad \text{and} \quad d(x_i, x_j) < d(x_j, x_k)$$
  3. **Stage 3 (Topological Pruning):** Under the default `--tomek_sampling_strategy "auto"` (corresponding to `'not minority'`), majority instances forming Tomek links are removed, clarifying the boundary margin and eliminating overlapping ambiguity.
- **When to Use:** Noisy tabular datasets with borderline overlap, sensor cross-talk, or high-dimensional boundary ambiguity.

---

## Defensive Data Validation Contract

The `IOHandler` acts as a gatekeeper executing pre-flight verification before data reaches the algorithmic layer:

| Check | Failure Condition | Raised Exception | Exit Code | Rationale |
|---|---|---|---|---|
| **File Integrity** | File missing or unreadable | `FileIOError` | `1` | Prevents runtime OS unhandled IO traps. |
| **Tabular Content** | Empty DataFrame (0 rows) | `DataValidationError` | `3` | Fails fast before allocating matrix memory. |
| **Target Presence** | `target_col` absent in columns | `DataValidationError` | `3` | Provides immediate feedback with available columns. |
| **Missing Values** | Any NaN / null in features or target | `DataValidationError` | `3` | Resampling algorithms reject un-imputed numerical data. |
| **Class Cardinality** | Target unique classes $< 2$ | `DataValidationError` | `3` | Resampling requires at least binary classification. |
| **$k$-NN Feasibility** | $N_{minority} \le k$ | `DataValidationError` | `3` | Mathematical constraint: k-NN graph construction fails if minority sample count $\le k$. Explicitly advises setting $k \le N_{minority} - 1$. |
| **Feature Typings** | Non-numeric data types present | `DataValidationError` | `3` | Euclidean distance calculations mandate purely numeric feature spaces. |

---

## Automated Path Resolution & File Prefixing

To guarantee idempotency and prevent accidental overwrites, `--output_path` is **mandatory** and features intelligent path resolution:

1. **Directory Input:** If `--output_path` specifies an existing folder or ends with a directory delimiter (`/` or `\`):
   $$\text{Output Path} = \text{Resolved Dir} \ / \ \texttt{"resampled\_\{method\}\_\{input\_filename\}"}$$
   *Example:* `--output_path "C:/datasets/"` with `--method SMOTE` $\rightarrow$ `C:/datasets/resampled_smote_data.csv`
2. **File Path Input:** If `--output_path` specifies a target file name:
   $$\text{Output Path} = \text{Resolved Parent} \ / \ \texttt{"resampled\_\{specified\_filename\}"}$$
   *Example:* `--output_path "balanced.csv"` $\rightarrow$ `resampled_balanced.csv`
   *(If the provided filename already includes `resampled_`, redundant prefixing is skipped).*

---

## Installation

### Prerequisites
- Python >= 3.9
- Virtual environment tool (`venv` or `conda`)

```bash
# Clone the repository
git clone https://github.com/your-username/imbalanced-data-resampling.git
cd imbalanced-data-resampling

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Command-Line Interface (CLI) Reference

### Parameter Contracts

| Argument | Type | Required? | Default | Permitted Values / Description |
|---|---|---|---|---|
| `--input_path` | `str` | **Yes** | — | Absolute or relative path to the input CSV dataset. |
| `--output_path` | `str` | **Yes** | — | Mandatory destination path or directory. Automatically resolves with `resampled_` prefix. |
| `--target_col` | `str` | **Yes** | — | Column name identifying the target class or label. |
| `--method` | `str` | No | `SMOTE` | `SMOTE`, `ADASYN`, `SMOTE-TOMEK` (case-insensitive). |
| `--k_neighbors` | `int` | No | `5` | Nearest neighbors ($k \ge 1$). Must satisfy $k < N_{minority}$. |
| `--sampling_strategy` | `str` | No | `auto` | `auto`, `minority`, `not majority`, `all`, or a numeric float ratio. |
| `--sampling_ratio` | `float` | No | `None` | Desired ratio of minority to majority ($0.0 < \text{ratio} \le 1.0$). Overrides `auto`. |
| `--random_seed` | `int` | No | `42` | Pseudo-random seed ensuring deterministic reproducibility. |
| `--tomek_sampling_strategy`| `str` | No | `auto` | Tomek Links pruning policy: `auto`, `all`, `not minority`. |

### Execution Examples

#### 1. Baseline SMOTE (1:1 Complete Balance)
```bash
python main.py \
  --input_path "data/fraud_records.csv" \
  --output_path "data/balanced_output/" \
  --target_col "is_fraud" \
  --method "SMOTE" \
  --k_neighbors 5 \
  --sampling_ratio 1.0
```

#### 2. Adaptive Boundary Oversampling with ADASYN (80% Ratio)
```bash
python main.py \
  --input_path "data/credit_risk.csv" \
  --output_path "data/resampled_credit_risk.csv" \
  --target_col "default" \
  --method "ADASYN" \
  --k_neighbors 7 \
  --sampling_ratio 0.8 \
  --random_seed 101
```

#### 3. Hybrid SMOTE-TOMEK Boundary Cleaning
```bash
python main.py \
  --input_path "data/rare_events.csv" \
  --output_path "data/clean_dataset.csv" \
  --target_col "anomaly" \
  --method "SMOTE-TOMEK" \
  --k_neighbors 3 \
  --tomek_sampling_strategy "auto"
```

---

## Process Exit Codes Contract

Process exit codes follow standardized POSIX semantics, enabling parent processes and external wrappers to determine error causality without parsing stdout/stderr text:

| Exit Code | Enumeration Constant | Failure Cause | Corrective Action |
|---|---|---|---|
| `0` | `ExitCode.SUCCESS` | Execution completed successfully; dataset saved. | Process resampled output data. |
| `1` | `ExitCode.FILE_IO_ERROR` | Input file not found, permission denied, or disk unwriteable. | Check file path and filesystem permissions. |
| `2` | `ExitCode.INVALID_ARGUMENTS` | Missing mandatory `--output_path`, unknown method, or $k \le 0$. | Inspect CLI parameters against argument contract. |
| `3` | `ExitCode.DATA_VALIDATION_ERROR`| Target missing, NaNs detected, or $N_{minority} \le k$. | Clean dataset, impute missing values, or lower $k$. |
| `4` | `ExitCode.RESAMPLING_FAILED` | Internal algorithmic failure (e.g., ADASYN zero density). | Adjust neighborhood $k$ or switch to SMOTE. |
| `5` | `ExitCode.UNEXPECTED_ERROR` | Uncaught system exception or severe runtime crash. | Check stderr stack trace. |

---

## External Process & IPC Integration (C++ / Qt)

This pipeline operates as a standalone CLI worker engine designed for decoupled process execution (e.g., via C++ / Qt `QProcess`, Node.js `child_process`, or Python `subprocess`).

