[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/sSkqmNLf)

# California Food Affordability Analysis

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-f25/final-group18/main)

## Overview
This project analyzes **food affordability across places in California** and investigates how affordability varies with **socioeconomic**, **geographic**, and **demographic** context. The motivation is to understand where affordability burdens are highest and which contextual predictors are most informative, using interpretable statistical tests and simple regression models.

The analysis is organized into four research questions (RQs), implemented as notebooks. Across the project we emphasize reproducibility by (1) using a shared utilities module, (2) saving outputs to disk, and (3) including unit tests for utility functions.

## Dataset
We use the **Food Affordability** indicator published by the **California Department of Public Health** on the State of California Open Data portal. The dataset describes the average cost of a nutritious market basket relative to income for **female-headed households with children**, reported for California and multiple geographic levels (regions/counties/places).

**Source:** https://data.ca.gov/dataset/food-affordability
**Temporal coverage:** 2006–2010
**Publisher:** California Department of Public Health

## Project Website
The project's JupyterBook website can be accessed [here](https://ucb-stat-159-f25.github.io/project-Group18/main.html)

## Repository Structure

* `main.ipynb`: Main narrative notebook (project summary + key results)

* `EDA.ipynb`: Exploratory analysis (tables/figures and initial patterns)

* `model_rq1.ipynb`: RQ1 modeling (baseline vs. ridge; small interpretable feature sets)

* `model_rq2.ipynb`: RQ2 inference (one-way ANOVA + Tukey HSD)

* `model_rq3.ipynb`: RQ3 exploration (summaries/visualizations by key groupings)

* `model_rq4.ipynb`: RQ4 modeling (preprocessing + evaluation variants)

* `data/`: Input dataset

* `figures/`: Generated figures

* `outputs/`: Generated tables/metrics

* `pdf_builds/`: PDF export outputs of the analysis notebooks

* `_build/`: MyST build artifacts

* `utils/`: Utility module(s)

  * `utils/model_utils.py`: Shared modeling/stats helpers
  * `utils/tests/`: Unit tests (pytest)

* `environment.yml`: Reproducible environment specification

* `conftest.py`: Pytest configuration for imports

## Setup and Installation

1. Clone this repository:

```python
git clone https://github.com/UCB-stat-159-f25/final-group18
cd final-group18
```

2. Create and activate the environment:

```python
mamba env create -f environment.yml --name stat159final
conda activate stat159final
```

```python
python -m ipykernel install --user --name stat159final --display-name "IPython - stat159final"
```

## Usage

### Run notebooks

Open JupyterLab and run:

1. `EDA.ipynb`
2. `model_rq1.ipynb`
3. `model_rq2.ipynb`
4. `model_rq3.ipynb`
5. `model_rq4.ipynb`
6. `main.ipynb`

Figures and tables are written to `figures/` and `outputs/`.

### Automation (MyST)

Build the MyST site / configured exports:

```python
myst build
```

Build PDF exports:

```python
myst build --pdf
```

## Package Structure (`utils/model_utils.py`)

The `model_utils` module provides small reusable helpers used across RQ1–RQ4.

### Metrics
- `rmse(y_true, y_pred)`: Computes root mean squared error (RMSE) between true and predicted values.

### RQ1: Interpretable regression modeling
- `make_ridge_model(feature_list, X_ref)`: Builds a RidgeCV regression pipeline that:
  - one-hot encodes categorical predictors,
  - scales numeric predictors,
  - fits on `log1p(y)` and returns predictions on the original scale via `expm1`.
- `eval_model_rq1(name, model, X_train, X_test, y_train, y_test)`: Fits a model and returns a metrics dictionary (`RMSE`, `MAE`, `R2`) labeled with `name`.

### RQ2: Group mean inference
- `one_way_anova(df, target, group, min_group_size=5)`: Runs a one-way ANOVA comparing mean `target` across levels of `group`, excluding groups with fewer than `min_group_size` observations. Returns `{F, p_value, n_groups, n_total}`.
- `tukey_hsd(df, target, group, alpha=0.05)`: Runs Tukey’s HSD post-hoc pairwise comparisons for `target` across `group`. Returns a tidy summary table as a DataFrame.

### RQ4: Preprocessing + evaluation utilities
- `make_ohe(dense)`: Creates a version-compatible `OneHotEncoder` with `handle_unknown="ignore"` (works across sklearn versions).
- `split_cols(X, feature_list)`: Splits the provided feature list into categorical vs. numeric columns using pandas dtypes from `X`.
- `make_preprocessor(feature_list, X_ref, dense=False, scale_num_for_linear=True)`: Builds a `ColumnTransformer` that one-hot encodes categorical columns and optionally scales numeric columns (useful for linear models).
- `wrap_log1p(model_pipeline)`: Wraps an estimator/pipeline with a `log1p` target transform (and `expm1` inverse) using `TransformedTargetRegressor`.
- `eval_model_rq4(model, Xtr, Xte, ytr, yte)`: Fits a model and returns `(metrics, predictions)`, where `metrics` includes `RMSE`, `MAE`, and `R2`.

## Testing

Run tests from the repository root:

```python
pytest -q
```

## License

This project is licensed under the BSD 3-Clause License.

## Additional Information

See `project-description.md` for the assignment specification and `main.ipynb` for the final narrative results.
