import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from utils.model_utils import rmse, make_ridge_model, eval_model_rq1, one_way_anova, tukey_hsd, make_ohe, split_cols, make_preprocessor, wrap_log1p, eval_model_rq4

def test_rmse_known_value():
    """rmse should match the direct definition."""
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 2.0, 2.0])
    expected = np.sqrt(((y_true - y_pred) ** 2).mean())
    assert np.isclose(float(rmse(y_true, y_pred)), expected)


def test_make_ohe_fit_transform_rows():
    """make_ohe should fit/transform without changing number of rows."""
    X = pd.DataFrame({"cat": ["a", "b", "a"]})
    enc = make_ohe(dense=False)
    Xt = enc.fit_transform(X[["cat"]])
    assert Xt.shape[0] == len(X)


def test_split_cols_separates_object_and_numeric():
    """split_cols should classify object columns as categorical and numeric as numeric."""
    X = pd.DataFrame(
        {
            "median_income": [50000, 60000, 70000],
            "region_name": ["North", "South", "North"],
            "race_eth_name": ["A", "B", "A"],
        }
    )
    cat_cols, num_cols = split_cols(X, ["median_income", "region_name", "race_eth_name"])
    assert set(cat_cols) == {"region_name", "race_eth_name"}
    assert set(num_cols) == {"median_income"}


def test_make_preprocessor_handles_unknown_category():
    """make_preprocessor should not error on unseen categories at transform time."""
    X_train = pd.DataFrame(
        {
            "median_income": [50, 60, 70, 80],
            "region_name": ["North", "South", "North", "South"],
            "race_eth_name": ["A", "B", "A", "B"],
        }
    )
    feats = ["median_income", "region_name", "race_eth_name"]

    pre = make_preprocessor(feats, X_train, dense=False, scale_num_for_linear=True)
    Z_train = pre.fit_transform(X_train)

    X_test = X_train.copy()
    X_test.loc[0, "region_name"] = "NEW_REGION"  # unseen category
    Z_test = pre.transform(X_test)

    assert Z_train.shape[0] == len(X_train)
    assert Z_test.shape[0] == len(X_test)
    assert Z_test.shape[1] == Z_train.shape[1]  # handle_unknown="ignore" keeps same width


def test_wrap_log1p_runs_and_predicts_finite():
    """wrap_log1p should fit and return finite predictions on original y scale."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([0.5, 1.0, 2.0, 4.0])  # must be > -1 for log1p

    model = wrap_log1p(Pipeline([("lr", LinearRegression())]))
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_make_ridge_model_fits_and_predicts():
    """make_ridge_model should fit on a toy dataset and predict finite values."""
    n = 40
    X = pd.DataFrame(
        {
            "median_income": np.linspace(40, 100, n),
            "region_name": np.where(np.arange(n) % 2 == 0, "North", "South"),
            "race_eth_name": np.where(np.arange(n) % 3 == 0, "A", "B"),
        }
    )
    y = 0.01 * X["median_income"].to_numpy() + (X["region_name"] == "North").to_numpy() * 0.1 + 1.0

    feats = ["median_income", "region_name", "race_eth_name"]
    model = make_ridge_model(feats, X)

    X_train, X_test = X.iloc[:30][feats], X.iloc[30:][feats]
    y_train, y_test = y[:30], y[30:]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    assert pred.shape == (10,)
    assert np.isfinite(pred).all()


def test_eval_model_rq1_returns_expected_keys():
    """eval_model_rq1 should return a dict with model, RMSE, MAE, R2."""
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    model = LinearRegression()
    out = eval_model_rq1("lin", model, X.iloc[:4], X.iloc[4:], y[:4], y[4:])

    assert isinstance(out, dict)
    for k in ["model", "RMSE", "MAE", "R2"]:
        assert k in out


def test_eval_model_rq4_returns_metrics_and_pred():
    """eval_model_rq4 should return (metrics_dict, pred_array)."""
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    model = LinearRegression()
    metrics, pred = eval_model_rq4(model, X.iloc[:4], X.iloc[4:], y[:4], y[4:])

    assert isinstance(metrics, dict)
    for k in ["RMSE", "MAE", "R2"]:
        assert k in metrics
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()

def _toy_df():
    # Clear separation: B >> A,C so ANOVA + Tukey should flag B comparisons.
    return pd.DataFrame(
        {
            "affordability_ratio": [
                0, 1, 0, 1, 0,      # A
                10, 11, 10, 11, 10,  # B
                0, 0, 1, 0, 1,       # C
                None,                # NA row (should be dropped)
            ],
            "region_name": (
                ["A"] * 5
                + ["B"] * 5
                + ["C"] * 5
                + ["A"]
            ),
        }
    )

def test_one_way_anova_basic_outputs_and_significance():
    df = _toy_df()
    out = one_way_anova(df, target="affordability_ratio", group="region_name", min_group_size=5)

    assert set(out.keys()) == {"F", "p_value", "n_groups", "n_total"}
    assert out["n_groups"] == 3
    assert out["n_total"] == 15  # NA row dropped
    assert out["F"] > 0
    assert out["p_value"] < 1e-6

def test_tukey_hsd_returns_expected_table_and_rejects_big_differences():
    df = _toy_df()
    res = tukey_hsd(df, target="affordability_ratio", group="region_name", alpha=0.05)

    expected_cols = {"group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"}
    assert expected_cols.issubset(res.columns)
    assert res.shape[0] == 3  # 3 groups -> 3 pairwise comparisons

    # Ensure B differs from A and C
    ab = res[((res["group1"] == "A") & (res["group2"] == "B")) | ((res["group1"] == "B") & (res["group2"] == "A"))]
    cb = res[((res["group1"] == "C") & (res["group2"] == "B")) | ((res["group1"] == "B") & (res["group2"] == "C"))]

    assert len(ab) == 1 and bool(ab.iloc[0]["reject"]) is True
    assert len(cb) == 1 and bool(cb.iloc[0]["reject"]) is True