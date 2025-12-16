import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def rmse(y_true, y_pred):
    """
    Compute RMSE between true and predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def make_ridge_model(feature_list, X_ref: pd.DataFrame):
    """
    Build a RidgeCV pipeline for regression with log1p(y) target transform.

    Parameters
    ----------
    feature_list : list[str]
        Predictor column names.
    X_ref : pandas.DataFrame
        Reference data used to infer categorical vs numeric columns.

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        Model that fits on log1p(y) and predicts on the original y scale.
    """
    cat_cols = [c for c in feature_list if X_ref[c].dtype == "object"]
    num_cols = [c for c in feature_list if c not in cat_cols]

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", StandardScaler(), num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))

    pipe = Pipeline([
        ("pre", pre),
        ("reg", ridge),
    ])

    model = TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return model

def eval_model_rq1(name, model, X_train, X_test, y_train, y_test):
    """
    Fit a model and return test metrics (RQ1 format).

    Parameters
    ----------
    name : str
        Model label.
    model : sklearn estimator
        Any object with .fit() and .predict().
    X_train, X_test : array-like or pandas.DataFrame
        Train/test features.
    y_train, y_test : array-like
        Train/test targets.

    Returns
    -------
    dict
        Metrics dict with keys: model, RMSE, MAE, R2.
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "model": name,
        "RMSE": rmse(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }

def make_ohe(dense: bool):
    """
    Create a version-compatible OneHotEncoder.

    Parameters
    ----------
    dense : bool
        If True, return dense output; otherwise sparse.

    Returns
    -------
    sklearn.preprocessing.OneHotEncoder
        Configured one-hot encoder with handle_unknown="ignore".
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)

def split_cols(X, feature_list):
    """
    Split features into categorical vs numeric using pandas dtypes.

    Parameters
    ----------
    X : pandas.DataFrame
        Reference data for dtype inspection.
    feature_list : list[str]
        Columns to split.

    Returns
    -------
    (list[str], list[str])
        (cat_cols, num_cols).
    """
    cat_cols, num_cols = [], []
    for c in feature_list:
        if X[c].dtype == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols

def make_preprocessor(feature_list, X_ref, dense=False, scale_num_for_linear=True):
    """
    Build a ColumnTransformer for one-hot encoding + optional numeric scaling.

    Parameters
    ----------
    feature_list : list[str]
        Columns to include.
    X_ref : pandas.DataFrame
        Reference data for dtype inspection.
    dense : bool, default False
        Dense (True) or sparse (False) one-hot output.
    scale_num_for_linear : bool, default True
        If True, StandardScale numeric columns.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessing transformer.
    """
    cat_cols, num_cols = split_cols(X_ref, feature_list)
    ohe = make_ohe(dense=dense)

    if scale_num_for_linear:
        num_transform = Pipeline([("scaler", StandardScaler())])
    else:
        num_transform = "passthrough"

    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", num_transform, num_cols),
        ],
        remainder="drop"
    )
    return pre

def wrap_log1p(model_pipeline):
    """
    Wrap an estimator to fit log1p(y) and invert predictions with expm1.

    Parameters
    ----------
    model_pipeline : sklearn estimator
        Any estimator/pipeline with .fit() and .predict().

    Returns
    -------
    sklearn.compose.TransformedTargetRegressor
        Wrapped estimator operating on log1p(y).
    """
    return TransformedTargetRegressor(
        regressor=model_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
def eval_model_rq4(model, Xtr, Xte, ytr, yte):
    """
    Fit a model and return metrics + predictions (RQ4 format).

    Parameters
    ----------
    model : sklearn estimator
        Any object with .fit() and .predict().
    Xtr, Xte : array-like or pandas.DataFrame
        Train/test features.
    ytr, yte : array-like
        Train/test targets.

    Returns
    -------
    (dict, numpy.ndarray)
        (metrics, y_pred) where metrics has RMSE, MAE, R2.
    """
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        "RMSE": rmse(yte, pred),
        "MAE": float(mean_absolute_error(yte, pred)),
        "R2":  float(r2_score(yte, pred))
    }
    return metrics, pred