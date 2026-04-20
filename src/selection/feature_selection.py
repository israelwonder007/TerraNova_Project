import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


# -------------------------------
# DROP USELESS FEATURES
# -------------------------------
def drop_constant_features(df: pd.DataFrame):
    nunique = df.nunique()
    cols_to_drop = nunique[nunique <= 1].index.tolist()

    return df.drop(columns=cols_to_drop), cols_to_drop


# -------------------------------
# DROP HIGH MISSING
# -------------------------------
def drop_high_missing(df: pd.DataFrame, threshold=0.5):
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    return df.drop(columns=cols_to_drop), cols_to_drop


# -------------------------------
# MUTUAL INFORMATION
# -------------------------------
def mutual_info_selection(X: pd.DataFrame, y: pd.Series, top_k=20):
    X = X.select_dtypes(include=[np.number]).fillna(0)

    mi_scores = mutual_info_regression(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns)

    mi_series = mi_series.sort_values(ascending=False)

    selected_features = mi_series.head(top_k).index.tolist()

    return selected_features, mi_series


# -------------------------------
# PREPARE X, y
# -------------------------------
def prepare_xy(df: pd.DataFrame, target="log_total_cost"):
    
    # Drop leakage columns
    leakage_cols = [
        "total_cost",
        "totalObligatedAmountPa",
        "totalObligatedAmountHmgp",
        "totalAmountIhpApproved"
    ]

    df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

    # Drop non-numeric for now (can encode later)
    X = df.select_dtypes(include=[np.number]).copy()
    y = df[target]

    return X, y