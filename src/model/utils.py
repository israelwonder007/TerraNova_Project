import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(filepath="data/processed/final_selected_features.csv"):
    """
    Load dataset and split into X (features) and y (target)
    """
    df = pd.read_csv(filepath)

    print("Loaded dataset:", df.shape)

    if "log_total_cost" not in df.columns:
        raise ValueError("Target column 'log_total_cost' not found in dataset")

    X = df.drop(columns=["log_total_cost"])
    y = df["log_total_cost"]

    # Remove ID column if present
    if "disasterNumber" in X.columns:
        X = X.drop(columns=["disasterNumber"])

    return X, y


# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -------------------------------
# PREPROCESSOR
# -------------------------------
def get_preprocessor():
    """
    Returns a scaler (used mainly for linear models)
    """
    return StandardScaler()


# -------------------------------
# EVALUATION METRICS
# -------------------------------
def evaluate_model(y_true, y_pred):
    """
    Compute evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


# -------------------------------
# PRINT METRICS
# -------------------------------
def print_metrics(run_name, metrics: dict):
    """
    Nicely print model performance
    """
    print(
        f"{run_name:20s} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"R2: {metrics['r2']:.4f}"
    )