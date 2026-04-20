import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import mlflow

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/processed/final_selected_features.csv")

print("Dataset shape:", df.shape)


# -------------------------------
# SPLIT FEATURES / TARGET
# -------------------------------
X = df.drop(columns=["log_total_cost"])
y = df["log_total_cost"]


# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)


# -------------------------------

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Import shared utilities
from src.model.utils import (
    load_data,
    split_data,
    get_preprocessor,
    evaluate_model,
    print_metrics
)


# -------------------------------
# MLFLOW SETUP
# -------------------------------
from src.model.config import MLFLOW_URI, EXPERIMENT_NAME

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# -------------------------------
# MAIN TRAIN FUNCTION
# -------------------------------
def run_training():

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    X, y = load_data()

    # -------------------------------
    # SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # -------------------------------
    # PREPROCESSOR
    # -------------------------------
    preprocessor = get_preprocessor()

    # -------------------------------
    # MODELS
    # -------------------------------
    models = [
        ("Linear Regression", LinearRegression(), {}),

        ("Ridge", Ridge(alpha=1.0), {
            "alpha": 1.0
        }),

        ("Lasso", Lasso(alpha=0.1), {
            "alpha": 0.1
        }),

        ("Random Forest", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ), {
            "n_estimators": 100
        }),

        ("XGBoost", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ), {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6
        })
    ]

    # -------------------------------
    # TRAIN LOOP
    # -------------------------------
    for run_name, model, params in models:

        with mlflow.start_run(run_name=run_name):

            pipeline = Pipeline([
                ('scaler', preprocessor),
                ('model', model)
            ])

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)

            # Evaluate
            metrics = evaluate_model(y_test, y_pred)

            # Print
            print_metrics(run_name, metrics)

            # -------------------------------
            # LOG METRICS
            # -------------------------------
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # -------------------------------
            # LOG PARAMETERS
            # -------------------------------
            mlflow.log_param("model", model.__class__.__name__)

            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # -------------------------------
            # LOG MODEL
            # -------------------------------
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=f"{model.__class__.__name__}_pipeline"
            )


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_training()



                # Beginning of XGBoost
# ............................................................
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import numpy as np
# from xgboost import XGBRegressor


# # -------------------------------
# # LOAD DATA
# # -------------------------------
# df = pd.read_csv("data/processed/final_selected_features.csv")

# print("Dataset shape:", df.shape)


# # -------------------------------
# # SPLIT FEATURES / TARGET
# # -------------------------------
# X = df.drop(columns=["log_total_cost"])
# y = df["log_total_cost"]


# # Remove ID column
# if "disasterNumber" in X.columns:
#     X = X.drop(columns=["disasterNumber"])


# # -------------------------------
# # TRAIN / TEST SPLIT
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# # -------------------------------
# # TRAIN MODEL (XGBoost)
# # -------------------------------
# model = XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)


# # -------------------------------
# # PREDICTION
# # -------------------------------
# y_pred = model.predict(X_test)


# # -------------------------------
# # EVALUATION
# # -------------------------------
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print("\nModel Performance:")
# print("RMSE:", rmse)
# print("R2 Score:", r2)
# print("MAE:", mae)
# print("MSE:", mse)

            # END OF XGBOOST
# .....................................................


# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
# importances = pd.Series(model.feature_importances_, index=X.columns)
# importances = importances.sort_values(ascending=False)

# print("\nTop Feature Importances:")
# print(importances.head(15))


# -------------------------------
# SAVE IMPORTANCE
# -------------------------------
# importances.to_csv("data/processed/feature_importance.csv")

# print("\nSaved: data/processed/feature_importance.csv")

