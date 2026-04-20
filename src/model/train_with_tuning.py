import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import Ridge, Lasso
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
# LOAD DATA
# -------------------------------
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

preprocessor = get_preprocessor()


# -------------------------------
# MODELS + HYPERPARAMETERS
# -------------------------------
models = [

    ('Ridge', Ridge(), {
        'regression__alpha': [0.1, 1.0, 10.0, 50.0]
    }),

    ('Lasso', Lasso(max_iter=5000), {
        'regression__alpha': [0.001, 0.01, 0.1, 1.0]
    }),

    ('Random Forest', RandomForestRegressor(random_state=42, n_jobs=-1), {
        'regression__n_estimators': [200, 300],
        'regression__max_depth': [15, 25],
        'regression__min_samples_leaf': [2, 5]
    }),

    ('XGBoost', XGBRegressor(random_state=42, n_jobs=-1), {
        'regression__n_estimators': [300, 500],
        'regression__learning_rate': [0.01, 0.03, 0.05],
        'regression__max_depth': [4, 6, 8],
        'regression__subsample': [0.7, 0.8, 1.0],
        'regression__colsample_bytree': [0.7, 0.8, 1.0]
    })
]


# -------------------------------
# TRAINING LOOP
# -------------------------------
def run_tuning():

    for run_name, model, param_grid in models:

        with mlflow.start_run(run_name=run_name):

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regression', model)
            ])

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=15,
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            # -------------------------------
            # TRAIN
            # -------------------------------
            search.fit(X_train, y_train)

            best_pipeline = search.best_estimator_
            y_pred = best_pipeline.predict(X_test)

            # -------------------------------
            # EVALUATE
            # -------------------------------
            metrics = evaluate_model(y_test, y_pred)

            # Print nicely
            print_metrics(run_name, metrics)

            # -------------------------------
            # LOG METRICS
            # -------------------------------
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # -------------------------------
            # LOG BEST PARAMETERS
            # -------------------------------
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model", model.__class__.__name__)

            # -------------------------------
            # LOG MODEL
            # -------------------------------
            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path=f"{model.__class__.__name__}_best_pipeline"
            )


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_tuning()