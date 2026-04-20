import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def tree_importance(X, y, top_k=20):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    selected_features = importance.head(top_k).index.tolist()

    return selected_features, importance