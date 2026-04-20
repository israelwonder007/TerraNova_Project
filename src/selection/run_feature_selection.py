import pandas as pd

from src.selection.feature_selection import (
    drop_constant_features,
    drop_high_missing,
    mutual_info_selection,
    prepare_xy
)

from src.selection.importance import tree_importance


# Load your engineered dataset
df = pd.read_csv("data/processed/fema_merged_dataset.csv")

print("Loaded dataset:", df.shape)

# -------------------------------
# CLEANING
# -------------------------------
df, dropped_const = drop_constant_features(df)
df, dropped_missing = drop_high_missing(df)

print("Dropped constant:", dropped_const)
print("Dropped missing:", dropped_missing)

# -------------------------------
# PREPARE DATA
# -------------------------------
X, y = prepare_xy(df)

# -------------------------------
# MUTUAL INFORMATION
# -------------------------------
mi_features, mi_scores = mutual_info_selection(X, y, top_k=20)

print("\nTop MI Features:")
print(mi_scores.head(20))

# -------------------------------
# TREE IMPORTANCE
# -------------------------------
tree_features, tree_scores = tree_importance(X, y, top_k=20)

print("\nTop Tree Features:")
print(tree_scores.head(20))

# -------------------------------
# FINAL FEATURES (INTERSECTION)
# -------------------------------
final_features = list(set(mi_features).union(set(tree_features)))

print("\nFinal Selected Features:")
print(final_features)


# remove target + ID from features first
final_features = [
    f for f in final_features
    if f not in ["log_total_cost", "log_total_cost.1", "disasterNumber"]
]

df_final = df[final_features + ["log_total_cost"]]

# Save final dataset
df_final.to_csv("data/processed/final_selected_features.csv", index=False)

print("\nSaved: data/processed/final_selected_features.csv")