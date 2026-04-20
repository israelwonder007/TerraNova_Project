import os
from src.data_ingestion.ingestion import load_all_data
from src.features.feature_engineering import build_features

def run_pipeline():

    print("\n=== STEP 1: INGESTION ===")

    datasets = load_all_data()

    df_decl = datasets["declarations"]
    df_pa = datasets["public_assistance"]
    df_sum = datasets["disaster_summaries"]

    print("\n=== STEP 2: FEATURE ENGINEERING ===")
    df_model = build_features(df_decl, df_sum, df_pa)

    # save engineered dataset
    os.makedirs("data/processed", exist_ok=True)
    df_model.to_csv("data/processed/fema_merged_dataset.csv", index=False)

    print("\n=== PIPELINE COMPLETE ===")
    print("\nSaved: data/processed/fema_merged_dataset.csv")
    print("Final dataset shape:", df_model.shape)

    return df_model


if __name__ == "__main__":
    run_pipeline()