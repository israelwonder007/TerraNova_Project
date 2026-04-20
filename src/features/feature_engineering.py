import pandas as pd
import numpy as np
from src.features.aggregations import aggregate_public_assistance


# -------------------------------
# TIME FEATURES
# -------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["declarationDate"] = pd.to_datetime(df["declarationDate"], errors="coerce")

    df["year"] = df["declarationDate"].dt.year
    df["month"] = df["declarationDate"].dt.month
    df["quarter"] = df["declarationDate"].dt.quarter

    return df


# -------------------------------
# DURATION FEATURE
# -------------------------------
def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    df["incidentBeginDate"] = pd.to_datetime(df["incidentBeginDate"], errors="coerce")
    df["incidentEndDate"] = pd.to_datetime(df["incidentEndDate"], errors="coerce")

    df["incident_duration_days"] = (
        df["incidentEndDate"] - df["incidentBeginDate"]
    ).dt.days.clip(lower=0).fillna(0)

    # # remove negative durations (bad data)
    # df["incident_duration_days"] = df["incident_duration_days"].clip(lower=0)

    # # flag missing AFTER aggregation
    # df["duration_missing"] = df["incident_duration_days"].isna().astype(int)

    return df


# -------------------------------
# PROGRAM FLAGS
# -------------------------------
def encode_programs(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",
    ]

    for col in cols:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    return df

# -------------------------------
# CATEGORICAL ENCODING
# -------------------------------
# def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:

    # ......................................................
    
    # if "incidentType" in df.columns:
    #     df["incidentType"] = df["incidentType"].astype("category")
    #     # df["incidentType"] = df["incidentType"].astype("category").cat.codes
    # return df
    # ......................................................
    # Group rare categories
    # top_types = ["Hurricane", "Severe Storm", "Flood", "Fire"]

    # if "incidentType" in df.columns:
    #     df["incidentType"] = df["incidentType"].apply(
    #         lambda x: x if x in top_types else "Other"
    #     )

    #     # One-hot encoding (BETTER than label encoding)
    #     dummies = pd.get_dummies(df["incidentType"], prefix="incidentType")
    #     df = pd.concat([df, dummies], axis=1)

    #     # Drop original column
    #     df.drop(columns=["incidentType"], inplace=True)

    # return df

def add_incident_features(df):

    # Cost tiers (based on your analysis)
    def cost_tier(x):
        if x == "Hurricane":
            return "very_high"
        elif x in ["Tropical Storm", "Severe Ice Storm", "Earthquake"]:
            return "high"
        elif x in ["Severe Storm", "Flood", "Fire"]:
            return "medium"
        else:
            return "low"

    df["incident_cost_tier"] = df["incidentType"].apply(cost_tier)

    # One-hot encode
    dummies = pd.get_dummies(df["incident_cost_tier"], prefix="tier")

    df = pd.concat([df, dummies], axis=1)

    return df


# -------------------------------
# LOCATION FEATURES
# -------------------------------
def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df["location_key"] = df["designatedArea"].astype(str) + "_" + df["state"].astype(str)
    return df


# -------------------------------
# TARGET ENGINEERING
# -------------------------------
def create_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    df["total_cost"] = (
        df["totalObligatedAmountPa"].fillna(0)
        + df["totalObligatedAmountHmgp"].fillna(0)
        + df["totalAmountIhpApproved"].fillna(0)
    )

    # FIX: remove negative values before log
    df["total_cost"] = df["total_cost"].clip(lower=0)

    df["log_total_cost"] = np.log1p(df["total_cost"])
    return df


# -------------------------------
# MAIN FEATURE BUILDER
# -------------------------------
def build_features(df_decl, df_sum, df_pa):
    """
    Build final modeling dataset (1 row per disaster)
    """

    df_pa_agg = aggregate_public_assistance(df_pa)

    df = df_decl.merge(df_sum, on="disasterNumber", how="left")
    df = df.merge(df_pa_agg, on="disasterNumber", how="left")

    df = add_time_features(df)
    df = add_duration(df)
    df = encode_programs(df)
    df = add_incident_features(df)
    # df = encode_categoricals(df)
    df = add_location_features(df)

    df = create_total_cost(df)

    # FORCE DISASTER-LEVEL DATA
    df = df.groupby("disasterNumber").agg({
        "year": "first",
        "month": "first",
        "quarter": "first",
        "incident_duration_days": "mean",

        # "incidentType": "first",
        #  ONE-HOT COLUMNS
        # "incidentType_Hurricane": "max",
        # "incidentType_Severe Storm": "max",
        # "incidentType_Flood": "max",
        # "incidentType_Fire": "max",
        # "incidentType_Other": "max",


        "ihProgramDeclared": "max",
        "iaProgramDeclared": "max",
        "paProgramDeclared": "max",
        "hmProgramDeclared": "max",
        "location_key": "nunique",
            # ADD THESE
        "tier_very_high": "max",
        "tier_high": "max",
        "tier_medium": "max",
        "tier_low": "max",

        "total_cost": "sum",
        "log_total_cost": "first"
    }).reset_index().rename(
    columns={"location_key": "num_counties"})

    df["duration_x_counties"] = (
    df["incident_duration_days"] * df["num_counties"]
    )

    df = df.loc[:, ~df.columns.duplicated()]

    assert df["disasterNumber"].is_unique, "Dataset is NOT at disaster level!"

    return df


# -------------------------------
# TEST RUN
# -------------------------------
if __name__ == "__main__":
    print("Running feature engineering test...")

    df_decl = pd.read_csv("data/raw/declarations.csv")
    df_pa = pd.read_csv("data/raw/public_assistance.csv")
    df_sum = pd.read_csv("data/raw/disaster_summaries.csv")

    df_model = build_features(df_decl, df_sum, df_pa)

    print("Final dataset shape:", df_model.shape)
    print(df_model.head())



# ....................................

    # def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:

    # top_types = {"Hurricane", "Severe Storm", "Flood", "Fire"}

    # if "incidentType" in df.columns:

    #     # -------------------------------
    #     # 1. BINARY FEATURE (AGGREGATED SIGNAL)
    #     # -------------------------------
    #     df["is_high_cost_type"] = df["incidentType"].isin(top_types).astype(int)

    #     # -------------------------------
    #     # 2. GROUP RARE CATEGORIES
    #     # -------------------------------
    #     df["incidentType"] = df["incidentType"].apply(
    #         lambda x: x if x in top_types else "Other"
    #     )

    #     # -------------------------------
    #     # 3. ONE-HOT ENCODING (DETAILED FEATURES)
    #     # -------------------------------
    #     dummies = pd.get_dummies(
    #         df["incidentType"], prefix="incidentType"
    #     ).astype(int)

    #     df = pd.concat([df, dummies], axis=1)

    #     # -------------------------------
    #     # 4. DROP ORIGINAL COLUMN
    #     # -------------------------------
    #     df.drop(columns=["incidentType"], inplace=True)

    # return df