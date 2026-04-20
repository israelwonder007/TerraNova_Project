import pandas as pd
import requests
import os

print("Script started...")

# =========================================================
# CONFIG (moved here instead of config.py)
# =========================================================

BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "declarations": {
        "url": f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
        "params": {
            "$format": "json",
            "$allrecords": "true"
        },
        "key": "DisasterDeclarationsSummaries",
        "select": [
            "disasterNumber", "state", "declarationType", "fyDeclared",
            "incidentType", "incidentBeginDate", "incidentEndDate",
            "declarationDate", "ihProgramDeclared", "iaProgramDeclared",
            "paProgramDeclared", "hmProgramDeclared",
            "fipsStateCode", "fipsCountyCode", "designatedArea"
        ]
    },

    "public_assistance": {
        "url": f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
        "params": {
            "$format": "json",
            "$allrecords": "true"
        },
        "key": "PublicAssistanceFundedProjectsDetails",
        "select": [
            "disasterNumber",'incidentType', "pwNumber", "projectAmount",
            "totalObligated", "federalShareObligated",
            "damageCategoryCode", "damageCategoryDescrip",
            "projectSize", "county", "state", "lastObligationDate"
        ]
    },

    "disaster_summaries": {
        "url": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
        "params": {
            "$format": "json",
            "$allrecords": "true"
        },
        "key": "FemaWebDisasterSummaries",
        "select": [
            "disasterNumber",
            "totalAmountIhpApproved",
            "totalObligatedAmountPa",
            "totalObligatedAmountHmgp"
        ]
    }
}

# =========================================================
# API FETCH
# =========================================================

def fetch_data(config):
    print("Calling API...")

    response = requests.get(config["url"], params=config["params"])
    print("Status code:", response.status_code)

    response.raise_for_status()
    data = response.json()

    return pd.DataFrame(data[config["key"]])


# =========================================================
# SAVE CSV
# =========================================================

def save_csv(df, name, config):
    os.makedirs("data/raw", exist_ok=True)

    if "select" in config:
        cols = config["select"]
        df = df[[c for c in cols if c in df.columns]]

    filepath = f"data/raw/{name}.csv"
    df.to_csv(filepath, index=False)

    print(f"{filepath} saved ({len(df)} rows)")


# =========================================================
# MAIN INGESTION FUNCTION (USED BY PIPELINE)
# =========================================================

def load_all_data():
    datasets = {}

    for name, config in ENDPOINTS.items():
        try:
            print(f"\nFetching {name}...")

            df = fetch_data(config)
            save_csv(df, name, config)

            datasets[name] = df

        except Exception as e:
            raise RuntimeError(f"Failed to load {name}: {e}")
        
    
    print("Loaded datasets:", list(datasets.keys()))
    print("\nIngestion complete.")
    return datasets


# =========================================================
# STANDALONE TEST RUN
# =========================================================

if __name__ == "__main__":
    data = load_all_data()

    for name, df in data.items():
        print(f"\n{name} preview:")
        print(df.head())