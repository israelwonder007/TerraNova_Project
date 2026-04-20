import pandas as pd

def aggregate_public_assistance(df_pa: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PA data to disaster level.
    """
    df_pa_agg = df_pa.groupby("disasterNumber").agg({
        "projectAmount": "sum",
        "totalObligated": "sum",
        "federalShareObligated": "sum",
        "pwNumber": "count"
    }).rename(columns={
        "pwNumber": "num_projects"
    }).reset_index()

    return df_pa_agg