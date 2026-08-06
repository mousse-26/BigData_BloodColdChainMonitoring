import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["temp_range"] = df["temp_max"] - df["temp_min"]

    df["temp_cv"] = df["temp_std"] / (df["temp_mean"] + 1e-6)

    df["temp_deviation"] = abs(df["temp_mean"] - 4)

    df["humidity_per_temp"] = (
        df["hum_mean"] / (df["temp_mean"] + 1e-6)
    )

    df["handling_per_door"] = (
        df["handling_stress"] / (df["door_count"] + 1)
    )

    return df
