import pandas as pd


def apply_forward_failure_labeling(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """
    Converts anomaly labels into forward-horizon failure labels.

    If an anomaly occurs within the next `horizon` windows for the same block,
    current window is labeled as failure prediction target.

    Parameters
    ----------
    df : DataFrame with columns
        block_id
        window_start
        label (anomaly label)

    horizon : number of future windows to check
    """

    df = df.copy()

    df = df.sort_values(["block_id", "window_start"])

    df["forward_failure"] = 0

    for h in range(1, horizon + 1):
        future = df.groupby("block_id")["label"].shift(-h)
        df["forward_failure"] = df["forward_failure"] | (future == 1)

    df["forward_failure"] = df["forward_failure"].astype(int)

    return df