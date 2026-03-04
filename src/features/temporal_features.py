import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical and temporal features from windowed sequences.

    Input columns expected:
        block_id
        window_start
        event_count
        error_count
        warn_count
        info_count
        event_sequence
    """

    df = df.copy()

    # Basic statistical features
    df["error_rate"] = df["error_count"] / df["event_count"].clip(lower=1)
    df["warn_rate"] = df["warn_count"] / df["event_count"].clip(lower=1)

    # Unique event types
    df["unique_event_types"] = df["event_sequence"].apply(
        lambda x: len(set(x.split()))
    )

    # Rare event proxy (events appearing once in sequence)
    def rare_event_count(seq):
        tokens = seq.split()
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return sum(1 for c in counts.values() if c == 1)

    df["rare_event_count"] = df["event_sequence"].apply(rare_event_count)

    # Burst activity signal
    df = df.sort_values(["block_id", "window_start"])

    df["prev_event_count"] = df.groupby("block_id")["event_count"].shift(1)
    df["burst_ratio"] = df["event_count"] / df["prev_event_count"].clip(lower=1)

    df["burst_ratio"] = df["burst_ratio"].fillna(1.0)

    # Transition count proxy (sequence complexity)
    def transition_count(seq):
        tokens = seq.split()
        return max(len(tokens) - 1, 0)

    df["transition_count"] = df["event_sequence"].apply(transition_count)

    return df