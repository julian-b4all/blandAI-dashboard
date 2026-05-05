"""Add 48-hour callback flag to calls_drivers.csv

For each call, checks if the same caller phone number called back
within 48 hours. This is the industry-standard signal for whether
a bot interaction actually resolved the caller's issue.

true_resolution = non-transfer call with no callback within 48h
"""

import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DRIVERS = PROJECT / "data" / "calls_drivers.csv"
SUMMARY = PROJECT / "data" / "calls_summary.csv"
OUTPUT = DRIVERS  # overwrite in place

CALLBACK_WINDOW_H = 48


def main():
    # Load drivers + phone numbers from summary
    df = pd.read_csv(DRIVERS, encoding="utf-8")
    summary = pd.read_csv(SUMMARY, encoding="utf-8", usecols=["call_id", "from", "to"])
    df = df.merge(summary, on="call_id", how="left")

    # Clean phone numbers — stored as float, convert to string
    df["caller_phone"] = df["from"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None)
    df.drop(columns=["from", "to"], inplace=True)

    # Parse timestamps (format='mixed' handles both ISO and pandas datetime strings)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True, format="mixed")
    df = df.sort_values("created_at")

    # For each call, check if same caller_phone appears again within 48h
    df["callback_48h"] = False
    df["hours_to_callback"] = None

    # Group by caller phone and check for repeat contacts
    for phone, group in df[df["caller_phone"].notna()].groupby("caller_phone"):
        if len(group) < 2:
            continue
        times = group["created_at"].tolist()
        indices = group.index.tolist()

        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                delta_h = (times[j] - times[i]).total_seconds() / 3600
                if delta_h <= CALLBACK_WINDOW_H:
                    # This call had a callback within 48h
                    df.loc[indices[i], "callback_48h"] = True
                    df.loc[indices[i], "hours_to_callback"] = round(delta_h, 1)
                    break  # only need the first callback

    # Compute true_resolution flag
    # A call is truly resolved if: not transferred, not abandoned, and no callback within 48h
    df["true_resolution"] = (
        df["resolution"].isin(["resolved", "partially_resolved"])
        & ~df["callback_48h"]
    )

    # Stats
    meaningful = df[
        ~df["resolution"].isin(["no_interaction"])
        & (df["component"] != "call routing")
    ]
    n = len(meaningful)
    n_haiku_resolved = len(meaningful[meaningful["resolution"].isin(["resolved", "partially_resolved"])])
    n_true_resolved = len(meaningful[meaningful["true_resolution"]])
    n_callbacks = len(meaningful[meaningful["callback_48h"]])

    print(f"Meaningful calls: {n}")
    print(f"Haiku 'resolved':  {n_haiku_resolved} ({n_haiku_resolved/n*100:.1f}%)")
    print(f"Had callback <48h: {n_callbacks} ({n_callbacks/n*100:.1f}%)")
    print(f"True resolution:   {n_true_resolved} ({n_true_resolved/n*100:.1f}%)")
    print(f"Gap (inflated by):  {n_haiku_resolved - n_true_resolved} calls "
          f"({(n_haiku_resolved - n_true_resolved)/n*100:.1f}pp)")

    # Save
    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"\nSaved to {OUTPUT}")
    print(f"New columns: caller_phone, callback_48h, hours_to_callback, true_resolution")


if __name__ == "__main__":
    main()
