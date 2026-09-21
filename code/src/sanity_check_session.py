#!/usr/bin/env python3
"""
Run this immediately after recording a new labeling session, in the field,
before packing up -- it catches the exact data-quality failure mode found
in this project's original five sessions (four of five had physically
implausible feature scales; the cause was narrowed but not fully confirmed,
see the paper's Check 3 section -- so treat any recurrence as a real
problem to investigate on the spot, not an already-understood quirk).

Usage: python sanity_check_session.py <path_to_new_session.csv>
"""
import sys
import pandas as pd

# Plausible ranges for a 1.524x1.524m ROI at ~2.5m altitude. These are wide
# on purpose (real terrain varies); the point is to catch order-of-magnitude
# problems, not to be a tight statistical test.
PLAUSIBLE_RANGES = {
    "flatness_std":   (0.0, 0.5),   # meters
    "obstacle_ratio": (0.0, 1.0),   # fraction, always valid by construction
    "angle_deg":      (0.0, 90.0),  # degrees
}


def main():
    if len(sys.argv) != 2:
        print("Usage: python sanity_check_session.py <path_to_new_session.csv>")
        raise SystemExit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    print(f"== {path} (n={len(df)}) ==")
    if "label" in df.columns:
        print(f"label distribution: {dict(df['label'].value_counts().sort_index())}")
        if df["label"].nunique() < 2:
            print("  NOTE: this session is single-class, as expected for one recording run.")

    problems = []
    for feature, (lo, hi) in PLAUSIBLE_RANGES.items():
        if feature not in df.columns:
            continue
        frac_out = ((df[feature] < lo) | (df[feature] > hi)).mean()
        print(f"{feature:16s} mean={df[feature].mean():.4f}  min={df[feature].min():.4f}  "
              f"max={df[feature].max():.4f}  fraction outside [{lo},{hi}] = {frac_out:.1%}")
        if frac_out > 0.05:
            problems.append(feature)

    # obstacle_ratio specifically: if this session is labeled Landable and
    # obstacle_ratio is high anyway, that's the exact failure signature found
    # in this project's original data (four of five sessions showed this).
    if "obstacle_ratio" in df.columns and "label" in df.columns:
        landable = df[df["label"] == 1]
        if len(landable) and landable["obstacle_ratio"].mean() > 0.3:
            problems.append("obstacle_ratio_high_on_landable")
            print(f"\n  WARNING: mean obstacle_ratio on Landable rows is "
                  f"{landable['obstacle_ratio'].mean():.3f} -- should be near 0. "
                  f"This is the exact signature that flagged four of five sessions "
                  f"as unreliable in this project's original dataset.")

    print()
    if problems:
        print(f"FLAGGED: {problems}")
        print("Do not trust this session's feature values without investigating "
              "before leaving the site -- check that Start was pressed only after "
              "the camera/scene had settled, and that the ROI is actually over "
              "the intended patch.")
        raise SystemExit(1)
    else:
        print("No red flags detected. Values are in the physically plausible range.")


if __name__ == "__main__":
    main()
