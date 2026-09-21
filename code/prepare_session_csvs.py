#!/usr/bin/env python3
"""Split the bundled merged dataset into the eight per-session CSV filenames
expected by the CLCF evaluation scripts.

By default this reads ../data/landing_dataset_release.csv relative to this
file and writes the generated session_*.csv files into this code directory.
"""
from pathlib import Path
import argparse
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "data" / "landing_dataset_release.csv"
SESSION_FILES = {
    "A": "session_A.csv",
    "B": "session_B.csv",
    "C": "session_C.csv",
    "D": "session_D.csv",
    "E": "session_E.csv",
    "F_flat": "session_flat.csv",
    "G_tilt": "session_tilt.csv",
    "H_clutter": "session_clutter.csv",
}
EXPECTED_ROWS = {
    "A": 290, "B": 174, "C": 198, "D": 166, "E": 943,
    "F_flat": 720, "G_tilt": 529, "H_clutter": 438,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=HERE)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    required = {"session_id", "angle_deg", "flatness_std", "obstacle_ratio", "label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for session_id, filename in SESSION_FILES.items():
        sub = df[df["session_id"] == session_id].copy()
        expected = EXPECTED_ROWS[session_id]
        if len(sub) != expected:
            raise ValueError(
                f"{session_id}: expected {expected} rows, found {len(sub)} in {args.dataset}"
            )
        if "source_file" in sub.columns:
            sub["source_file"] = filename
        path = args.output_dir / filename
        sub.to_csv(path, index=False)
        total += len(sub)
        print(f"wrote {path.name}: {len(sub)} rows")

    print(f"prepared {len(SESSION_FILES)} session files ({total} rows total)")

if __name__ == "__main__":
    main()
