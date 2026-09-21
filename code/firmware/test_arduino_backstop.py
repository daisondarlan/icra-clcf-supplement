"""
Sends real, held-out feature rows to the physically-flashed Arduino Uno over
serial, compares its SAFE/NOT SAFE decision against a Python reference using
the identical Gaussian parameters, and records per-decision latency. This is
a hardware-feasibility check for the backstop layer (paper Section: Fail-
Closed Backstop on Embedded Hardware): can the fail-closed decision rule run
independently of the main compute stack, on hardware simple enough to be a
redundant safety channel.

Set the SERIAL_PORT environment variable to match your board (defaults to
COM4 on Windows; use e.g. /dev/ttyUSB0 or /dev/ttyACM0 on Linux/macOS).
"""
import json
import math
import os
import time
from pathlib import Path
import pandas as pd
import serial

PORT = os.environ.get("SERIAL_PORT", "COM4")
BAUD = 115200

PARAMS_PATH = Path(__file__).resolve().with_name("arduino_gaussian_params.json")
with PARAMS_PATH.open() as f:
    P = json.load(f)
FEATURES = ["angle_deg", "flatness_std", "obstacle_ratio"]


def log_gaussian_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.9189385332


def reference_classify(row):
    for feat in FEATURES:
        lo, hi = P[feat]["lo"], P[feat]["hi"]
        if row[feat] < lo or row[feat] > hi:
            return 0
    logp_safe = sum(log_gaussian_pdf(row[feat], P[feat]["mu_s"], P[feat]["sd_s"]) for feat in FEATURES)
    logp_unsafe = sum(log_gaussian_pdf(row[feat], P[feat]["mu_u"], P[feat]["sd_u"]) for feat in FEATURES)
    return 1 if logp_safe >= logp_unsafe else 0


def load_test_rows(n_per_session=40):
    files = {
        "session_E.csv": "Session E",
        "session_flat.csv": "Flat",
        "session_tilt.csv": "Tilt",
        "session_clutter.csv": "Clutter",
    }
    rows = []
    for fname, label in files.items():
        df = pd.read_csv(fname)
        sample = df.sample(n=min(n_per_session, len(df)), random_state=42)
        for _, r in sample.iterrows():
            rows.append((label, r))
    return rows


def main():
    rows = load_test_rows()
    print(f"Loaded {len(rows)} held-out test rows across 4 sessions")

    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2.2)  # let the Uno finish its post-upload reset
    ser.reset_input_buffer()

    agree = 0
    mismatches = []
    latencies_us = []

    for session, row in rows:
        line = f"{row['angle_deg']},{row['flatness_std']},{row['obstacle_ratio']}\n"
        ser.write(line.encode())
        reply = ser.readline().decode(errors="replace").strip()
        if not reply or reply == "ERR":
            print(f"  [WARN] no/bad reply for {session}: sent={line.strip()!r} reply={reply!r}")
            continue
        dec_str, us_str = reply.split(",")
        arduino_decision = int(dec_str)
        latency_us = int(us_str)
        latencies_us.append(latency_us)

        ref_decision = reference_classify(row)
        match = arduino_decision == ref_decision
        agree += int(match)
        if not match:
            mismatches.append((session, dict(row), arduino_decision, ref_decision))

    ser.close()

    n = len(latencies_us)
    print(f"\n{agree}/{n} Arduino decisions match the Python reference exactly")
    if mismatches:
        print("Mismatches:")
        for m in mismatches:
            print(" ", m)
    if latencies_us:
        print(f"Decision latency (on-board, micros()): "
              f"mean={sum(latencies_us)/n:.1f}us  min={min(latencies_us)}us  max={max(latencies_us)}us")


if __name__ == "__main__":
    main()
