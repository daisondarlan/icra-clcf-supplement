# CLCF Reference Implementation

Reference implementation and evaluation code for the **Cross-Layer Consistency
Framework (CLCF)** used in the accompanying double-masked submission.

This release is intentionally scoped to the reference code used for the
paper's checks, quantitative evaluations, and figures. Data-capture GUIs and
one-off development utilities are not required for those results and are not
included.

## Layout

```text
code/
├── prepare_session_csvs.py                # split bundled merged dataset into the 8 session files
├── src/
│   ├── features.py                        # ROI sizing and geometric feature extraction
│   ├── temporal_gate.py                   # N-consecutive-frame decision gate
│   ├── sanity_check_session.py            # C3: physical-range session check
│   ├── evaluate_classifiers.py            # C1-C2: splits, LOSO, model comparison, ablations
│   ├── simulate_depth_scenes.py           # C4: synthetic altitude sweep
│   ├── rebuild_likelihoods_loso_check.py  # C5-C6: stale/rebuilt decision-table analysis
│   ├── deployed_pretix_ranges.json        # historical deployed pre-fix table ranges
│   └── plot_style.py                      # shared plotting style
├── tests/
│   └── test_temporal_gate.py
├── firmware/
│   ├── arduino_backstop/arduino_backstop.ino
│   ├── arduino_gaussian_params.json
│   └── test_arduino_backstop.py
├── requirements.txt
└── LICENSE
```

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r code/requirements.txt
python code/prepare_session_csvs.py
cd code
```

`prepare_session_csvs.py` reads the released
`../data/landing_dataset_release.csv` and reconstructs the eight anonymized
per-session CSV files expected by the evaluation scripts. The helper verifies
all expected session row counts before writing anything.

## Running the released analysis

Run these commands from `code/` after the setup step above:

```bash
# C1-C2: naive split diagnostic, session-grouped evaluation, LOSO, ablations
PYTHONPATH=src python src/evaluate_classifiers.py

# C4: synthetic altitude sweep and feature/ROI checks
PYTHONPATH=src python src/simulate_depth_scenes.py

# C5-C6: decision-table staleness and leave-one-session-out rebuild
PYTHONPATH=src python src/rebuild_likelihoods_loso_check.py

# C3: range check for any newly collected session
PYTHONPATH=src python src/sanity_check_session.py path/to/new_session.csv

# Temporal-gate synthetic tests
PYTHONPATH=src python tests/test_temporal_gate.py
```

The scripts print their numeric results and write the corresponding figures
and result files into the current working directory.

## Embedded backstop check

Flash `firmware/arduino_backstop/arduino_backstop.ino` to an Arduino Uno (or
compatible AVR board), then from `code/` run:

```bash
SERIAL_PORT=/dev/ttyACM0 python firmware/test_arduino_backstop.py
```

On Windows, set `SERIAL_PORT=COM8` (or the appropriate port). The harness
replays held-out real feature rows, compares the board's SAFE/NOT SAFE output
with the Python reference using the same Gaussian parameters, and reports
agreement and on-board decision latency.

## Data

The released dataset is `../data/landing_dataset_release.csv` (3,458 labeled
rows across eight sessions). See `../data/README.md` for schema, collection
phases, and the documented Phase-1 limitation.

## License

MIT. See `LICENSE`.

## Citation

Citation details are withheld during double-masked review and will be added on
publication.
