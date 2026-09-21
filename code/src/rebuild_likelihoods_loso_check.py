"""
Quantifies the likelihood-table staleness bug and its fix (Checks 5-6 in the
paper), using the same leave-one-session-out discipline as
evaluate_classifiers.py: for each of the three post-fix real sessions held
out in turn, rebuild angle_deg/flatness_std/obstacle_ratio likelihood tables
from the OTHER two, then measure what fraction of the held-out session's
rows would be marked out-of-range (and therefore fail-closed) under the OLD
deployed table vs. the NEW, rebuilt-from-corrected-data table.

The "NEW" rate (Table V, "LOO rebuild" column) is a mean +/- SD over all
three non-empty training subsets of the other two sessions (each session
alone, and both combined), not one fixed pairing -- a single fixed pairing
understates the problem (see the paper's Sec. III-G for the single-ordering
number this reruns).

Expects session_flat.csv, session_tilt.csv, session_clutter.csv in the
working directory. The deployed pre-fix table's range per feature (the
"Old" column of Table V) is a fixed historical value from before the
tilt-feature fix, loaded from deployed_pretix_ranges.json alongside this
script rather than rebuilt, since the pre-fix source data predates this
repository.
"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import plot_style as ps
import matplotlib.pyplot as plt

FEATURES = ["angle_deg", "flatness_std", "obstacle_ratio"]
SESSIONS = {
    "session_flat.csv": "session_flat.csv",
    "session_tilt.csv": "session_tilt.csv",
    "session_clutter.csv": "session_clutter.csv",
}

dfs = {name: pd.read_csv(fname) for name, fname in SESSIONS.items()}

_ranges_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployed_pretix_ranges.json")
with open(_ranges_path) as f:
    _deployed_ranges = json.load(f)
old_range = {feat: (_deployed_ranges[feat]["lo"], _deployed_ranges[feat]["hi"]) for feat in FEATURES}

print("Deployed (pre-fix) table ranges:")
for feat, (lo, hi) in old_range.items():
    print(f"  {feat:15s} {lo:.4f} - {hi:.4f}")

print("\nLeave-one-session-out: rebuild table from the other two post-fix sessions, "
      "check held-out session's out-of-range rate.\n")

old_pcts = {}   # (session_label, feat) -> pct
new_pcts = {}   # (session_label, feat) -> mean pct over the 3 subsets
new_pcts_sd = {}
session_level_means = []  # one mean per held-out session, for the table's overall Mean+-SD row
SESSION_LABELS = {"session_flat.csv": "Landable", "session_tilt.csv": "Tilt",
                   "session_clutter.csv": "Clutter"}

for held_out in SESSIONS:
    others = [s for s in SESSIONS if s != held_out]
    # All 3 non-empty subsets of the 2 other sessions: each alone, and both combined.
    train_subsets = [[others[0]], [others[1]], others]
    test = dfs[held_out]
    print(f"== held out: {held_out} (n={len(test)}) ==")
    per_session_feat_means = []
    for feat in FEATURES:
        old_lo, old_hi = old_range[feat]
        old_pct = ((test[feat] < old_lo) | (test[feat] > old_hi)).mean() * 100
        subset_pcts = []
        for subset in train_subsets:
            train = pd.concat([dfs[s] for s in subset], ignore_index=True)
            new_lo, new_hi = train[feat].min(), train[feat].max()
            subset_pcts.append(((test[feat] < new_lo) | (test[feat] > new_hi)).mean() * 100)
        mean_pct = float(np.mean(subset_pcts))
        sd_pct = float(np.std(subset_pcts, ddof=0))
        old_pcts[(SESSION_LABELS[held_out], feat)] = old_pct
        new_pcts[(SESSION_LABELS[held_out], feat)] = mean_pct
        new_pcts_sd[(SESSION_LABELS[held_out], feat)] = sd_pct
        per_session_feat_means.append(mean_pct)
        print(f"  {feat:15s} OLD table out-of-range: {old_pct:5.1f}%   "
              f"NEW table (mean +/- SD over 3 training subsets): {mean_pct:5.1f} +/- {sd_pct:4.1f}%   "
              f"[per-subset: {[f'{p:.1f}' for p in subset_pcts]}]")
    session_level_means.append(float(np.mean(per_session_feat_means)))
    print()

overall_old_mean = float(np.mean(list(old_pcts.values())))
overall_new_mean = float(np.mean(session_level_means))
overall_new_sd = float(np.std(session_level_means, ddof=0))
print(f"Mean across all sessions/features: OLD {overall_old_mean:.1f}%   "
      f"NEW {overall_new_mean:.1f} +/- {overall_new_sd:.1f}%")

# Figure: rows outside the DEPLOYED (pre-fix) table's range, by session and
# feature -- the headline staleness result.
sessions = ["Landable", "Tilt", "Clutter"]
feats = FEATURES
x = np.arange(len(sessions))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"angle_deg": ps.RED, "flatness_std": ps.BLUE, "obstacle_ratio": ps.GREEN}
feat_display = {"angle_deg": "angle_deg (the fixed feature)", "flatness_std": "flatness_std",
                 "obstacle_ratio": "obstacle_ratio"}
for i, feat in enumerate(feats):
    vals = [old_pcts[(s, feat)] for s in sessions]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=feat_display[feat],
                   color=colors[feat], edgecolor="black", linewidth=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(sessions)
ax.set_ylabel("Rows outside deployed table's range (%)")
ax.set_ylim(0, 112)
ax.set_title("Deployed decision table rejects post-fix data")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=1, frameon=True)
ax.grid(axis="y")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig_likelihood_staleness.png")
plt.close(fig)
print("Wrote fig_likelihood_staleness.png")
