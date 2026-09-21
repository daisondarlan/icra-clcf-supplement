"""
Controlled synthetic depth-scene validation of the altitude-adaptive ROI
(Check 4 in the paper).

Motivation: the real field sessions never varied altitude much, so nothing
in the real data can test whether the altitude-adaptive ROI actually keeps
classification correct as altitude changes, which is a central claim of the
pipeline. This script builds a small, fully synthetic depth-scene
generator -- flat ground plus known-height obstacles and known-angle tilt,
placed by construction, so the ground-truth Landable/Not-Landable label
comes from the placed geometry, never from the features the classifier
computes. It calls the SAME feature-extraction functions used by the
deployed classifier (via features.py), not a reimplementation, and sweeps
the same physical scenes across a 1-5 m altitude range to compare the
altitude-adaptive ROI against a fixed-altitude baseline.

This is a controlled sensitivity study, not real-world evidence: the scenes
are simple geometric primitives (flat plane + circular bumps + planar
tilt), not photorealistic, and the sensor-noise model is a single-point
calibration against one real session (Session E), not an empirically fit
noise curve. Session E's real feature rows are also run through the
resulting classifiers as a modest real-sensor check, not a generalization
claim.
"""
import math
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import plot_style as ps
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from features import (
    HEIGHT_M, RES_X, RES_Y, FOV_X, FOV_Y, AREA_SIZE_M, MIN_VALID_FRACTION,
    OBSTACLE_THRESH, estimate_altitude, get_crop_indices_for_altitude,
    get_crop_indices, fit_plane_angle, compute_flatness_std,
    compute_obstacle_ratio,
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

SESSION_E_CSV = "session_E.csv"

# Ground-truth hazard definitions -- independent of OBSTACLE_THRESH (used only
# inside the feature extractor). These are the "would a human call this
# unsafe" thresholds used purely to label the synthetic scenes.
SAFETY_HEIGHT_M = 0.03     # obstacle height inside the landing footprint
SAFETY_TILT_DEG = 10.0     # slope beyond typical UAV landing-gear tolerance

ALTITUDES_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
N_SCENES_PER_ALTITUDE = 300


def calibrate_noise_constant():
    """Single-point calibration of an altitude-dependent depth-noise model
    (sigma_Z = Z^2 / K, quadratic growth with range) against Session E, the
    one real session not flagged as data-quality compromised."""
    df = pd.read_csv(SESSION_E_CSV)
    z = float(df["altitude_m"].mean())
    sigma = float(df["flatness_std"].mean())
    k = z ** 2 / sigma
    return k, z, sigma


def meters_per_pixel(altitude_m):
    mx = (2 * altitude_m * math.tan(math.radians(FOV_X / 2))) / RES_X
    my = (2 * altitude_m * math.tan(math.radians(FOV_Y / 2))) / RES_Y
    return mx, my


def sample_scene_spec(rng):
    """Sample one scene definition in physical (metre/degree) units, so the
    same scene can be rendered at any altitude."""
    is_hazard = rng.random() < 0.5

    if not is_hazard:
        tilt_deg = float(np.abs(rng.normal(0, 2.0)))          # small, safe
        n_obstacles = rng.integers(0, 2)
        obstacles = []
        for _ in range(n_obstacles):
            # keep height/placement below the hazard thresholds
            obstacles.append(dict(
                cx_m=float(rng.uniform(-0.9, 0.9)),
                cy_m=float(rng.uniform(-0.9, 0.9)),
                radius_m=float(rng.uniform(0.03, 0.12)),
                height_m=float(rng.uniform(0.0, SAFETY_HEIGHT_M * 0.6)),
            ))
    else:
        # tilt hazard, obstacle hazard, or both
        kind = rng.choice(["tilt", "obstacle", "both"])
        tilt_deg = float(rng.uniform(SAFETY_TILT_DEG + 2, 35)) if kind in ("tilt", "both") \
            else float(np.abs(rng.normal(0, 2.0)))
        obstacles = []
        if kind in ("obstacle", "both"):
            n_obstacles = rng.integers(1, 3)
            for _ in range(n_obstacles):
                obstacles.append(dict(
                    cx_m=float(rng.uniform(-0.5, 0.5)),   # biased into the footprint
                    cy_m=float(rng.uniform(-0.5, 0.5)),
                    radius_m=float(rng.uniform(0.08, 0.25)),
                    height_m=float(rng.uniform(SAFETY_HEIGHT_M * 1.5, 0.5)),
                ))
        else:
            # tilt-only hazard: still add 0-1 harmless clutter obstacle for realism
            for _ in range(rng.integers(0, 2)):
                obstacles.append(dict(
                    cx_m=float(rng.uniform(-1.2, 1.2)),
                    cy_m=float(rng.uniform(-1.2, 1.2)),
                    radius_m=float(rng.uniform(0.03, 0.1)),
                    height_m=float(rng.uniform(0.0, SAFETY_HEIGHT_M * 0.6)),
                ))

    tilt_axis_deg = float(rng.uniform(0, 360))
    return dict(tilt_deg=tilt_deg, tilt_axis_deg=tilt_axis_deg, obstacles=obstacles)


def ground_truth_label(spec):
    if spec["tilt_deg"] >= SAFETY_TILT_DEG:
        return "NOT SAFE"
    half = AREA_SIZE_M / 2
    for obs in spec["obstacles"]:
        if obs["height_m"] <= SAFETY_HEIGHT_M:
            continue
        # approximate circle-square overlap with the central landing footprint
        if abs(obs["cx_m"]) - obs["radius_m"] < half and abs(obs["cy_m"]) - obs["radius_m"] < half:
            return "NOT SAFE"
    return "SAFE"


def render_depth_image(altitude_m, spec, noise_k, rng):
    mx, my = meters_per_pixel(altitude_m)
    ys, xs = np.mgrid[0:RES_Y, 0:RES_X]
    x_m = (xs - RES_X / 2) * mx
    y_m = (ys - RES_Y / 2) * my

    tilt_rad = math.radians(spec["tilt_deg"])
    axis_rad = math.radians(spec["tilt_axis_deg"])
    d = x_m * math.cos(axis_rad) + y_m * math.sin(axis_rad)
    depth = altitude_m + d * math.tan(tilt_rad)

    for obs in spec["obstacles"]:
        dx = x_m - obs["cx_m"]
        dy = y_m - obs["cy_m"]
        mask = (dx ** 2 + dy ** 2) <= obs["radius_m"] ** 2
        depth[mask] -= obs["height_m"]

    noise_sigma = (altitude_m ** 2) / noise_k
    depth = depth + rng.normal(0, noise_sigma, depth.shape)
    return np.clip(depth, 0.01, None)


def extract_with_roi(depth_image, x0, y0, w, h, altitude_for_angle_m=None,
                      obstacle_thresh=OBSTACLE_THRESH):
    """altitude_for_angle_m: altitude used to convert the plane fit's
    pixel-space slope into a real physical slope (see fit_plane_angle) --
    pass the same altitude assumption the ROI itself was sized from, so the
    fixed-ROI baseline's angle reflects what that naive system would
    actually compute, pixel-pitch mistake included."""
    crop = depth_image[y0:y0 + h, x0:x0 + w]
    min_valid_points = int(MIN_VALID_FRACTION * w * h)
    px = py = None
    if altitude_for_angle_m is not None:
        px, py = meters_per_pixel(altitude_for_angle_m)
    angle = fit_plane_angle(crop, min_valid_points, px, py)
    flatness = compute_flatness_std(crop, min_valid_points)
    obstacle = compute_obstacle_ratio(crop, obstacle_thresh, min_valid_points)
    return angle, flatness, obstacle


def scaled_obstacle_thresh(altitude_m, cal_z):
    """Proposed fix: scale OBSTACLE_THRESH up with the same quadratic
    depth-noise growth used to calibrate the simulator's noise model, so the
    threshold represents a constant number of noise standard deviations at
    altitudes beyond the calibration point instead of a constant absolute
    distance. Floored at OBSTACLE_THRESH itself (never scaled down) --
    an earlier version without the floor also shrank the threshold below the
    calibration altitude, which made it more sensitive than intended to
    ordinary small-scale terrain unevenness that does not shrink with
    altitude the way sensor noise does. Scaling the threshold down is not
    safe; only scaling it up in response to worse sensing at range is."""
    return max(OBSTACLE_THRESH, OBSTACLE_THRESH * (altitude_m / cal_z) ** 2)


def build_dataset(noise_k, cal_z):
    rows = []
    for altitude in ALTITUDES_SWEEP:
        for i in range(N_SCENES_PER_ALTITUDE):
            spec = sample_scene_spec(rng)
            label = ground_truth_label(spec)
            depth_image = render_depth_image(altitude, spec, noise_k, rng)

            est_altitude = estimate_altitude(depth_image)
            if est_altitude is None:
                x0a, y0a, wa, ha = get_crop_indices()
            else:
                x0a, y0a, wa, ha = get_crop_indices_for_altitude(est_altitude)
            angle_a, flat_a, obs_a = extract_with_roi(
                depth_image, x0a, y0a, wa, ha, altitude_for_angle_m=est_altitude)

            obs_a_scaled = None
            if est_altitude is not None:
                thresh_scaled = scaled_obstacle_thresh(est_altitude, cal_z)
                _, _, obs_a_scaled = extract_with_roi(
                    depth_image, x0a, y0a, wa, ha, altitude_for_angle_m=est_altitude,
                    obstacle_thresh=thresh_scaled)

            x0f, y0f, wf, hf = get_crop_indices()
            angle_f, flat_f, obs_f = extract_with_roi(
                depth_image, x0f, y0f, wf, hf, altitude_for_angle_m=HEIGHT_M)

            rows.append(dict(
                true_altitude=altitude, est_altitude=est_altitude, label=label,
                angle_a=angle_a, flat_a=flat_a, obs_a=obs_a, obs_a_scaled=obs_a_scaled,
                angle_f=angle_f, flat_f=flat_f, obs_f=obs_f,
            ))
    return pd.DataFrame(rows)


def to_feature_matrix(df, cols):
    X = df[cols].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    return X, valid


def fit_and_evaluate(df, cols, model_cls):
    X, valid = to_feature_matrix(df, cols)
    y = (df["label"] == "SAFE").to_numpy()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    train_valid = train_idx[valid[train_idx]]
    model = model_cls()
    model.fit(X[train_valid], y[train_valid])

    # Predict on the FULL test set; invalid-feature rows fail closed to
    # NOT SAFE, matching the deployed classifier's real behaviour, rather
    # than being dropped from evaluation.
    pred = np.zeros(len(test_idx), dtype=bool)
    test_valid_mask = valid[test_idx]
    if test_valid_mask.any():
        pred[test_valid_mask] = model.predict(X[test_idx][test_valid_mask])

    result = df.iloc[test_idx].copy()
    result["pred"] = np.where(pred, "SAFE", "NOT SAFE")
    result["correct"] = result["pred"] == result["label"]
    return model, result


def accuracy_by_altitude(result_df):
    return result_df.groupby("true_altitude")["correct"].mean()


def hazard_recall_by_altitude(result_df):
    """Fraction of ground-truth NOT SAFE scenes correctly flagged NOT SAFE --
    the safety-critical number (a miss here means landing on a real hazard)."""
    hazard = result_df[result_df["label"] == "NOT SAFE"]
    return hazard.groupby("true_altitude")["correct"].mean()


def main():
    noise_k, cal_z, cal_sigma = calibrate_noise_constant()
    print(f"Noise calibration: Z={cal_z:.4f} m, flatness_std={cal_sigma:.5f} m "
          f"-> K={noise_k:.2f} (sigma_Z = Z^2 / K)")

    df = build_dataset(noise_k, cal_z)
    print(f"Generated {len(df)} synthetic scenes across {len(ALTITUDES_SWEEP)} altitudes")
    print("Label balance:", dict(df["label"].value_counts()))
    none_altitude = df["est_altitude"].isna().sum()
    print(f"Scenes where altitude estimation failed (obstacle covered center patch): {none_altitude}")

    # Headline mechanism check: does obstacle_ratio on ground-truth-SAFE scenes
    # inflate with altitude purely from sensor noise crossing the fixed
    # OBSTACLE_THRESH, and does scaling the threshold with the noise model fix it?
    safe_df = df[df["label"] == "SAFE"]
    obs_fixed_by_alt = safe_df.groupby("true_altitude")["obs_a"].mean()
    obs_scaled_by_alt = safe_df.groupby("true_altitude")["obs_a_scaled"].mean()
    print("\nMean obstacle_ratio on ground-truth-SAFE scenes (should be ~constant if the "
          "threshold were properly calibrated):")
    print("  fixed OBSTACLE_THRESH=0.05m:", obs_fixed_by_alt.round(3).to_dict())
    print("  altitude-scaled threshold:  ", obs_scaled_by_alt.round(3).to_dict())

    strategies = {
        "adaptive": ["angle_a", "flat_a", "obs_a"],
        "fixed": ["angle_f", "flat_f", "obs_f"],
        "adaptive_scaled_thresh": ["angle_a", "flat_a", "obs_a_scaled"],
    }

    summary = {"obstacle_ratio_on_safe_scenes": {
        "fixed_threshold": {float(k): float(v) for k, v in obs_fixed_by_alt.items()},
        "scaled_threshold": {float(k): float(v) for k, v in obs_scaled_by_alt.items()},
    }}
    figures_data = {}
    hazard_recall_data = {}
    session_e = pd.read_csv(SESSION_E_CSV)

    for model_name, model_cls in [("GNB", GaussianNB), ("LDA", LinearDiscriminantAnalysis)]:
        models, results = {}, {}
        for strat_name, cols in strategies.items():
            model, result = fit_and_evaluate(df, cols, model_cls)
            models[strat_name] = model
            results[strat_name] = result

        summary[model_name] = {}
        for strat_name, result in results.items():
            acc = accuracy_by_altitude(result)
            recall = hazard_recall_by_altitude(result)
            summary[model_name][strat_name] = dict(
                overall_accuracy=float(result["correct"].mean()),
                by_altitude_accuracy={float(k): float(v) for k, v in acc.items()},
                by_altitude_hazard_recall={float(k): float(v) for k, v in recall.items()},
            )

        figures_data[model_name] = {s: accuracy_by_altitude(r) for s, r in results.items()}
        hazard_recall_data[model_name] = {s: hazard_recall_by_altitude(r) for s, r in results.items()}

        # Real-sensor check: Session E's real feature rows through the adaptive model
        Xe = session_e[["angle_deg", "flatness_std", "obstacle_ratio"]].to_numpy(dtype=float)
        pred_e = models["adaptive"].predict(Xe)
        real_sensor_safe_rate = float(pred_e.mean())
        summary[model_name]["real_sensor_session_e_predicted_safe_rate"] = real_sensor_safe_rate

        print(f"\n=== {model_name} ===")
        for strat_name, result in results.items():
            print(f"[{strat_name}] overall accuracy: {result['correct'].mean():.3f}")
        print(f"Session E (real sensor data, n={len(session_e)}, all ground-truth Landable) "
              f"predicted SAFE rate under adaptive-ROI-trained {model_name}: {real_sensor_safe_rate:.3f}")

    with open("simulation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Headline figure: mean obstacle_ratio on ground-truth-SAFE scenes, fixed
    # vs altitude-scaled threshold -- shows the noise-driven miscalibration
    # mechanism and that scaling the threshold with altitude fixes it.
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    ax0.plot(obs_fixed_by_alt.index, obs_fixed_by_alt.values, marker="o",
             color=ps.RED, label="Fixed OBSTACLE_THRESH = 0.05 m")
    ax0.plot(obs_scaled_by_alt.index, obs_scaled_by_alt.values, marker="s",
             color=ps.GREEN, label="Altitude-scaled threshold")
    ax0.set_xlabel("True altitude (m)")
    ax0.set_ylabel("Mean obstacle_ratio on SAFE scenes")
    ax0.set_title("Sensor-noise-driven threshold miscalibration")
    ax0.legend()
    ax0.grid(alpha=0.35)
    fig0.tight_layout()
    fig0.savefig("fig_sim_obstacle_threshold_fix.png")
    plt.close(fig0)

    # Figure: hazard recall vs altitude, one panel per model, all 3 strategies
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    markers = {"adaptive": "o", "fixed": "s", "adaptive_scaled_thresh": "^"}
    colors = {"adaptive": ps.BLUE, "fixed": ps.RED, "adaptive_scaled_thresh": ps.GREEN}
    labels = {"adaptive": "Adaptive ROI", "fixed": "Fixed ROI (assumes 2.5 m)",
              "adaptive_scaled_thresh": "Adaptive ROI + scaled threshold"}
    for ax, (model_name, strat_map) in zip(axes, hazard_recall_data.items()):
        for strat_name, recall in strat_map.items():
            ax.plot(recall.index, recall.values, marker=markers[strat_name],
                     color=colors[strat_name], label=labels[strat_name])
        ax.axvline(2.5, color="gray", linestyle=":", linewidth=1.5)
        ax.set_title(model_name)
        ax.set_xlabel("True altitude (m)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("Hazard recall (fraction of true\nNOT SAFE scenes correctly flagged)")
    axes[0].legend(loc="lower left", fontsize=10)
    fig.suptitle("Safety-critical metric: recall on ground-truth hazards vs. altitude")
    fig.tight_layout()
    fig.savefig("fig_sim_hazard_recall.png")
    plt.close(fig)

    # Feature-stability figure: flatness_std on ground-truth-SAFE scenes only,
    # adaptive vs fixed, across altitude -- isolates feature distortion from
    # classifier decision-boundary effects.
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    stats_a = safe_df.groupby("true_altitude")["flat_a"].agg(["mean", "std"])
    stats_f = safe_df.groupby("true_altitude")["flat_f"].agg(["mean", "std"])
    ax2.errorbar(stats_a.index, stats_a["mean"], yerr=stats_a["std"], marker="o",
                 color=ps.BLUE, capsize=4, label="Adaptive ROI")
    ax2.errorbar(stats_f.index, stats_f["mean"], yerr=stats_f["std"], marker="s",
                 color=ps.RED, capsize=4, label="Fixed ROI")
    ax2.set_xlabel("True altitude (m)")
    ax2.set_ylabel("flatness_std on SAFE scenes (m)")
    ax2.set_title("Feature stability across altitude")
    ax2.legend()
    ax2.grid(alpha=0.35)
    fig2.tight_layout()
    fig2.savefig("fig_sim_feature_stability.png")
    plt.close(fig2)

    df.to_csv("simulation_scenes.csv", index=False)
    print("\nWrote simulation_results.json, simulation_scenes.csv, "
          "fig_sim_obstacle_threshold_fix.png, fig_sim_hazard_recall.png, fig_sim_feature_stability.png")


if __name__ == "__main__":
    main()
