#!/usr/bin/env python3
"""
Evaluation pipeline for the landing-site classifiers: reproduces Checks 1-2
(split independence, session-grouped split, leave-one-session-out) and both
feature-subset ablations reported in the paper (the initial session-grouped
grid, and its full-LOSO repeat that the paper's Ablation section actually
draws its conclusion from), across a broad set of classical and modern
gradient-boosted-tree models.

Expects the eight labeled session CSVs (see data/README.md for the column
schema) in the working directory, named session_A.csv ... session_D.csv
(Phase 1, real outdoor flights), session_E.csv (Phase 1, the session not
flagged by the training-data-defect check), and session_flat.csv /
session_tilt.csv / session_clutter.csv (Phase 2, controlled).

Known limitation this script does NOT hide: four of five Phase-1 sessions
show a data-quality signature inconsistent with the fifth (see
sanity_check_session.py and the paper's Check 3 section). Every number
below is computed honestly from that data as it stands; it is not a
substitute for clean data.
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import plot_style as ps
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["angle_deg", "flatness_std", "obstacle_ratio"]

LABELED_FILES = [
    "session_A.csv",
    "session_B.csv",
    "session_C.csv",
    "session_D.csv",
    "session_E.csv",
    # Collected after the angle_deg pixel-pitch bug fix in features.py; see
    # data/README.md and sanity_check_session.py. Controlled setup, not an
    # outdoor flight.
    "session_flat.csv",
    "session_tilt.csv",
    "session_clutter.csv",
]
TEST_FILES = {"session_B.csv", "session_D.csv"}

RANDOM_STATE = 42

def make_models():
    return {
        "GNB":                 GaussianNB(),
        "SVM_RBF":             SVC(kernel="rbf", random_state=RANDOM_STATE),
        "LogisticRegression":  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "DecisionTree":        DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest":        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "ExtraTrees":          ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "kNN":                 KNeighborsClassifier(n_neighbors=5),
        "LDA":                 LinearDiscriminantAnalysis(),
        "QDA":                 QuadraticDiscriminantAnalysis(),
        "AdaBoost":            AdaBoostClassifier(random_state=RANDOM_STATE),
        "GradientBoosting":    GradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=RANDOM_STATE),
        "HistGradBoosting":    HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                              eval_metric="logloss", random_state=RANDOM_STATE),
    }


def load_all():
    frames = []
    for fname in LABELED_FILES:
        df = pd.read_csv(fname)
        missing = [c for c in FEATURES + ["label"] if c not in df.columns]
        if missing:
            raise ValueError(f"{fname} missing columns: {missing}")
        df = df[FEATURES + ["label"]].dropna()
        df["source_file"] = fname
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def safe_fit_predict(model, Xtr, ytr, Xte):
    """Some models (QDA in particular) can fail on near-singular covariance
    from the tightly-clustered sessions; report that honestly rather than
    crashing the whole benchmark."""
    try:
        model.fit(Xtr, ytr)
        return model.predict(Xte), None
    except Exception as e:
        return None, str(e)


def eval_one(model, Xtr, ytr, Xte, yte):
    pred, err = safe_fit_predict(model, Xtr, ytr, Xte)
    if pred is None:
        return dict(accuracy=None, precision=None, recall=None, f1=None,
                    confusion_matrix=None, error=err)
    acc = accuracy_score(yte, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    cm = confusion_matrix(yte, pred, labels=[0, 1]).tolist()
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, confusion_matrix=cm, error=None)


def main():
    df = load_all()
    print(f"[FULL MERGED DATASET] n={len(df)}  label counts: {dict(df['label'].value_counts().sort_index())}")
    for fname in LABELED_FILES:
        sub = df[df["source_file"] == fname]
        print(f"  {fname:20s} n={len(sub):4d}  label={sub['label'].unique().tolist()}")

    all_results = {}

    # ---- 1. Naive random split: Check 1 (leakage warning, GNB/SVM only) ----
    X = df[FEATURES].values
    y = df["label"].values
    Xtr_n, Xte_n, ytr_n, yte_n = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler_n = StandardScaler().fit(Xtr_n)
    naive = {}
    for name in ["GNB", "SVM_RBF"]:
        m = make_models()[name]
        naive[name] = eval_one(m, scaler_n.transform(Xtr_n), ytr_n, scaler_n.transform(Xte_n), yte_n)
    all_results["check1_naive_random_split_DO_NOT_USE"] = naive
    print("\n=== Check 1: naive random split (leakage warning, GNB/SVM only) ===")
    for name, r in naive.items():
        print(f"  {name:20s} acc={r['accuracy']:.3f}")

    # ---- 2. Session-grouped split across ALL models: Check 2, Part 1 ----
    print("\n=== Check 2, Part 1: session-grouped split, all models ===")
    train_df = df[~df["source_file"].isin(TEST_FILES)]
    test_df = df[df["source_file"].isin(TEST_FILES)]
    scaler_g = StandardScaler().fit(train_df[FEATURES].values)
    Xtr_g = scaler_g.transform(train_df[FEATURES].values)
    Xte_g = scaler_g.transform(test_df[FEATURES].values)
    ytr_g = train_df["label"].values
    yte_g = test_df["label"].values

    grouped = {}
    for name, model in make_models().items():
        r = eval_one(model, Xtr_g, ytr_g, Xte_g, yte_g)
        grouped[name] = r
        if r["accuracy"] is not None:
            print(f"  {name:20s} acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
                  f"rec={r['recall']:.3f}  f1={r['f1']:.3f}")
        else:
            print(f"  {name:20s} FAILED: {r['error']}")
    all_results["check2_part1_session_grouped_split"] = grouped

    # ---- 3. Leave-one-session-out, all models: Check 2, Part 2 ----
    print("\n=== Check 2, Part 2: leave-one-session-out, all models ===")
    loso = {}
    for name, _ in make_models().items():
        loso[name] = {}
    for holdout in LABELED_FILES:
        tr = df[df["source_file"] != holdout]
        te = df[df["source_file"] == holdout]
        scaler = StandardScaler().fit(tr[FEATURES].values)
        Xtr = scaler.transform(tr[FEATURES].values)
        Xte = scaler.transform(te[FEATURES].values)
        ytr = tr["label"].values
        yte = te["label"].values
        true_label = int(te["label"].iloc[0])
        for name, model in make_models().items():
            pred, err = safe_fit_predict(model, Xtr, ytr, Xte)
            if pred is None:
                loso[name][holdout] = dict(recall=None, n=len(te), true_label=true_label, error=err)
            else:
                recall = float((pred == yte).mean())  # single-class fold: this IS recall for that class
                loso[name][holdout] = dict(recall=recall, n=len(te), true_label=true_label, error=None)
    all_results["check2_part2_leave_one_session_out"] = loso
    for name in loso:
        row = "  ".join(f"{loso[name][f]['recall']:.2f}" if loso[name][f]['recall'] is not None else " N/A"
                         for f in LABELED_FILES)
        print(f"  {name:20s} {row}")

    # ---- 4. Full feature-subset ablation across a representative model set ----
    # Every non-empty subset of the 3 features x a fixed model set spanning
    # classical (kNN, LDA), classical-ensemble (RandomForest), and modern
    # gradient-boosted-tree (XGBoost) families.
    ablation_models = ["kNN", "LDA", "RandomForest", "XGBoost"]
    feature_subsets = {
        "angle": ["angle_deg"],
        "flatness": ["flatness_std"],
        "obstacle": ["obstacle_ratio"],
        "angle+flatness": ["angle_deg", "flatness_std"],
        "angle+obstacle": ["angle_deg", "obstacle_ratio"],
        "flatness+obstacle": ["flatness_std", "obstacle_ratio"],
        "all three": FEATURES,
    }
    print(f"\n=== Feature-subset ablation, {len(ablation_models)} models x "
          f"{len(feature_subsets)} subsets, session-grouped split ===")
    ablation = {}
    for model_name in ablation_models:
        for subset_name, cols in feature_subsets.items():
            scaler = StandardScaler().fit(train_df[cols].values)
            Xtr_c = scaler.transform(train_df[cols].values)
            Xte_c = scaler.transform(test_df[cols].values)
            model = make_models()[model_name]
            r = eval_one(model, Xtr_c, ytr_g, Xte_c, yte_g)
            ablation[f"{model_name}::{subset_name}"] = r
            print(f"  {model_name:14s} {subset_name:20s} acc={r['accuracy']:.3f}  "
                  f"prec={r['precision']:.3f}  rec={r['recall']:.3f}  f1={r['f1']:.3f}")
    all_results["ablation_session_grouped_full_grid"] = dict(
        models=ablation_models, subsets=list(feature_subsets.keys()), results=ablation)

    # ---- 5. Same ablation grid repeated under full LOSO ----
    # The session-grouped ablation above is checked against only one held-out
    # pair, which lacks the hazard types introduced by Check 2, Part 2. This
    # repeats it under full LOSO (same 4 models x 8 sessions = 32 folds per
    # feature subset) to see whether the session-grouped conclusion (dropping
    # obstacle_ratio costs nothing) survives a held-out session it was never
    # checked against.
    print(f"\n=== Feature-subset ablation under full LOSO, {len(ablation_models)} models x "
          f"8 sessions x {len(feature_subsets)} subsets ===")
    ablation_loso = {}
    clutter_file = "session_clutter.csv"
    for subset_name, cols in feature_subsets.items():
        fold_recalls = []
        clutter_recalls = []
        for holdout in LABELED_FILES:
            tr = df[df["source_file"] != holdout]
            te = df[df["source_file"] == holdout]
            scaler = StandardScaler().fit(tr[cols].values)
            Xtr = scaler.transform(tr[cols].values)
            Xte = scaler.transform(te[cols].values)
            ytr = tr["label"].values
            yte = te["label"].values
            for model_name in ablation_models:
                model = make_models()[model_name]
                pred, err = safe_fit_predict(model, Xtr, ytr, Xte)
                recall = float((pred == yte).mean()) if pred is not None else 0.0
                fold_recalls.append(recall)
                if holdout == clutter_file:
                    clutter_recalls.append(recall)
        ablation_loso[subset_name] = dict(
            n_folds=len(fold_recalls),
            mean_recall=float(np.mean(fold_recalls)),
            clutter_mean_recall=float(np.mean(clutter_recalls)),
        )
        print(f"  {subset_name:20s} n={len(fold_recalls):3d}  "
              f"mean_recall={np.mean(fold_recalls):.3f}  "
              f"clutter_mean_recall={np.mean(clutter_recalls):.3f}")
    all_results["ablation_loso_full_grid"] = dict(
        models=ablation_models, subsets=list(feature_subsets.keys()), results=ablation_loso)

    valid_grouped = {k: v for k, v in grouped.items() if v["accuracy"] is not None}
    best_model_name = max(valid_grouped, key=lambda k: valid_grouped[k]["accuracy"])

    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nWrote evaluation_results.json")

    make_figures(df, grouped, loso, ablation, ablation_models, feature_subsets, best_model_name)
    print("Wrote fig_session_grouped_accuracy.png, fig_loso_recall.png, "
          "fig_feature_distributions.png, fig_ablation.png")


def make_figures(df, grouped, loso, ablation, ablation_models, feature_subsets, best_model_name):
    # Fig: session-grouped accuracy per model
    names = list(grouped.keys())
    accs = [grouped[n]["accuracy"] if grouped[n]["accuracy"] is not None else 0 for n in names]
    failed = [grouped[n]["accuracy"] is None for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [ps.GRAY if f else ps.BLUE for f in failed]
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=1.1)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Session-grouped accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Session-grouped test accuracy (gray = model failed to fit)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("fig_session_grouped_accuracy.png")
    plt.close(fig)

    # Fig: LOSO recall heatmap (models x sessions)
    sessions = list(next(iter(loso.values())).keys())
    matrix = np.full((len(loso), len(sessions)), np.nan)
    for i, name in enumerate(loso):
        for j, s in enumerate(sessions):
            r = loso[name][s]["recall"]
            if r is not None:
                matrix[i, j] = r
    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([s.replace("session_", "").replace(".csv", "") for s in sessions],
                        rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(loso)))
    ax.set_yticklabels(list(loso.keys()), fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                txt_color = "white" if matrix[i, j] < 0.35 or matrix[i, j] > 0.85 else "black"
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=txt_color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")
    cbar = fig.colorbar(im)
    cbar.set_label("Recall on held-out session's true class", fontweight="bold")
    ax.set_title("Leave-one-session-out recall, all models")
    plt.tight_layout()
    plt.savefig("fig_loso_recall.png")
    plt.close(fig)

    # Fig: feature distributions per session (the data-quality diagnostic, visualized)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, feat in zip(axes, ["flatness_std", "obstacle_ratio", "angle_deg"]):
        data = [df[df["source_file"] == f][feat].values for f in LABELED_FILES]
        labels = [f.replace("session_", "").replace(".csv", "") for f in LABELED_FILES]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=2))
        for patch, c in zip(bp["boxes"], ps.MODEL_PALETTE):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_title(feat, fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.suptitle("Feature distributions by session (four of five show the same anomalous scale)")
    plt.tight_layout()
    plt.savefig("fig_feature_distributions.png")
    plt.close(fig)

    # Fig: ablation grid, one bar-group per feature subset, grouped by model
    subset_names = list(feature_subsets.keys())
    x = np.arange(len(subset_names))
    width = 0.8 / len(ablation_models)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, model_name in enumerate(ablation_models):
        accs = [ablation[f"{model_name}::{s}"]["accuracy"] or 0 for s in subset_names]
        ax.bar(x + i * width, accs, width, label=model_name,
               color=ps.MODEL_PALETTE[i], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(ablation_models) - 1) / 2)
    ax.set_xticklabels(subset_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Feature-subset ablation, session-grouped split")
    ax.legend(ncol=2, loc="lower right", framealpha=0.9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("fig_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
