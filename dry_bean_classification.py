import os
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
from zipfile import ZipFile

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR    = "resultados_rna"
DATASET_URL   = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DATASET_XLSX  = "Dry_Bean_Dataset.xlsx"
RANDOM_SEEDS  = [0, 7, 21, 42, 99]
MAX_ITER      = 500
LOSS_THRESHOLD = 0.001

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  [SAVED] {path}")


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_XLSX):
        print("[DATA] Downloading Dry Bean Dataset...")
        zip_path = "dry_bean.zip"
        urlretrieve(DATASET_URL, zip_path)
        with ZipFile(zip_path, "r") as z:
            z.extractall(".")
        print("[DATA] Download complete.")
    else:
        print("[DATA] Dataset file found locally.")

    # Try both common filenames
    for fname in [DATASET_XLSX, "DryBeanDataset/Dry_Bean_Dataset.xlsx"]:
        if os.path.exists(fname):
            df = pd.read_excel(fname)
            print(f"[DATA] Loaded {len(df):,} rows × {df.shape[1]} cols from '{fname}'")
            return df

    raise FileNotFoundError(
        "Could not find Dry_Bean_Dataset.xlsx. "
        "Please download manually from https://archive.ics.uci.edu/dataset/602/dry+bean+dataset "
        "and place the .xlsx file in the same directory as this script."
    )


def preprocess(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values.astype(np.float64)
    y_raw = df["Class"].values

    # Z-Score normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Integer label encoding (CrossEntropy in sklearn expects 1-D integer labels)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Stratified split: 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_scaled, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    print(
        f"[DATA] Split — Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}"
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te, le


# ---------------------------------------------------------------------------
# 2. STAGE 1 — Weight Initialisation & Activation Sensitivity
# ---------------------------------------------------------------------------
def stage1(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "=" * 60)
    print("STAGE 1 — Weight Init / Activation (5 seeds, 50 neurons)")
    print("=" * 60)

    records   = []
    loss_curves = {}

    for idx, seed in enumerate(RANDOM_SEEDS, 1):
        print(f"  Run {idx}/{len(RANDOM_SEEDS)} | seed={seed} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            clf = MLPClassifier(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="sgd",
                learning_rate_init=0.01,
                momentum=0.9,
                max_iter=MAX_ITER,
                random_state=seed,
                warm_start=False,
                early_stopping=False,
                verbose=False,
                n_iter_no_change=MAX_ITER,  # disable sklearn's own early stop
            )
            clf.fit(X_tr, y_tr)
            duration = time.time() - t0

            train_loss  = clf.loss_curve_
            val_acc     = accuracy_score(y_val, clf.predict(X_val))
            test_acc    = accuracy_score(y_te,  clf.predict(X_te))
            final_loss  = train_loss[-1]
            epochs_ran  = len(train_loss)

            loss_curves[seed] = train_loss
            records.append({
                "seed":        seed,
                "epochs":      epochs_ran,
                "final_loss":  round(final_loss, 6),
                "val_accuracy":  round(val_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "time_s":      round(duration, 2),
            })
            print(f"loss={final_loss:.4f}  val_acc={val_acc:.4f}  [{elapsed(t0)}]")

        except Exception as exc:
            print(f"ERROR — {exc}")

    # Save table
    df_res = pd.DataFrame(records)
    save_csv(df_res, "stage1_seed_comparison.csv")
    print(df_res.to_string(index=False))

    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, curve in loss_curves.items():
        ax.plot(curve, label=f"seed={seed}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Stage 1 — Convergence Curves (5 random seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "stage1_convergence_curves.png")


# ---------------------------------------------------------------------------
# 3. STAGE 2 — Grid Search (LR × Momentum)
# ---------------------------------------------------------------------------
def stage2(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "=" * 60)
    print("STAGE 2 — Grid Search: LR × Momentum")
    print("=" * 60)

    learning_rates = [0.01, 0.1, 0.5]
    momenta        = [0.5, 0.7, 0.9]
    combos         = list(itertools.product(learning_rates, momenta))
    records        = []

    for ci, (lr, mo) in enumerate(combos, 1):
        print(f"  Combo {ci:>2}/{len(combos)} | lr={lr}  momentum={mo} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            clf = MLPClassifier(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="sgd",
                learning_rate_init=lr,
                momentum=mo,
                max_iter=MAX_ITER,
                tol=LOSS_THRESHOLD,          # stop when loss improvement < tol
                n_iter_no_change=10,
                random_state=42,
                early_stopping=False,
                verbose=False,
            )
            clf.fit(X_tr, y_tr)
            duration   = time.time() - t0
            final_loss = clf.loss_curve_[-1]
            epochs_ran = len(clf.loss_curve_)
            val_acc    = accuracy_score(y_val, clf.predict(X_val))
            converged  = final_loss <= LOSS_THRESHOLD

            records.append({
                "lr":           lr,
                "momentum":     mo,
                "epochs":       epochs_ran,
                "final_loss":   round(final_loss, 6),
                "val_accuracy": round(val_acc, 4),
                "converged":    converged,
                "time_s":       round(duration, 2),
            })
            print(
                f"loss={final_loss:.5f}  epochs={epochs_ran}  "
                f"val_acc={val_acc:.4f}  conv={converged}  [{elapsed(t0)}]"
            )

        except Exception as exc:
            print(f"ERROR — {exc}")
            records.append({
                "lr": lr, "momentum": mo,
                "epochs": None, "final_loss": None,
                "val_accuracy": None, "converged": False,
                "time_s": round(time.time() - t0, 2),
            })

        # Incremental save after every run
        save_csv(pd.DataFrame(records), "stage2_grid_search.csv")

    df_res = pd.DataFrame(records)
    print("\n--- Grid Search Summary ---")
    print(df_res.to_string(index=False))

    # Heatmap — val accuracy
    try:
        pivot = df_res.pivot(index="momentum", columns="lr", values="val_accuracy")
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Momentum")
        ax.set_title("Stage 2 — Val Accuracy Heatmap (LR × Momentum)")
        plt.colorbar(im, ax=ax, label="Val Accuracy")
        for (r, c), val in np.ndenumerate(pivot.values):
            if val is not None and not np.isnan(float(val if val else 0)):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=9)
        save_fig(fig, "stage2_accuracy_heatmap.png")
    except Exception as exc:
        print(f"  [WARN] Heatmap skipped: {exc}")


# ---------------------------------------------------------------------------
# 4. STAGE 3 — Topology Search
# ---------------------------------------------------------------------------
def build_hidden_layers(n_layers: int, n_neurons: int) -> tuple:
    return tuple([n_neurons] * n_layers)


def stage3(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "=" * 60)
    print("STAGE 3 — Topology Search (layers × neurons)")
    print("=" * 60)

    layer_counts  = [1, 2, 3]
    neuron_counts = [10, 30, 50, 70, 100]
    combos        = list(itertools.product(layer_counts, neuron_counts))
    records       = []

    for ci, (n_layers, n_neurons) in enumerate(combos, 1):
        topology = build_hidden_layers(n_layers, n_neurons)
        label    = "×".join(str(n) for n in topology)
        print(
            f"  Combo {ci:>2}/{len(combos)} | topology=({label}) ...",
            end=" ", flush=True
        )
        t0 = time.time()
        try:
            clf = MLPClassifier(
                hidden_layer_sizes=topology,
                activation="relu",
                solver="sgd",
                learning_rate_init=0.01,
                momentum=0.9,
                max_iter=MAX_ITER,
                tol=LOSS_THRESHOLD,
                n_iter_no_change=15,
                validation_fraction=0.0,   # we handle validation manually
                random_state=42,
                early_stopping=False,
                verbose=False,
            )
            clf.fit(X_tr, y_tr)
            duration = time.time() - t0

            train_loss   = clf.loss_curve_[-1]
            val_preds    = clf.predict(X_val)
            val_proba    = clf.predict_proba(X_val)
            val_acc      = accuracy_score(y_val, val_preds)
            val_loss     = log_loss(y_val, val_proba)
            val_f1_macro = f1_score(y_val, val_preds, average="macro")
            val_f1_w     = f1_score(y_val, val_preds, average="weighted")
            epochs_ran   = len(clf.loss_curve_)

            records.append({
                "topology":       label,
                "n_layers":       n_layers,
                "n_neurons":      n_neurons,
                "total_params":   sum(
                    w.size for w in clf.coefs_
                ) + sum(b.size for b in clf.intercepts_),
                "epochs":         epochs_ran,
                "train_loss":     round(train_loss, 6),
                "val_loss":       round(val_loss, 6),
                "val_accuracy":   round(val_acc, 4),
                "val_f1_macro":   round(val_f1_macro, 4),
                "val_f1_weighted":round(val_f1_w, 4),
                "time_s":         round(duration, 2),
            })
            print(
                f"train_loss={train_loss:.5f}  val_f1={val_f1_macro:.4f}  "
                f"val_acc={val_acc:.4f}  [{elapsed(t0)}]"
            )

        except Exception as exc:
            print(f"ERROR — {exc}")
            records.append({
                "topology": label, "n_layers": n_layers, "n_neurons": n_neurons,
                "total_params": None, "epochs": None,
                "train_loss": None, "val_loss": None,
                "val_accuracy": None, "val_f1_macro": None,
                "val_f1_weighted": None,
                "time_s": round(time.time() - t0, 2),
            })

        # Incremental save
        save_csv(pd.DataFrame(records), "stage3_topology_search.csv")

    df_res = pd.DataFrame(records).dropna(subset=["val_f1_macro"])

    # Top 4
    top4 = df_res.nlargest(4, "val_f1_macro")
    print("\n--- TOP 4 TOPOLOGIES (by Val F1 Macro) ---")
    print(top4[["topology", "epochs", "train_loss", "val_loss",
                "val_accuracy", "val_f1_macro", "val_f1_weighted", "time_s"]]
          .to_string(index=False))
    save_csv(top4, "stage3_top4_topologies.csv")

    # Bar chart — val F1 macro
    try:
        df_plot = df_res.sort_values("val_f1_macro", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#2ecc71" if t in top4["topology"].values else "#3498db"
                  for t in df_plot["topology"]]
        ax.bar(df_plot["topology"], df_plot["val_f1_macro"], color=colors)
        ax.set_xlabel("Topology (neurons per layer)")
        ax.set_ylabel("Val F1 Macro")
        ax.set_title("Stage 3 — Topology vs Val F1 Macro (green = Top 4)")
        ax.set_ylim(df_plot["val_f1_macro"].min() - 0.02, 1.0)
        plt.xticks(rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
        save_fig(fig, "stage3_topology_f1_bar.png")
    except Exception as exc:
        print(f"  [WARN] Bar chart skipped: {exc}")

    # Heatmap — F1 across layers × neurons
    try:
        pivot = df_res.pivot(index="n_layers", columns="n_neurons", values="val_f1_macro")
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("Neurons per Layer")
        ax.set_ylabel("Number of Hidden Layers")
        ax.set_title("Stage 3 — F1 Macro Heatmap (Layers × Neurons)")
        plt.colorbar(im, ax=ax, label="F1 Macro")
        for (r, c), val in np.ndenumerate(pivot.values):
            if not np.isnan(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=8)
        save_fig(fig, "stage3_f1_heatmap.png")
    except Exception as exc:
        print(f"  [WARN] Heatmap skipped: {exc}")

    # Final evaluation of best model on test set
    best_row = top4.iloc[0]
    best_topo = build_hidden_layers(int(best_row["n_layers"]), int(best_row["n_neurons"]))
    print(f"\n[FINAL] Evaluating best topology {best_row['topology']} on TEST set...")
    clf_best = MLPClassifier(
        hidden_layer_sizes=best_topo,
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        momentum=0.9,
        max_iter=MAX_ITER,
        tol=LOSS_THRESHOLD,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    clf_best.fit(X_tr, y_tr)
    te_preds = clf_best.predict(X_te)
    print(classification_report(y_te, te_preds, digits=4))
    report_df = pd.DataFrame(
        classification_report(y_te, te_preds, digits=4, output_dict=True)
    ).T
    save_csv(report_df, "stage3_best_model_test_report.csv")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    total_start = time.time()
    print("=" * 60)
    print("  DRY BEAN MLP PIPELINE — START")
    print("=" * 60)

    df = load_dataset()
    X_tr, X_val, X_te, y_tr, y_val, y_te, le = preprocess(df)

    stage1(X_tr, X_val, X_te, y_tr, y_val, y_te)
    stage2(X_tr, X_val, X_te, y_tr, y_val, y_te)
    stage3(X_tr, X_val, X_te, y_tr, y_val, y_te)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE — total time: {elapsed(total_start)}")
    print(f"  All results saved in '{OUTPUT_DIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Dry Bean Dataset — MLP Pipeline (Stages 4–7)
"""

warnings.filterwarnings("ignore")

OUTPUT_DIR     = "resultados_rna"
MAX_ITER       = 500
LOSS_THRESHOLD = 0.001
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAMPION = dict(
    hidden_layer_sizes=(50, 50),
    activation="relu",
    solver="sgd",
    learning_rate_init=0.1,
    momentum=0.7,
    max_iter=MAX_ITER,
    tol=LOSS_THRESHOLD,
    n_iter_no_change=15,
    random_state=42,
    verbose=False,
)

TOP4_TOPOLOGIES = [
    (50, 50),
    (70, 70),
    (30, 30),
    (100, 100, 100),
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  [SAVED] {path}")


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")


def elapsed(t0):
    return f"{time.time() - t0:.2f}s"


def make_clf(**overrides):
    params = {**CHAMPION, **overrides}
    return MLPClassifier(**params)


def eval_metrics(clf, X, y, label=""):
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    return {
        f"{label}loss":       round(log_loss(y, proba), 6),
        f"{label}accuracy":   round(accuracy_score(y, preds), 4),
        f"{label}f1_macro":   round(f1_score(y, preds, average="macro"), 4),
        f"{label}f1_weighted":round(f1_score(y, preds, average="weighted"), 4),
    }


# ---------------------------------------------------------------------------
# STANDALONE DATA LOADER (only used when running this file directly)
# ---------------------------------------------------------------------------
def _load_data_standalone():
    DATASET_XLSX = "Dry_Bean_Dataset.xlsx"
    for fname in [DATASET_XLSX, "DryBeanDataset/Dry_Bean_Dataset.xlsx"]:
        if os.path.exists(fname):
            df = pd.read_excel(fname)
            break
    else:
        raise FileNotFoundError(
            "Dataset not found. Run the main pipeline first, or place "
            "Dry_Bean_Dataset.xlsx in the current directory."
        )

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values.astype(np.float64)
    y = LabelEncoder().fit_transform(df["Class"].values)
    X = StandardScaler().fit_transform(X)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )
    print(f"[DATA] Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ---------------------------------------------------------------------------
# STAGE 4 — Training Data Size Influence
# ---------------------------------------------------------------------------
def stage4(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "=" * 60)
    print("STAGE 4 — Training Data Size Influence")
    print("=" * 60)

    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    records   = []

    for fi, frac in enumerate(fractions, 1):
        print(f"  Fraction {fi}/{len(fractions)} | frac={frac:.0%} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            if frac < 1.0:
                X_sub, _, y_sub, _ = train_test_split(
                    X_tr, y_tr,
                    train_size=frac,
                    stratify=y_tr,
                    random_state=42,
                )
            else:
                X_sub, y_sub = X_tr, y_tr

            clf = make_clf()
            clf.fit(X_sub, y_sub)
            duration = time.time() - t0

            m = eval_metrics(clf, X_val, y_val, label="val_")
            records.append({
                "fraction":      frac,
                "n_samples":     len(X_sub),
                "epochs":        len(clf.loss_curve_),
                "train_loss":    round(clf.loss_curve_[-1], 6),
                **m,
                "time_s":        round(duration, 2),
            })
            print(
                f"n={len(X_sub):,}  val_loss={m['val_loss']:.5f}  "
                f"val_acc={m['val_accuracy']:.4f}  val_f1={m['val_f1_macro']:.4f}  [{elapsed(t0)}]"
            )

        except Exception as exc:
            print(f"ERROR — {exc}")
            records.append({"fraction": frac, "n_samples": None, "error": str(exc)})

        save_csv(pd.DataFrame(records), "stage4_data_influence.csv")

    df_res = pd.DataFrame(records)

    # Line plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("val_loss", "Val Loss"), ("val_accuracy", "Val Accuracy"), ("val_f1_macro", "Val F1 Macro")]
    colors  = ["#e74c3c", "#2ecc71", "#3498db"]
    for ax, (col, title), color in zip(axes, metrics, colors):
        ax.plot(df_res["fraction"] * 100, df_res[col], marker="o", color=color, linewidth=2)
        ax.set_xlabel("Training Set Size (%)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Stage 4 — Metrics vs Training Data Size", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "stage4_data_influence.png")

    print("\n--- Stage 4 Summary ---")
    print(df_res[["fraction", "n_samples", "val_loss", "val_accuracy", "val_f1_macro", "time_s"]]
          .to_string(index=False))


# ---------------------------------------------------------------------------
# STAGE 5 — Feature Influence (SelectKBest)
# ---------------------------------------------------------------------------
def stage5(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "=" * 60)
    print("STAGE 5 — Feature Influence (16 vs 8 features)")
    print("=" * 60)

    records = []

    configs = [
        ("All 16 features", X_tr, X_val, None),
    ]

    # Build 8-feature subset using SelectKBest on training data only
    try:
        selector = SelectKBest(f_classif, k=8)
        X_tr_8   = selector.fit_transform(X_tr, y_tr)
        X_val_8  = selector.transform(X_val)
        selected_mask   = selector.get_support()
        selected_indices = np.where(selected_mask)[0]
        print(f"  [INFO] Selected feature indices: {selected_indices.tolist()}")
        configs.append(("Top 8 features (SelectKBest)", X_tr_8, X_val_8, selector))
    except Exception as exc:
        print(f"  [WARN] SelectKBest failed: {exc}")

    for label, X_train_use, X_val_use, sel in configs:
        n_features = X_train_use.shape[1]
        print(f"  Training with '{label}' (n_features={n_features}) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            clf = make_clf()
            clf.fit(X_train_use, y_tr)
            duration = time.time() - t0

            m_tr  = eval_metrics(clf, X_train_use, y_tr,  label="train_")
            m_val = eval_metrics(clf, X_val_use,   y_val, label="val_")

            records.append({
                "config":        label,
                "n_features":    n_features,
                "epochs":        len(clf.loss_curve_),
                **m_tr,
                **m_val,
                "time_s":        round(duration, 2),
            })
            print(
                f"train_acc={m_tr['train_accuracy']:.4f}  "
                f"val_acc={m_val['val_accuracy']:.4f}  "
                f"val_f1={m_val['val_f1_macro']:.4f}  [{elapsed(t0)}]"
            )
        except Exception as exc:
            print(f"ERROR — {exc}")

    df_res = pd.DataFrame(records)
    save_csv(df_res, "stage5_feature_influence.csv")

    # Bar chart comparison
    try:
        metrics_to_plot = ["val_accuracy", "val_f1_macro", "val_f1_weighted"]
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (_, row) in enumerate(df_res.iterrows()):
            vals = [row[m] for m in metrics_to_plot]
            ax.bar(x + i * width, vals, width, label=row["config"])
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(["Val Accuracy", "Val F1 Macro", "Val F1 Weighted"])
        ax.set_ylim(0.85, 1.0)
        ax.set_title("Stage 5 — 16 Features vs 8 Features (Validation)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        save_fig(fig, "stage5_feature_influence.png")
    except Exception as exc:
        print(f"  [WARN] Chart skipped: {exc}")

    print("\n--- Stage 5 Summary ---")
    cols = ["config", "n_features", "train_accuracy", "val_accuracy",
            "val_f1_macro", "val_f1_weighted", "time_s"]
    print(df_res[[c for c in cols if c in df_res.columns]].to_string(index=False))


# ---------------------------------------------------------------------------
# STAGE 6 — Test Set Validation of Top 4 Topologies
# ---------------------------------------------------------------------------
def stage6(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "=" * 60)
    print("STAGE 6 — Test Set Validation (Top 4 Topologies)")
    print("=" * 60)

    # Combine train + val for final training before test evaluation
    X_trainval = np.vstack([X_tr, X_val])
    y_trainval = np.concatenate([y_tr, y_val])

    records = []

    for ti, topo in enumerate(TOP4_TOPOLOGIES, 1):
        label = "×".join(str(n) for n in topo)
        print(f"  Topology {ti}/{len(TOP4_TOPOLOGIES)} | ({label}) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            clf = make_clf(hidden_layer_sizes=topo)
            clf.fit(X_trainval, y_trainval)
            duration = time.time() - t0

            m_tr = eval_metrics(clf, X_trainval, y_trainval, label="train_")
            m_te = eval_metrics(clf, X_te, y_te, label="test_")

            records.append({
                "topology":       label,
                "n_layers":       len(topo),
                "epochs":         len(clf.loss_curve_),
                **m_tr,
                **m_te,
                "time_s":         round(duration, 2),
            })
            print(
                f"train_acc={m_tr['train_accuracy']:.4f}  "
                f"test_acc={m_te['test_accuracy']:.4f}  "
                f"test_f1={m_te['test_f1_macro']:.4f}  [{elapsed(t0)}]"
            )

        except Exception as exc:
            print(f"ERROR — {exc}")

        save_csv(pd.DataFrame(records), "stage6_test_validation.csv")

    df_res = pd.DataFrame(records)
    best   = df_res.loc[df_res["test_f1_macro"].idxmax()]

    print("\n--- Stage 6 Summary (Train vs Test) ---")
    cols = ["topology", "epochs", "train_accuracy", "test_accuracy",
            "train_f1_macro", "test_f1_macro", "time_s"]
    print(df_res[[c for c in cols if c in df_res.columns]].to_string(index=False))
    print(f"\n  >>> CHAMPION: topology=({best['topology']})  "
          f"test_f1={best['test_f1_macro']:.4f}  test_acc={best['test_accuracy']:.4f}")

    # Grouped bar — train vs test accuracy & F1
    try:
        x     = np.arange(len(df_res))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 1.5 * width, df_res["train_accuracy"],   width, label="Train Acc",   color="#3498db")
        ax.bar(x - 0.5 * width, df_res["test_accuracy"],    width, label="Test Acc",    color="#2980b9")
        ax.bar(x + 0.5 * width, df_res["train_f1_macro"],   width, label="Train F1",    color="#e74c3c")
        ax.bar(x + 1.5 * width, df_res["test_f1_macro"],    width, label="Test F1",     color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(df_res["topology"])
        ax.set_ylim(0.88, 1.0)
        ax.set_xlabel("Topology")
        ax.set_title("Stage 6 — Train vs Test Performance (Top 4 Topologies)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        save_fig(fig, "stage6_train_vs_test.png")
    except Exception as exc:
        print(f"  [WARN] Chart skipped: {exc}")

    return tuple(int(n) for n in best["topology"].split("×"))


# ---------------------------------------------------------------------------
# STAGE 7 — Stratified K-Fold Cross Validation
# ---------------------------------------------------------------------------
def stage7(X_tr, X_val, X_te, y_tr, y_val, y_te, best_topology):
    print("\n" + "=" * 60)
    print(f"STAGE 7 — Stratified K-Fold (K=5) | topology={best_topology}")
    print("=" * 60)

    # Use train + val union for cross-validation
    X_cv = np.vstack([X_tr, X_val])
    y_cv = np.concatenate([y_tr, y_val])

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), 1):
        print(f"  Fold {fold}/5 ...", end=" ", flush=True)
        t0 = time.time()
        try:
            X_fold_tr, X_fold_val = X_cv[train_idx], X_cv[val_idx]
            y_fold_tr, y_fold_val = y_cv[train_idx], y_cv[val_idx]

            clf = make_clf(hidden_layer_sizes=best_topology)
            clf.fit(X_fold_tr, y_fold_tr)
            duration = time.time() - t0

            m = eval_metrics(clf, X_fold_val, y_fold_val, label="")
            records.append({
                "fold":       fold,
                "n_train":    len(X_fold_tr),
                "n_val":      len(X_fold_val),
                "epochs":     len(clf.loss_curve_),
                "train_loss": round(clf.loss_curve_[-1], 6),
                **m,
                "time_s":     round(duration, 2),
            })
            print(
                f"loss={m['loss']:.5f}  acc={m['accuracy']:.4f}  "
                f"f1={m['f1_macro']:.4f}  [{elapsed(t0)}]"
            )
        except Exception as exc:
            print(f"ERROR — {exc}")
            records.append({"fold": fold, "error": str(exc)})

        save_csv(pd.DataFrame(records), "stage7_kfold_results.csv")

    df_folds = pd.DataFrame(records)
    numeric_cols = ["loss", "accuracy", "f1_macro", "f1_weighted", "time_s"]
    numeric_cols = [c for c in numeric_cols if c in df_folds.columns]

    mean_row = {c: round(df_folds[c].mean(), 6) for c in numeric_cols}
    std_row  = {c: round(df_folds[c].std(),  6) for c in numeric_cols}
    mean_row["fold"] = "MEAN"
    std_row["fold"]  = "STD"

    df_summary = pd.concat(
        [df_folds, pd.DataFrame([mean_row, std_row])],
        ignore_index=True
    )
    save_csv(df_summary, "stage7_kfold_results.csv")

    print("\n--- Stage 7 K-Fold Summary ---")
    print(df_summary[["fold"] + numeric_cols].to_string(index=False))

    mean = df_folds["f1_macro"].mean()
    std  = df_folds["f1_macro"].std()
    print(f"\n  >>> F1 Macro — Mean: {mean:.4f}  Std: {std:.4f}")

    # Bar chart — F1 per fold
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#3498db"] * 5
        colors[df_folds["f1_macro"].idxmax()] = "#2ecc71"  # highlight best
        ax.bar([f"Fold {i}" for i in df_folds["fold"]], df_folds["f1_macro"], color=colors)
        ax.axhline(mean, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.4f}")
        ax.fill_between(range(5), mean - std, mean + std, alpha=0.15, color="#e74c3c",
                        label=f"±1 Std = {std:.4f}")
        ax.set_ylim(mean - 4 * std if std > 0 else mean - 0.05, 1.005)
        ax.set_xlabel("Fold")
        ax.set_ylabel("F1 Macro")
        ax.set_title(f"Stage 7 — K-Fold F1 Macro per Fold | topology={best_topology}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        save_fig(fig, "stage7_kfold_f1_bars.png")
    except Exception as exc:
        print(f"  [WARN] Chart skipped: {exc}")

    # Convergence curves per fold (loss at final epoch only — bar)
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([f"Fold {i}" for i in df_folds["fold"]], df_folds["train_loss"], color="#9b59b6")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Final Training Loss")
        ax.set_title("Stage 7 — Final Training Loss per Fold")
        ax.grid(axis="y", alpha=0.3)
        save_fig(fig, "stage7_kfold_loss_bars.png")
    except Exception as exc:
        print(f"  [WARN] Loss chart skipped: {exc}")


# ---------------------------------------------------------------------------
# MAIN (stages 4–7)
# ---------------------------------------------------------------------------
def main_stages_4to7(X_tr=None, X_val=None, X_te=None,
                     y_tr=None, y_val=None, y_te=None):

    # If called standalone (not appended to main pipeline), load data
    if X_tr is None:
        X_tr, X_val, X_te, y_tr, y_val, y_te = _load_data_standalone()

    total_start = time.time()
    print("\n" + "=" * 60)
    print("  STAGES 4–7 — START")
    print("=" * 60)

    stage4(X_tr, X_val, X_te, y_tr, y_val, y_te)
    stage5(X_tr, X_val, X_te, y_tr, y_val, y_te)
    best_topology = stage6(X_tr, X_val, X_te, y_tr, y_val, y_te)
    stage7(X_tr, X_val, X_te, y_tr, y_val, y_te, best_topology)

    print("\n" + "=" * 60)
    print(f"  STAGES 4–7 COMPLETE — total time: {elapsed(total_start)}")
    print(f"  All results saved in '{OUTPUT_DIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main_stages_4to7()