from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import feature_extractor
from feature_extractor import extract_features

print(f"Using feature_extractor module from: {feature_extractor.__file__}")

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = MODEL_DIR / "gbm_model_cv.pkl"

# Health classification only. Whether an image is a valid kangkong leaf at
# all is handled separately by the gate (see gate.py / train_gate.py) —
# this model should only ever see images that already passed the gate.
CLASS_NAMES = ["healthy", "discolored", "diseased"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Current feature_extractor.py (color hist 512 + LBP 10 + GLCM 24 + edge 1
# + engineered 17) produces a 570-length vector. If this ever drifts, it
# almost always means a stale/duplicate feature_extractor.py got imported
# instead of the one this script sits next to - which is exactly what
# caused the last UI/model mismatch. Fail loudly here instead of silently
# training a model the UI can't use.
EXPECTED_FEATURE_LENGTH = 571

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
    """
    Expects:
        dataset/healthy/...
        dataset/discolored/...
        dataset/diseased/...
    """
    X = []
    y = []
    paths = []

    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}

    for label in CLASS_NAMES:
        class_dir = DATASET_DIR / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        n_loaded = 0
        for file in class_dir.iterdir():
            if file.suffix.lower() in VALID_EXTENSIONS:
                try:
                    feat = extract_features(file)
                    X.append(feat)
                    y.append(label_to_idx[label])
                    paths.append(str(file))
                    n_loaded += 1
                except Exception as e:
                    print(f"[WARNING] Skipped {file}: {e}")

        print(f"  {label}: {n_loaded} images")

    if len(X) == 0:
        raise ValueError("No valid images found in dataset.")

    return np.array(X, dtype=np.float32), np.array(y), paths, label_to_idx


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_confusion_matrix(cm, class_names, path, normalize=False, title=""):
    if normalize:
        cm_display = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format=fmt, colorbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_results_grid(stage_metrics, path, title="Training Results"):
    """
    GBM equivalent of the YOLO 'results.png' grid: one row of train-set
    curves and one row of val-set curves, plotted per boosting stage.

    stage_metrics: dict with keys
        train_loss, val_loss, train_acc, val_acc,
        train_precision, val_precision, train_recall, val_recall
    each a list of per-stage values (stage 1..N).
    """
    n_stages = len(stage_metrics["train_loss"])
    x = range(1, n_stages + 1)

    panels = [
        ("Loss (log loss)", "train_loss", "val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("Precision (macro)", "train_precision", "val_precision"),
        ("Recall (macro)", "train_recall", "val_recall"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for col, (label, train_key, val_key) in enumerate(panels):
        ax_train = axes[0, col]
        ax_train.plot(x, stage_metrics[train_key], color="tab:blue")
        ax_train.set_title(f"train/{label}")
        ax_train.set_xlabel("Boosting stage")
        ax_train.grid(True, alpha=0.3)

        ax_val = axes[1, col]
        ax_val.plot(x, stage_metrics[val_key], color="tab:orange")
        ax_val.set_title(f"val/{label}")
        ax_val.set_xlabel("Boosting stage")
        ax_val.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def report_pairwise_confusion(cm, class_names, label_a, label_b, out_lines):
    """Explicitly surface how often two specific classes get mixed up."""
    idx_a = class_names.index(label_a)
    idx_b = class_names.index(label_b)

    a_total = cm[idx_a].sum()
    b_total = cm[idx_b].sum()
    a_as_b = cm[idx_a, idx_b]
    b_as_a = cm[idx_b, idx_a]

    out_lines.append(f"\n--- {label_a} vs {label_b} confusion ---")
    out_lines.append(
        f"{label_a} predicted as {label_b}: {a_as_b}/{a_total} "
        f"({(a_as_b / a_total * 100 if a_total else 0):.1f}%)"
    )
    out_lines.append(
        f"{label_b} predicted as {label_a}: {b_as_a}/{b_total} "
        f"({(b_as_a / b_total * 100 if b_total else 0):.1f}%)"
    )


def main():
    print("Loading dataset...")
    X, y, paths, label_to_idx = load_dataset()
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature length: {X.shape[1]}")

    if X.shape[1] != EXPECTED_FEATURE_LENGTH:
        raise RuntimeError(
            f"Feature length mismatch: got {X.shape[1]}, expected "
            f"{EXPECTED_FEATURE_LENGTH}. This almost always means the "
            f"feature_extractor.py being imported here (see path printed "
            f"above) is not the current one. Fix the import before "
            f"training, or this model will fail in the UI the same way "
            f"the last one did."
        )

    # -----------------------------------------
    # FINAL TEST SET (keep this untouched)
    # -----------------------------------------
    X_dev, X_test, y_dev, y_test, paths_dev, test_paths = train_test_split(
        X, y, paths,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    print(f"Development set: {len(X_dev)}")
    print(f"Final test set:   {len(X_test)}")

    sample_weight_dev = compute_sample_weight(class_weight="balanced", y=y_dev)

    # -----------------------------------------
    # CROSS-VALIDATION ON DEVELOPMENT SET
    # -----------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        "n_estimators": [25, 30, 35, 40, 45, 50, 60, 80, 100],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": [None, "sqrt", "log2"],
        # min_samples_leaf no longer allows 1: a leaf built from a single
        # training sample is pure memorization, which is exactly what the
        # train/val curve showed happening (train accuracy hit 1.0 while
        # validation stayed noisy).
        "min_samples_leaf": [2, 4, 8, 12],
        "min_samples_split": [4, 8, 16],
    }

    # Early stopping: each candidate GBM carves off part of its own training
    # fold as an internal validation set and stops adding boosting stages
    # once that validation score stops improving for `n_iter_no_change`
    # rounds in a row. This directly targets the pattern in the diagnostic
    # curve — training accuracy climbing to 1.0 long after validation
    # accuracy has plateaued — by not letting the model keep boosting past
    # the point where it's just fitting training noise.
    gbm = GradientBoostingClassifier(
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=10,
        tol=1e-4,
    )

    search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("Running cross-validated hyperparameter search...")
    search.fit(X_dev, y_dev, sample_weight=sample_weight_dev)

    best_model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)

    print(f"\nBest CV macro F1: {search.best_score_:.4f}")

    # -----------------------------------------
    # SAVE CV RESULTS SUMMARY
    # -----------------------------------------
    results = search.cv_results_
    ranked_idx = np.argsort(results["rank_test_score"])

    lines = []
    lines.append(f"Best CV macro F1: {search.best_score_:.4f}")
    lines.append(f"Best parameters: {search.best_params_}")
    lines.append("\nTop 10 parameter sets (sorted by CV score):\n")

    top10 = ranked_idx[:10]
    gaps = []

    for rank_pos, idx in enumerate(top10, start=1):
        mean_test = results["mean_test_score"][idx]
        std_test = results["std_test_score"][idx]
        mean_train = results["mean_train_score"][idx]
        params = results["params"][idx]
        gap = mean_train - mean_test
        gaps.append((idx, gap, mean_test))

        lines.append(
            f"{rank_pos}. mean_cv_f1={mean_test:.4f} ± {std_test:.4f} | "
            f"mean_train_f1={mean_train:.4f} | train_cv_gap={gap:.4f} | params={params}"
        )

    # A large train/CV gap means the model is fitting patterns that don't
    # hold up outside its own training folds — a smaller gap usually means
    # steadier real-world performance even if the raw CV score is a touch
    # lower. Surface this instead of only optimizing for the top score.
    best_idx, best_gap, best_test = gaps[0]
    smallest_gap_entry = min(gaps, key=lambda g: g[1])

    if smallest_gap_entry[0] != best_idx:
        sg_idx, sg_gap, sg_test = smallest_gap_entry
        lines.append(
            "\nNote: the #1 CV-score candidate has train_cv_gap="
            f"{best_gap:.4f}. A candidate within the top 10 generalizes "
            f"more tightly (train_cv_gap={sg_gap:.4f}, mean_cv_f1={sg_test:.4f}): "
            f"{results['params'][sg_idx]}\n"
            "Consider this alternative if you want steadier real-world "
            "behavior over the single best CV score."
        )
    else:
        lines.append(
            f"\nNote: the #1 CV-score candidate also has the smallest "
            f"train_cv_gap ({best_gap:.4f}) among the top 10 — good sign, "
            "no trade-off needed here."
        )

    save_text(OUTPUT_DIR / "cv_results_summary.txt", "\n".join(lines))

    # -----------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------
    y_test_pred = best_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_report = classification_report(
        y_test, y_test_pred, target_names=CLASS_NAMES, digits=4
    )
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Macro F1: {test_f1_macro:.4f}")
    print(test_report)
    print("Confusion Matrix:\n", cm)

    report_lines = [
        f"Final Test Accuracy: {test_acc:.4f}",
        f"Final Test Macro F1: {test_f1_macro:.4f}",
        "",
        test_report,
    ]

    report_pairwise_confusion(cm, CLASS_NAMES, "discolored", "diseased", report_lines)

    save_text(OUTPUT_DIR / "final_test_report.txt", "\n".join(report_lines))
    save_text(OUTPUT_DIR / "final_test_confusion_matrix.txt", str(cm))

    plot_confusion_matrix(
        cm, CLASS_NAMES,
        OUTPUT_DIR / "final_test_confusion_matrix.png",
        normalize=False,
        title="Final Test Confusion Matrix (counts)"
    )
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        OUTPUT_DIR / "final_test_confusion_matrix_normalized.png",
        normalize=True,
        title="Final Test Confusion Matrix (row-normalized)"
    )

    print("\nMisclassified final test images:")
    mis_lines = []
    for i in range(len(y_test)):
        if y_test[i] != y_test_pred[i]:
            true_label = CLASS_NAMES[y_test[i]]
            pred_label = CLASS_NAMES[y_test_pred[i]]
            line = f"- {test_paths[i]} | TRUE={true_label} -> PRED={pred_label}"
            print(line)
            mis_lines.append(line)

    save_text(OUTPUT_DIR / "final_test_misclassified.txt", "\n".join(mis_lines))
    print("\nSample correctly classified final test images:")
    import random
    random.seed(42)

    correct_by_class = {name: [] for name in CLASS_NAMES}
    for i in range(len(y_test)):
        if y_test[i] == y_test_pred[i]:
            true_label = CLASS_NAMES[y_test[i]]
            correct_by_class[true_label].append(test_paths[i])

    correct_lines = []
    for class_name in CLASS_NAMES:
        samples = correct_by_class[class_name]
        n_pick = min(2, len(samples))
        picked = random.sample(samples, n_pick)
        for p in picked:
            line = f"- {p} | TRUE={class_name} -> PRED={class_name}"
            print(line)
            correct_lines.append(line)

    save_text(OUTPUT_DIR / "final_test_correct_samples.txt", "\n".join(correct_lines))
    # -----------------------------------------
    # OPTIONAL: STAGED CURVE ON FULL DEV SET
    # -----------------------------------------
    X_train_curve, X_val_curve, y_train_curve, y_val_curve, sw_train_curve, _ = train_test_split(
        X_dev, y_dev, sample_weight_dev,
        test_size=0.2,
        random_state=42,
        stratify=y_dev
    )

    curve_model = GradientBoostingClassifier(
        **search.best_params_,
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=10,
        tol=1e-4,
    )
    curve_model.fit(X_train_curve, y_train_curve, sample_weight=sw_train_curve)

    all_class_labels = list(range(len(CLASS_NAMES)))

    train_stage_acc = []
    val_stage_acc = []
    train_stage_loss = []
    val_stage_loss = []
    train_stage_precision = []
    val_stage_precision = []
    train_stage_recall = []
    val_stage_recall = []

    for (
        y_train_stage_pred,
        y_val_stage_pred,
        train_stage_proba,
        val_stage_proba,
    ) in zip(
        curve_model.staged_predict(X_train_curve),
        curve_model.staged_predict(X_val_curve),
        curve_model.staged_predict_proba(X_train_curve),
        curve_model.staged_predict_proba(X_val_curve),
    ):
        train_stage_acc.append(accuracy_score(y_train_curve, y_train_stage_pred))
        val_stage_acc.append(accuracy_score(y_val_curve, y_val_stage_pred))

        # labels= pins the class order/count so log_loss doesn't choke if a
        # given stage's predictions happen to miss a class entirely.
        train_stage_loss.append(
            log_loss(y_train_curve, train_stage_proba, labels=all_class_labels)
        )
        val_stage_loss.append(
            log_loss(y_val_curve, val_stage_proba, labels=all_class_labels)
        )

        train_stage_precision.append(
            precision_score(y_train_curve, y_train_stage_pred, average="macro", zero_division=0)
        )
        val_stage_precision.append(
            precision_score(y_val_curve, y_val_stage_pred, average="macro", zero_division=0)
        )

        train_stage_recall.append(
            recall_score(y_train_curve, y_train_stage_pred, average="macro", zero_division=0)
        )
        val_stage_recall.append(
            recall_score(y_val_curve, y_val_stage_pred, average="macro", zero_division=0)
        )

    # Original single-panel accuracy curve — kept as-is for continuity with
    # earlier reports/write-ups that already reference this file.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_stage_acc) + 1), train_stage_acc, label="Training Accuracy")
    plt.plot(range(1, len(val_stage_acc) + 1), val_stage_acc, label="Validation Accuracy")
    plt.xlabel("Number of Boosting Stages")
    plt.ylabel("Accuracy")
    plt.title("GBM Training vs Validation Accuracy (Diagnostic)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "cv_diagnostic_train_val_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved diagnostic curve to: {plot_path}")

    # New multi-panel grid (loss / accuracy / precision / recall, train row
    # + val row) — the GBM equivalent of a YOLO-style training results figure.
    stage_metrics = {
        "train_loss": train_stage_loss,
        "val_loss": val_stage_loss,
        "train_acc": train_stage_acc,
        "val_acc": val_stage_acc,
        "train_precision": train_stage_precision,
        "val_precision": val_stage_precision,
        "train_recall": train_stage_recall,
        "val_recall": val_stage_recall,
    }

    grid_path = OUTPUT_DIR / "training_results_grid.png"
    plot_training_results_grid(stage_metrics, grid_path, title="GBM Training Results")
    print(f"Saved training results grid to: {grid_path}")

    save_text(
        OUTPUT_DIR / "training_results_grid_values.txt",
        "\n".join(
            f"stage={i+1} "
            f"train_loss={stage_metrics['train_loss'][i]:.4f} val_loss={stage_metrics['val_loss'][i]:.4f} "
            f"train_acc={stage_metrics['train_acc'][i]:.4f} val_acc={stage_metrics['val_acc'][i]:.4f} "
            f"train_precision={stage_metrics['train_precision'][i]:.4f} val_precision={stage_metrics['val_precision'][i]:.4f} "
            f"train_recall={stage_metrics['train_recall'][i]:.4f} val_recall={stage_metrics['val_recall'][i]:.4f}"
            for i in range(len(train_stage_acc))
        ),
    )

    # -----------------------------------------
    # SAVE FINAL MODEL
    # -----------------------------------------
    save_obj = {
        "model": best_model,
        "class_names": CLASS_NAMES,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feature_length": X.shape[1],
        "best_params": search.best_params_,
        "best_cv_macro_f1": search.best_score_
    }

    joblib.dump(save_obj, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
