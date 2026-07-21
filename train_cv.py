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
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from feature_extractor import extract_features

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

    train_stage_acc = []
    val_stage_acc = []

    for y_train_stage_pred, y_val_stage_pred in zip(
        curve_model.staged_predict(X_train_curve),
        curve_model.staged_predict(X_val_curve)
    ):
        train_stage_acc.append(accuracy_score(y_train_curve, y_train_stage_pred))
        val_stage_acc.append(accuracy_score(y_val_curve, y_val_stage_pred))

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