"""
Trains an optional "gate" model: is this image a kangkong leaf at all,
or something else (other plant, background, junk photo, etc.)?

This is stage 1 of the two-stage pipeline. Stage 2 (train_model.py) only
ever sees images that pass this gate, so it can stay focused purely on
healthy vs. discolored vs. diseased.

You do NOT need this script to get a working gate — segment_leaf() in
feature_extractor.py already rejects obviously-invalid images for free
(see predict.py, which uses that rule-based check by default). Only run
this if you find real "other plant" photos slipping past the rule-based
check and want a learned second layer.

Expects:
    dataset/healthy/...        \
    dataset/discolored/...      >  all treated as the positive "leaf" class
    dataset/diseased/...       /
    dataset/other/...          -> negative "not a kangkong leaf" class
                                   (other plants/leaves, background, blurry
                                   shots, random objects, etc.)
"""

from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from feature_extractor import extract_features

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
GATE_MODEL_PATH = MODEL_DIR / "gate_model.pkl"

HEALTH_CLASSES = ["healthy", "discolored", "diseased"]
OTHER_CLASS = "other"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_gate_dataset():
    other_dir = DATASET_DIR / OTHER_CLASS
    if not other_dir.exists():
        raise FileNotFoundError(
            f"Missing folder: {other_dir}\n"
            "The gate model needs negative examples (photos that are NOT "
            "kangkong leaves) in dataset/other/ before it can be trained."
        )

    X, y, paths = [], [], []

    for label_dir, label in [(DATASET_DIR / c, 1) for c in HEALTH_CLASSES] + [(other_dir, 0)]:
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing folder: {label_dir}")

        n_loaded = 0
        for file in label_dir.iterdir():
            if file.suffix.lower() in VALID_EXTENSIONS:
                try:
                    feat = extract_features(file)
                    X.append(feat)
                    y.append(label)
                    paths.append(str(file))
                    n_loaded += 1
                except Exception as e:
                    print(f"[WARNING] Skipped {file}: {e}")

        tag = "leaf (positive)" if label == 1 else "other (negative)"
        print(f"  {label_dir.name} [{tag}]: {n_loaded} images")

    if len(X) == 0:
        raise ValueError("No valid images found for gate training.")

    return np.array(X, dtype=np.float32), np.array(y), paths


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    print("Loading gate dataset (leaf vs. other)...")
    X, y, paths = load_gate_dataset()

    print(f"\nTotal samples: {len(X)}  (leaf={int(y.sum())}, other={int((1 - y).sum())})")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    sample_weight_dev = compute_sample_weight(class_weight="balanced", y=y_dev)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_dist = {
        "n_estimators": [30, 40, 50, 60, 80],
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth": [2, 3],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": [None, "sqrt"],
    }

    gbm = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("Running cross-validated search for the gate model...")
    search.fit(X_dev, y_dev, sample_weight=sample_weight_dev)

    best_model = search.best_estimator_
    print(f"\nBest gate params: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")

    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    report = classification_report(y_test, y_test_pred, target_names=["other", "leaf"], digits=4)
    cm = confusion_matrix(y_test, y_test_pred)

    print(f"\nGate Test Accuracy: {test_acc:.4f}")
    print(f"Gate Test Macro F1: {test_f1:.4f}")
    print(report)
    print("Confusion Matrix:\n", cm)

    save_text(
        OUTPUT_DIR / "gate_test_report.txt",
        f"Gate Test Accuracy: {test_acc:.4f}\nGate Test Macro F1: {test_f1:.4f}\n\n{report}\n{cm}"
    )

    save_obj = {
        "model": best_model,
        "positive_class": "leaf",
        "negative_class": "other",
        "feature_length": X.shape[1],
        "best_params": search.best_params_,
    }
    joblib.dump(save_obj, GATE_MODEL_PATH)
    print(f"\nGate model saved to: {GATE_MODEL_PATH}")


if __name__ == "__main__":
    main()