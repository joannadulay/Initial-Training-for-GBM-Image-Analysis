from pathlib import Path
import joblib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from feature_extractor import extract_features

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = MODEL_DIR / "knn_model_cv.pkl"

CLASS_NAMES = ["healthy", "discolored", "diseased"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_FEATURE_LENGTH = 571

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
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


def main():
    print("Loading dataset...")
    X, y, paths, label_to_idx = load_dataset()
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature length: {X.shape[1]}")

    if X.shape[1] != EXPECTED_FEATURE_LENGTH:
        raise RuntimeError(
            f"Feature length mismatch: got {X.shape[1]}, expected "
            f"{EXPECTED_FEATURE_LENGTH}. Check which feature_extractor.py "
            f"is being imported."
        )

    # -----------------------------------------
    # SAME SPLIT METHODOLOGY AS train_cv.py
    # -----------------------------------------
    X_dev, X_test, y_dev, y_test, paths_dev, test_paths = train_test_split(
        X, y, paths,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    print(f"Development set: {len(X_dev)}")
    print(f"Final test set:   {len(X_test)}")

    # NOTE: k-NN has no native class_weight / sample_weight support at fit
    # time (it just looks up the k nearest neighbors), so class imbalance
    # here is instead handled via weights="distance" in the param grid,
    # which lets closer neighbors count more than farther ones - this is
    # not the same mechanism as the class-balanced sample weighting used
    # for GBM/RF/SVM, so that difference should be noted when comparing.

    # -----------------------------------------
    # CROSS-VALIDATION ON DEVELOPMENT SET
    # -----------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # k-NN is distance-based like SVM, so features are standardized first
    # for the same reason (raw-scale features would otherwise dominate the
    # distance calculation regardless of how informative they are).
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    param_dist = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan", "minkowski"],
        "knn__p": [1, 2],  # only used when metric="minkowski"
    }

    search = RandomizedSearchCV(
        estimator=pipe,
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
    search.fit(X_dev, y_dev)

    best_model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)
    print(f"\nBest CV macro F1: {search.best_score_:.4f}")

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
        f"Best CV macro F1: {search.best_score_:.4f}",
        f"Best parameters: {search.best_params_}",
        "",
        f"Final Test Accuracy: {test_acc:.4f}",
        f"Final Test Macro F1: {test_f1_macro:.4f}",
        "",
        test_report,
        "",
        "Confusion Matrix:",
        str(cm),
    ]

    save_text(OUTPUT_DIR / "knn_final_test_report.txt", "\n".join(report_lines))

    print("\nMisclassified test images:")
    mis_lines = []
    for i in range(len(y_test)):
        if y_test[i] != y_test_pred[i]:
            true_label = CLASS_NAMES[y_test[i]]
            pred_label = CLASS_NAMES[y_test_pred[i]]
            line = f"- {test_paths[i]} | TRUE={true_label} -> PRED={pred_label}"
            print(line)
            mis_lines.append(line)

    save_text(OUTPUT_DIR / "knn_final_test_misclassified.txt", "\n".join(mis_lines))

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
