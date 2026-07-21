from pathlib import Path
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from feature_extractor import extract_features

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_PATH = MODEL_DIR / "gbm_model.pkl"

CLASS_NAMES = ["healthy", "discolored", "diseased"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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

        for file in class_dir.iterdir():
            if file.suffix.lower() in VALID_EXTENSIONS:
                try:
                    feat = extract_features(file)
                    X.append(feat)
                    y.append(label_to_idx[label])
                    paths.append(str(file))
                except Exception as e:
                    print(f"[WARNING] Skipped {file}: {e}")

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

    print(f"Total samples: {len(X)}")
    print(f"Feature length: {X.shape[1]}")

    # 70 train, 15 val, 15 test
    X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
        X, y, paths,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test, val_paths, test_paths = train_test_split(
        X_temp, y_temp, paths_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"Train: {len(X_train)}")
    print(f"Val:   {len(X_val)}")
    print(f"Test:  {len(X_test)}")

    gbm = GradientBoostingClassifier(random_state=42)

    param_dist = {
        "n_estimators": [25, 30, 35, 40, 45, 50, 60],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "max_depth": [2, 3],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": [None, "sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("Training model...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best parameters:", search.best_params_)

    # Validation results
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(
        y_val, y_val_pred, target_names=CLASS_NAMES, digits=4
    )

    print("\nValidation Accuracy:", val_acc)
    print(val_report)

    # Test results
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(
        y_test, y_test_pred, target_names=CLASS_NAMES, digits=4
    )
    cm = confusion_matrix(y_test, y_test_pred)

    print("\nTest Accuracy:", test_acc)
    print(test_report)
    print("Confusion Matrix:\n", cm)

    save_text(OUTPUT_DIR / "val_report.txt", f"Validation Accuracy: {val_acc:.4f}\n\n{val_report}")
    save_text(OUTPUT_DIR / "test_report.txt", f"Test Accuracy: {test_acc:.4f}\n\n{test_report}")
    save_text(OUTPUT_DIR / "confusion_matrix.txt", str(cm))

    save_obj = {
        "model": best_model,
        "class_names": CLASS_NAMES,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feature_length": X.shape[1]
    }

    joblib.dump(save_obj, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    y_test_pred = best_model.predict(X_test)

    print("\nMisclassified test images:")
    for i in range(len(y_test)):
        if y_test[i] != y_test_pred[i]:
            true_label = CLASS_NAMES[y_test[i]]
            pred_label = CLASS_NAMES[y_test_pred[i]]
            print(f"- {test_paths[i]} | TRUE={true_label} -> PRED={pred_label}")

    # ---------------------------
    # Visualization: train vs validation accuracy over boosting stages
    # ---------------------------
    train_stage_acc = []
    val_stage_acc = []

    for y_train_stage_pred, y_val_stage_pred in zip(
        best_model.staged_predict(X_train),
        best_model.staged_predict(X_val)
    ):
        train_stage_acc.append(accuracy_score(y_train, y_train_stage_pred))
        val_stage_acc.append(accuracy_score(y_val, y_val_stage_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_stage_acc) + 1), train_stage_acc, label="Training Accuracy")
    plt.plot(range(1, len(val_stage_acc) + 1), val_stage_acc, label="Validation Accuracy")
    plt.xlabel("Number of Boosting Stages")
    plt.ylabel("Accuracy")
    plt.title("GBM Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "train_val_accuracy_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    main()