"""
Two-stage inference:
  1. Gate: is this actually a kangkong leaf?
     - Always applies the free, rule-based check from segment_leaf()
       (seg_rejected flag from feature_extractor.py).
     - If a trained gate model exists (models/gate_model.pkl, produced by
       train_gate.py), also applies it as a second check.
  2. Health: if the image passes the gate, run the 3-class GBM
     (models/gbm_model_cv.pkl, produced by train_model.py) to get
     healthy / discolored / diseased.

Usage:
    python predict.py path/to/image.jpg
"""

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from feature_extractor import extract_features_from_bgr, segment_leaf, IMG_SIZE

MODEL_DIR = Path("models")
HEALTH_MODEL_PATH = MODEL_DIR / "gbm_model_cv.pkl"
GATE_MODEL_PATH = MODEL_DIR / "gate_model.pkl"

# Only used if a trained gate model is present. Lower = stricter about
# letting things through as "leaf".
GATE_THRESHOLD = 0.5


def load_models():
    if not HEALTH_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HEALTH_MODEL_PATH}. Run train_model.py first."
        )
    health_obj = joblib.load(HEALTH_MODEL_PATH)

    gate_obj = None
    if GATE_MODEL_PATH.exists():
        gate_obj = joblib.load(GATE_MODEL_PATH)

    return health_obj, gate_obj


def predict_image(image_path, health_obj, gate_obj=None):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))

    # --- Stage 1a: free rule-based gate (always applied) ---
    _, leaf_ratio, seg_rejected = segment_leaf(image_resized)
    if seg_rejected:
        return {
            "label": "unclassified",
            "reason": "rule_based_gate",
            "leaf_ratio": float(leaf_ratio),
        }

    # Feature vector is computed once and reused for both the optional
    # gate model and the health model, since they use the same features.
    features = extract_features_from_bgr(image_bgr).reshape(1, -1)

    # --- Stage 1b: learned gate (only if trained) ---
    if gate_obj is not None:
        gate_model = gate_obj["model"]
        gate_pred_proba = gate_model.predict_proba(features)[0]
        # class 1 = "leaf" (see train_gate.py)
        leaf_proba = gate_pred_proba[1]

        if leaf_proba < GATE_THRESHOLD:
            return {
                "label": "unclassified",
                "reason": "learned_gate",
                "leaf_probability": float(leaf_proba),
            }

    # --- Stage 2: health classification ---
    health_model = health_obj["model"]
    idx_to_label = health_obj["idx_to_label"]

    pred_idx = health_model.predict(features)[0]
    pred_proba = health_model.predict_proba(features)[0]

    return {
        "label": idx_to_label[pred_idx],
        "reason": "health_model",
        "probabilities": {
            idx_to_label[i]: float(p) for i, p in enumerate(pred_proba)
        },
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    health_obj, gate_obj = load_models()

    if gate_obj is None:
        print("[INFO] No trained gate model found — using rule-based gate only.\n")

    result = predict_image(image_path, health_obj, gate_obj)

    print(f"Image: {image_path}")
    print(f"Prediction: {result['label']}")
    print(f"Decided by: {result['reason']}")

    if "probabilities" in result:
        print("Probabilities:")
        for label, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {label}: {p:.4f}")
    elif "leaf_probability" in result:
        print(f"Leaf probability: {result['leaf_probability']:.4f} (threshold {GATE_THRESHOLD})")
    elif "leaf_ratio" in result:
        print(f"Segmented leaf ratio: {result['leaf_ratio']:.4f}")


if __name__ == "__main__":
    main()