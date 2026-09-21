import os
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from feature_extractor import (
    extract_features_from_bgr,
    extract_features_with_details_from_bgr,
    segment_leaf,
    count_leaflike_components,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gbm_model_cv.pkl")
GATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "gate_model.pkl")

# Analysis visuals live at the project root, as a sibling of captured_images/,
# not nested inside it.
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ANALYSIS_VISUALS_DIR = os.path.join(PROJECT_ROOT, "analysis_visuals")

# Rejection / acceptance thresholds
MIN_LEAF_RATIO = 0.03
MAX_LEAF_RATIO = 0.70
CONFIDENCE_THRESHOLD = 0.40
GATE_THRESHOLD = 0.5  # only used if models/gate_model.pkl exists

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "discolored": "Discolored",
    "diseased": "Diseased",
}

PAYLOAD = None
MODEL = None
CLASS_NAMES = None
LOAD_ERROR = None

GATE_PAYLOAD = None
GATE_MODEL = None
GATE_LOAD_ATTEMPTED = False


def _install_numpy_pickle_compat() -> None:
    """Helps older NumPy runtimes load pickles written with NumPy 2.x paths."""
    try:
        import numpy._core  # noqa: F401
        return
    except Exception:
        pass

    try:
        import numpy.core as npcore
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric
        import numpy.core.umath as umath

        try:
            import numpy.core._multiarray_umath as multiarray_umath
        except Exception:
            multiarray_umath = multiarray

        sys.modules.setdefault("numpy._core", npcore)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.numeric", numeric)
        sys.modules.setdefault("numpy._core.umath", umath)
        sys.modules.setdefault("numpy._core._multiarray_umath", multiarray_umath)
    except Exception:
        pass


def _ensure_model_loaded() -> None:
    global PAYLOAD, MODEL, CLASS_NAMES, LOAD_ERROR

    if PAYLOAD is not None:
        return

    if LOAD_ERROR is not None:
        raise RuntimeError(LOAD_ERROR)

    if not os.path.exists(MODEL_PATH):
        LOAD_ERROR = (
            "Could not find models/gbm_model_cv.pkl. "
            f"Expected location: {MODEL_PATH}"
        )
        raise RuntimeError(LOAD_ERROR)

    _install_numpy_pickle_compat()

    try:
        payload = joblib.load(MODEL_PATH)
    except Exception as exc:
        LOAD_ERROR = (
            "Could not load models/gbm_model_cv.pkl. "
            "Make sure your UI is using the same ML environment used to train/save the model. "
            f"Original error: {exc}"
        )
        raise RuntimeError(LOAD_ERROR) from exc

    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(
            "gbm_model_cv.pkl does not contain the expected payload. "
            "Expected a dictionary with at least: {'model': ..., 'class_names': ...}."
        )

    if "class_names" not in payload:
        raise RuntimeError(
            "gbm_model_cv.pkl is missing 'class_names'. "
            "Please save class_names together with the model during training."
        )

    PAYLOAD = payload
    MODEL = payload["model"]
    CLASS_NAMES = payload["class_names"]


def _ensure_gate_loaded() -> None:
    """
    Loads the optional leaf-vs-other gate model (produced by train_gate.py),
    if one has been trained. This is intentionally best-effort: if the file
    doesn't exist, the app runs exactly as before with no gate. If the file
    exists but fails to load for some reason, we skip the gate rather than
    crash the whole prediction.
    """
    global GATE_PAYLOAD, GATE_MODEL, GATE_LOAD_ATTEMPTED

    if GATE_LOAD_ATTEMPTED:
        return
    GATE_LOAD_ATTEMPTED = True

    if not os.path.exists(GATE_MODEL_PATH):
        return

    try:
        _install_numpy_pickle_compat()
        payload = joblib.load(GATE_MODEL_PATH)
        if isinstance(payload, dict) and "model" in payload:
            GATE_PAYLOAD = payload
            GATE_MODEL = payload["model"]
    except Exception as exc:
        print(f"[predict_core] Gate model failed to load, skipping gate: {exc}")


def _class_id_to_name(class_id) -> str:
    """
    Converts model class IDs into readable names.
    Works with class_names as either:
    - list/tuple: ['healthy', 'diseased', 'discolored']
    - dict: {0: 'healthy', 1: 'diseased', 2: 'discolored'}
    """
    _ensure_model_loaded()

    try:
        if isinstance(CLASS_NAMES, dict):
            raw = CLASS_NAMES.get(class_id, CLASS_NAMES.get(int(class_id), str(class_id)))
        else:
            raw = CLASS_NAMES[int(class_id)]
    except Exception:
        raw = str(class_id)

    raw = str(raw).strip().lower()
    return DISPLAY_NAMES.get(raw, raw.capitalize())


def _blank_probs():
    return {
        "Healthy": 0.0,
        "Discolored": 0.0,
        "Diseased": 0.0,
    }


def _make_analysis_visual(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    segmented_bgr: np.ndarray,
    result_text: str,
    reason_text: str,
    prob_dict: dict,
    leaf_ratio: float,
    component_count: int,
    image_path: str,
) -> str:
    """
    Creates a single image like predict_visual.py:
    Original image, detected mask, and segmented region side-by-side on top,
    with a full-width confidence bar chart below.
    This image can be shown inside the Tkinter UI after Analyze is pressed.

    Saved under the project-level analysis_visuals/ folder (a sibling of
    captured_images/), named after the source image.
    """
    image_path = Path(image_path)
    out_dir = Path(ANALYSIS_VISUALS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_analysis.jpg"

    W, H = 1200, 720
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    def put(text, x, y, scale=0.7, thickness=2):
        cv2.putText(
            canvas,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def fit_img(img, box_w, box_h):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min(box_w / max(w, 1), box_h / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        box = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
        x0 = (box_w - nw) // 2
        y0 = (box_h - nh) // 2
        box[y0:y0 + nh, x0:x0 + nw] = resized
        return box

    def paste(img, x, y, box_w=320, box_h=240):
        box = fit_img(img, box_w, box_h)
        canvas[y:y + box_h, x:x + box_w] = box

    # Header
    put(f"Result: {result_text}", 40, 38, 0.85, 2)
    put(f"Reason: {reason_text}", 40, 74, 0.62, 2)

    # --- Top row: three larger image panels side by side ------------------
    panel_w, panel_h = 360, 260
    col_xs = [40, 420, 800]
    title_y = 112
    img_y = 140

    put("Original Image", col_xs[0], title_y, 0.68, 2)
    paste(image_bgr, col_xs[0], img_y, panel_w, panel_h)

    put("Detected Leaf Mask", col_xs[1], title_y, 0.68, 2)
    put(f"Leaf Ratio={leaf_ratio:.4f}  Components={component_count}", col_xs[1], title_y + 22, 0.5, 1)
    paste(mask, col_xs[1], img_y, panel_w, panel_h)

    put("Segmented Leaf Region", col_xs[2], title_y, 0.68, 2)
    paste(segmented_bgr, col_xs[2], img_y, panel_w, panel_h)

    # --- Bottom: full-width confidence chart, color-coded by class --------
    # Colors match the rest of the app's theme (green/yellow/orange), in
    # OpenCV's BGR order.
    BAR_COLORS_BGR = {
        "Healthy": (110, 191, 47),      # green
        "Discolored": (61, 185, 224),   # yellow
        "Diseased": (58, 151, 240),     # orange
    }

    chart_x, chart_y = 40, 440
    chart_w, chart_h = W - 80, 170
    put("Prediction Confidence Scores", chart_x, chart_y - 16, 0.65, 2)

    cv2.rectangle(canvas, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (0, 0, 0), 1)
    cv2.line(canvas, (chart_x, chart_y + chart_h), (chart_x + chart_w, chart_y + chart_h), (0, 0, 0), 2)
    cv2.line(canvas, (chart_x, chart_y), (chart_x, chart_y + chart_h), (0, 0, 0), 2)

    labels = ["Healthy", "Discolored", "Diseased"]
    vals = [float(prob_dict.get(lbl, 0.0)) for lbl in labels]

    n = len(labels)
    slot_w = chart_w / n
    bar_w = min(160, slot_w * 0.4)

    for i, (lbl, val) in enumerate(zip(labels, vals)):
        slot_center = chart_x + slot_w * i + slot_w / 2
        x1 = int(slot_center - bar_w / 2)
        x2 = int(slot_center + bar_w / 2)
        bar_h = int(max(0.0, min(1.0, val)) * (chart_h - 30))
        y1 = chart_y + chart_h - bar_h
        y2 = chart_y + chart_h
        bar_color = BAR_COLORS_BGR.get(lbl, (80, 140, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bar_color, -1)
        put(f"{val:.3f}", int(slot_center - 24), max(chart_y + 20, y1 - 10), 0.5, 1)
        put(lbl, int(slot_center - 34), chart_y + chart_h + 30, 0.5, 1)

    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def get_prediction(image_path: str, return_details: bool = False):
    """
    Called by predict_runner.py.

    Two-stage pipeline:
      1. Gate — rejects images that don't look like a valid kangkong leaf.
         Always applies the free rule-based check (segmentation rejection,
         leaf-region count). If models/gate_model.pkl exists (produced by
         train_gate.py), also applies the learned leaf-vs-other classifier
         as a second check.
      2. Health — only images that pass the gate reach the 3-class GBM.

    Returns by default:
        label, probabilities

    If return_details=True:
        label, probabilities, details

    The details dictionary contains:
        reason, leaf_ratio, component_count, visual_path, gate_leaf_probability,
        timings (per-stage milliseconds: Image Loading, Segmentation,
        Feature Extraction, Gate Model, GBM Predict, Predict Probability,
        Visualization)
    """
    _ensure_model_loaded()
    _ensure_gate_loaded()

    timings = {}

    t0 = time.perf_counter()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        timings["Image Loading"] = (time.perf_counter() - t0) * 1000
        label = "Image Error"
        probs = _blank_probs()
        details = {
            "reason": "Image could not be read.",
            "leaf_ratio": 0.0,
            "component_count": 0,
            "visual_path": "",
            "timings": timings,
        }
        return (label, probs, details) if return_details else (label, probs)

    image_bgr = cv2.resize(image_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    timings["Image Loading"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    component_count = count_leaflike_components(image_bgr)
    mask, leaf_ratio, seg_rejected = segment_leaf(image_bgr)
    timings["Segmentation"] = (time.perf_counter() - t0) * 1000

    segmented_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    result_text = "No clear kangkong leaf detected"
    reason_text = ""
    probs = _blank_probs()
    gate_leaf_probability = None

    run_model = False
    features = None
    feature_details = None

    if component_count == 0:
        reason_text = "No leaf-like region detected."
    elif component_count > 1:
        reason_text = f"Multiple leaf-like regions detected ({component_count}). Please isolate one leaf."
    elif seg_rejected:
        reason_text = f"Leaf region rejected by segmentation (leaf ratio={leaf_ratio:.4f})."
    elif leaf_ratio < MIN_LEAF_RATIO:
        reason_text = f"Leaf area too small ({leaf_ratio:.4f})."
    elif leaf_ratio > MAX_LEAF_RATIO:
        reason_text = f"Scene too cluttered or mask too large ({leaf_ratio:.4f}). Please isolate one leaf."
    else:
        # Feature vector is computed once and reused for both the optional
        # gate model and the health model.
        t0 = time.perf_counter()
        features, feature_details = extract_features_with_details_from_bgr(image_bgr)
        features = features.reshape(1, -1)
        timings["Feature Extraction"] = (time.perf_counter() - t0) * 1000

        gate_ok = True
        if GATE_MODEL is not None:
            try:
                t0 = time.perf_counter()
                gate_leaf_probability = float(GATE_MODEL.predict_proba(features)[0][1])
                timings["Gate Model"] = (time.perf_counter() - t0) * 1000
                gate_ok = gate_leaf_probability >= GATE_THRESHOLD
                if not gate_ok:
                    reason_text = (
                        f"Rejected by gate model (leaf probability={gate_leaf_probability:.4f})."
                    )
            except Exception as exc:
                # If the gate errors out for any reason, don't block a
                # working health model because of it.
                print(f"[predict_core] Gate model prediction failed, skipping gate: {exc}")
                gate_ok = True

        run_model = gate_ok

    if run_model and features is not None:
        if hasattr(MODEL, "n_features_in_") and features.shape[1] != MODEL.n_features_in_:
            raise ValueError(
                f"Feature length mismatch: got {features.shape[1]} features, "
                f"but gbm_model_cv.pkl expects {MODEL.n_features_in_}. "
                "This usually means feature_extractor.py does not match the one used during training."
            )

        t0 = time.perf_counter()
        pred_class = MODEL.predict(features)[0]
        timings["GBM Predict"] = (time.perf_counter() - t0) * 1000
        pred_label = _class_id_to_name(pred_class)

        if hasattr(MODEL, "predict_proba"):
            t0 = time.perf_counter()
            raw_probs = MODEL.predict_proba(features)[0]
            timings["Predict Probability"] = (time.perf_counter() - t0) * 1000
            model_classes = getattr(MODEL, "classes_", range(len(raw_probs)))

            for class_id, prob in zip(model_classes, raw_probs):
                label_name = _class_id_to_name(class_id)
                if label_name in probs:
                    probs[label_name] = float(prob)

            max_prob = max(probs.values()) if probs else 0.0
            if max_prob < CONFIDENCE_THRESHOLD:
                result_text = "No clear kangkong leaf detected"
                reason_text = f"Low model confidence ({max_prob:.4f})."
            else:
                result_text = pred_label
                reason_text = f"Accepted prediction (confidence={max_prob:.4f})."
        else:
            result_text = pred_label
            reason_text = "Prediction made without probability output."

    t0 = time.perf_counter()
    visual_path = _make_analysis_visual(
        image_bgr=image_bgr,
        mask=mask,
        segmented_bgr=segmented_bgr,
        result_text=result_text,
        reason_text=reason_text,
        prob_dict=probs,
        leaf_ratio=leaf_ratio,
        component_count=component_count,
        image_path=image_path,
    )
    timings["Visualization"] = (time.perf_counter() - t0) * 1000

    details = {
        "reason": reason_text,
        "leaf_ratio": float(leaf_ratio),
        "component_count": int(component_count),
        "visual_path": visual_path,
        "timings": timings,
    }
    if gate_leaf_probability is not None:
        details["gate_leaf_probability"] = gate_leaf_probability
    if feature_details is not None:
        # yellow_ratio, lesion_ratio, dark_ratio, lesion_count, mean_circularity,
        # fragmentation, edge_density, etc. - used by the "Visual Findings"
        # section of the details popup. Keys already in `details` win, so this
        # never overwrites reason/leaf_ratio/etc.
        for key, value in feature_details.items():
            details.setdefault(key, value)

    return (result_text, probs, details) if return_details else (result_text, probs)
