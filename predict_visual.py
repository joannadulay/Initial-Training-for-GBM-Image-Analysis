from pathlib import Path
import time
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature_extractor import extract_features_from_bgr, segment_leaf, count_leaflike_components

MODEL_PATH = Path("models/gbm_model_cv.pkl")
GATE_MODEL_PATH = Path("models/gate_model.pkl")
IMG_SIZE = 128

# Rejection thresholds
MIN_LEAF_RATIO = 0.03
MAX_LEAF_RATIO = 0.65
CONFIDENCE_THRESHOLD = 0.45
GATE_THRESHOLD = 0.5  # only used if a trained gate model is present

DISPLAY_NAMES = {
    "healthy": "Healthy Kangkong Leaf",
    "discolored": "Discolored Kangkong Leaf",
    "diseased": "Diseased Kangkong Leaf"
}

STAGE_ORDER = [
    "Image Loading", "Segmentation", "Feature Extraction",
    "Gate Model", "GBM Predict", "Predict Probability",
]

_MODEL_CACHE = {}


def _load_models():
    """Load and cache models so repeated benchmark runs don't hit disk each time."""
    if "model_data" not in _MODEL_CACHE:
        _MODEL_CACHE["model_data"] = joblib.load(MODEL_PATH)
        _MODEL_CACHE["gate_data"] = joblib.load(GATE_MODEL_PATH) if GATE_MODEL_PATH.exists() else None
    return _MODEL_CACHE["model_data"], _MODEL_CACHE["gate_data"]


def run_inference(
    image_path,
    model_data,
    gate_data,
    min_leaf_ratio=MIN_LEAF_RATIO,
    max_leaf_ratio=MAX_LEAF_RATIO,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    """
    Runs one full inference pass and returns a dict with the results plus
    per-stage timings in milliseconds. Contains no plotting/printing so it
    can be reused for both a single prediction and repeated benchmarking.
    """
    timings = {}
    model = model_data["model"]
    class_names = model_data["class_names"]
    gate_model = gate_data["model"] if gate_data else None

    # --- Image loading ---
    t0 = time.perf_counter()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        timings["Image Loading"] = (time.perf_counter() - t0) * 1000
        return {
            "result_text": "No clear kangkong leaf detected",
            "reason_text": "Image could not be read.",
            "probs": None, "pred_label": None, "gate_leaf_prob": None,
            "class_names": None, "mask": None, "leaf_ratio": None,
            "scene_component_count": None, "image_rgb": None, "segmented_rgb": None,
            "timings": timings,
        }
    image_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    timings["Image Loading"] = (time.perf_counter() - t0) * 1000

    # --- Component count + segmentation ---
    t0 = time.perf_counter()
    scene_component_count = count_leaflike_components(image_bgr)
    mask, leaf_ratio, seg_rejected = segment_leaf(image_bgr)
    timings["Segmentation"] = (time.perf_counter() - t0) * 1000

    segmented_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    result_text = "No clear kangkong leaf detected"
    reason_text = ""
    probs = None
    pred_label = None
    gate_leaf_prob = None

    if scene_component_count == 0:
        reason_text = "No leaf-like region detected."

    elif scene_component_count > 1:
        reason_text = f"Multiple leaf-like regions detected ({scene_component_count}). Please isolate one leaf."

    elif seg_rejected:
        reason_text = f"Leaf region rejected by segmentation (leaf ratio={leaf_ratio:.4f})."

    elif leaf_ratio < min_leaf_ratio:
        reason_text = f"Leaf area too small ({leaf_ratio:.4f})"

    elif leaf_ratio > max_leaf_ratio:
        reason_text = f"Scene too cluttered or mask too large ({leaf_ratio:.4f}). Please isolate one leaf."

    else:
        # --- Feature extraction ---
        t0 = time.perf_counter()
        features = extract_features_from_bgr(image_bgr).reshape(1, -1)
        timings["Feature Extraction"] = (time.perf_counter() - t0) * 1000

        # --- Learned gate, if models/gate_model.pkl exists ---
        gate_ok = True
        if gate_model is not None:
            t0 = time.perf_counter()
            gate_leaf_prob = float(gate_model.predict_proba(features)[0][1])
            timings["Gate Model"] = (time.perf_counter() - t0) * 1000
            gate_ok = gate_leaf_prob >= GATE_THRESHOLD
            if not gate_ok:
                reason_text = f"Rejected by gate model (leaf probability={gate_leaf_prob:.4f})"

        if gate_ok:
            # --- Class prediction ---
            t0 = time.perf_counter()
            pred_idx = model.predict(features)[0]
            timings["GBM Predict"] = (time.perf_counter() - t0) * 1000
            pred_label = class_names[pred_idx]

            if hasattr(model, "predict_proba"):
                t0 = time.perf_counter()
                probs = model.predict_proba(features)[0]
                timings["Predict Probability"] = (time.perf_counter() - t0) * 1000
                max_prob = float(np.max(probs))

                if max_prob < confidence_threshold:
                    reason_text = f"Low model confidence ({max_prob:.4f})"
                else:
                    result_text = DISPLAY_NAMES.get(pred_label, pred_label)
                    reason_text = f"Accepted prediction (confidence={max_prob:.4f})"
            else:
                result_text = DISPLAY_NAMES.get(pred_label, pred_label)
                reason_text = "Prediction made without probability output"

    return {
        "result_text": result_text,
        "reason_text": reason_text,
        "probs": probs,
        "pred_label": pred_label,
        "gate_leaf_prob": gate_leaf_prob,
        "class_names": class_names,
        "mask": mask,
        "leaf_ratio": leaf_ratio,
        "scene_component_count": scene_component_count,
        "image_rgb": image_rgb,
        "segmented_rgb": segmented_rgb,
        "timings": timings,
    }


def print_performance(timings):
    print("========== PERFORMANCE ==========")
    total = 0.0
    for stage in STAGE_ORDER:
        if stage in timings:
            ms = timings[stage]
            total += ms
            print(f"{stage:<20}: {ms:.3f} ms")
    print("---------------------------------")
    print(f"{'TOTAL PIPELINE':<20}: {total:.3f} ms")
    print("=================================\n")


def predict_and_visualize(
    image_path,
    min_leaf_ratio=MIN_LEAF_RATIO,
    max_leaf_ratio=MAX_LEAF_RATIO,
    confidence_threshold=CONFIDENCE_THRESHOLD
):
    model_data, gate_data = _load_models()
    out = run_inference(image_path, model_data, gate_data, min_leaf_ratio, max_leaf_ratio, confidence_threshold)

    print_performance(out["timings"])
    print(f"Result: {out['result_text']}")
    print(f"Reason: {out['reason_text']}")

    if out["image_rgb"] is None:
        return  # image failed to load, nothing to plot

    if out["probs"] is not None:
        print("\nConfidence scores:")
        for label, prob in zip(out["class_names"], out["probs"]):
            print(f"  {DISPLAY_NAMES.get(label, label)}: {prob:.4f}")

    # --- Visualization ---
    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(out["image_rgb"])
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(out["mask"], cmap="gray")
    ax2.set_title(
        f"Detected Leaf Mask\nLeaf Ratio = {out['leaf_ratio']:.4f}\nComponents = {out['scene_component_count']}"
    )
    ax2.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(out["segmented_rgb"])
    ax3.set_title("Segmented Leaf Region")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    if out["probs"] is not None:
        labels = [DISPLAY_NAMES.get(lbl, lbl) for lbl in out["class_names"]]
        ax4.bar(labels, out["probs"])
        ax4.set_ylim(0, 1)
        ax4.set_ylabel("Confidence")
        ax4.set_title("Prediction Confidence Scores")
        plt.setp(ax4.get_xticklabels(), rotation=15, ha="right")
    else:
        skip_reason = "Classification skipped due to rejection"
        if out["gate_leaf_prob"] is not None:
            skip_reason += f"\n(gate leaf probability={out['gate_leaf_prob']:.4f})"
        ax4.text(0.5, 0.5, skip_reason, ha="center", va="center")
        ax4.set_title("Prediction Confidence Scores")
        ax4.axis("off")

    total_ms = sum(out["timings"].values())
    fig.suptitle(
        f"Result: {out['result_text']}\nReason: {out['reason_text']}\nTotal pipeline: {total_ms:.3f} ms",
        fontsize=13
    )
    plt.tight_layout()
    plt.show()


def benchmark_prediction(image_path, n_runs=1000, warmup=10):
    """
    Runs the inference pipeline repeatedly on the same image and reports
    mean / min / max / std timing per stage (in ms).

    A single prediction's timing is noisy (OS scheduling, CPU frequency
    scaling, disk/page caching, etc.), so for reporting inference time in a
    thesis or performance evaluation, run the same prediction many times
    (100-1000+) and report the aggregate statistics below rather than one
    measurement.

    A few warm-up runs are executed first and excluded from the stats,
    since the very first call can be slower due to cold caches / lazy
    imports and would otherwise bias the results.
    """
    model_data, gate_data = _load_models()

    for _ in range(warmup):
        run_inference(image_path, model_data, gate_data)

    collected = {stage: [] for stage in STAGE_ORDER}
    totals = []
    last_out = None

    for _ in range(n_runs):
        out = run_inference(image_path, model_data, gate_data)
        last_out = out
        run_total = 0.0
        for stage in STAGE_ORDER:
            ms = out["timings"].get(stage, 0.0)
            collected[stage].append(ms)
            run_total += ms
        totals.append(run_total)

    header = f"{'Stage':<20}{'Mean':>10}{'Min':>10}{'Max':>10}{'Std':>10}"
    print(f"========== BENCHMARK ({n_runs} runs, {warmup} warm-up) ==========")
    print(header)
    print("-" * len(header))
    for stage in STAGE_ORDER:
        arr = np.array(collected[stage])
        if arr.sum() == 0:
            continue  # stage never ran for this image (e.g. no gate model)
        print(f"{stage:<20}{arr.mean():>10.3f}{arr.min():>10.3f}{arr.max():>10.3f}{arr.std():>10.3f}")
    print("-" * len(header))
    arr_total = np.array(totals)
    print(f"{'TOTAL PIPELINE':<20}{arr_total.mean():>10.3f}{arr_total.min():>10.3f}{arr_total.max():>10.3f}{arr_total.std():>10.3f}")
    print("=" * len(header))
    print(f"\nLast run -> Result: {last_out['result_text']}  |  Reason: {last_out['reason_text']}")

    return {
        "per_stage": {stage: np.array(collected[stage]) for stage in STAGE_ORDER},
        "total": arr_total,
    }


if __name__ == "__main__":
    test_image = ("test/test (4).jpg")   # replace with your image path

    # Single prediction, with visualization and a one-shot timing breakdown
    predict_and_visualize(test_image)

    # Thesis-quality benchmark: repeats the prediction and reports
    # mean/min/max/std so the numbers aren't just one noisy measurement
    benchmark_prediction(test_image, n_runs=1000, warmup=10)