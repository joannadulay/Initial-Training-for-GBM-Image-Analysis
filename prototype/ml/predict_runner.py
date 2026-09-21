import argparse
import json
import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GBM-CV prediction and emit JSON.")
    parser.add_argument("image_path", help="Path to the captured image")
    args = parser.parse_args()

    try:
        from predict_core import get_prediction

        result = get_prediction(args.image_path, return_details=True)

        if len(result) == 3:
            label, probabilities, details = result
        else:
            label, probabilities = result
            details = {}

        print(json.dumps({
            "label": label,
            "probabilities": probabilities,
            "details": details,
        }))
        return 0

    except Exception as exc:
        print(str(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
