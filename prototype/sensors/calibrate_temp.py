import json
import os
import time
from pathlib import Path

import fullcomby

_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CAL_FILE = _SCRIPT_DIR / "calibration" / "temp_calibration.json"
CAL_FILE.parent.mkdir(parents=True, exist_ok=True)


def average_raw_temp(samples=30, delay=0.5):
    readings = []

    for i in range(samples):
        temp_c = fullcomby.read_water_temperature_c()
        readings.append(temp_c)
        print(f"Reading {i + 1}/{samples}: {temp_c:.3f} °C")
        time.sleep(delay)

    return sum(readings) / len(readings)


def get_calibration_point(point_name):
    print(f"\n=== {point_name} ===")
    print("Place the DS18B20 and the reference thermometer in the SAME water.")
    print("Wait until both readings are stable.")
    input("Press ENTER when stable...")

    raw_temp = average_raw_temp()

    print(f"\nAverage DS18B20 raw temp: {raw_temp:.3f} °C")
    reference_temp = float(input("Enter reference thermometer temp in °C: "))

    return raw_temp, reference_temp


def main():
    print("\nDS18B20 2-Point Temperature Calibration")
    print("--------------------------------------")
    print("Recommended:")
    print("Point 1 = normal/room temperature water")
    print("Point 2 = warmer water")
    print("Do not use boiling water.\n")

    raw1, ref1 = get_calibration_point("POINT 1: Lower/normal temperature")

    print("\nNow prepare the higher/warm temperature water.")
    input("Press ENTER when ready for Point 2...")

    raw2, ref2 = get_calibration_point("POINT 2: Higher/warm temperature")

    if raw1 == raw2:
        raise ValueError("Raw temperatures are the same. Use two different water temperatures.")

    slope = (ref2 - ref1) / (raw2 - raw1)
    intercept = ref1 - slope * raw1

    data = {
        "type": "two_point",
        "point_1": {
            "raw_temp_c": round(raw1, 4),
            "reference_temp_c": round(ref1, 4)
        },
        "point_2": {
            "raw_temp_c": round(raw2, 4),
            "reference_temp_c": round(ref2, 4)
        },
        "slope": round(slope, 8),
        "intercept": round(intercept, 8)
    }

    with open(CAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("\nCalibration complete!")
    print(f"Point 1: raw {raw1:.3f} °C -> reference {ref1:.3f} °C")
    print(f"Point 2: raw {raw2:.3f} °C -> reference {ref2:.3f} °C")
    print(f"Slope: {slope:.8f}")
    print(f"Intercept: {intercept:.8f}")
    print(f"Saved to: {CAL_FILE}")

    test_raw = fullcomby.read_water_temperature_c()
    test_calibrated = slope * test_raw + intercept

    print("\nCurrent test reading:")
    print(f"Raw DS18B20:        {test_raw:.3f} °C")
    print(f"Calibrated DS18B20: {test_calibrated:.3f} °C")


if __name__ == "__main__":
    main()