import time
import json
import argparse
import os
from pathlib import Path

import board
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15


# ============================================================
# CONFIG
# ============================================================

# Anchored to sensors/calibration/ regardless of the current working
# directory this script is launched from, so it always lines up with the
# file ui_main.py reads (PH_CALIBRATION_FILE).
_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_FILE = _SCRIPT_DIR / "calibration" / "ph_calibration.json"
CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)

# Your buffer solutions
BUFFER_POINTS = [
    6.11,
    7.33,
    8.60
]

# DFRobot pH sensor signal wire connected to ADS1115 A0
ADS_CHANNEL = ads1x15.Pin.A0

# ADS1115 gain
# gain = 1 means +/- 4.096V range
ADS_GAIN = 1

# Calibration sampling settings
SAMPLE_COUNT = 30
SAMPLE_DELAY = 0.2


# ============================================================
# ADS1115 SETUP
# ============================================================

def setup_ads1115():
    i2c = board.I2C()
    ads = ADS1115(i2c)
    ads.gain = ADS_GAIN

    channel = AnalogIn(ads, ADS_CHANNEL)
    return channel


# ============================================================
# VOLTAGE READING
# ============================================================

def read_average_voltage(channel, sample_count=SAMPLE_COUNT, delay=SAMPLE_DELAY):
    readings = []

    for _ in range(sample_count):
        voltage = channel.voltage
        readings.append(voltage)
        time.sleep(delay)

    average_voltage = sum(readings) / len(readings)
    return average_voltage


# ============================================================
# CALIBRATION MODE
# ============================================================

def calibrate():
    print("\nDFRobot pH Sensor 3-Point Calibration")
    print("-------------------------------------")
    print("Buffers needed: pH 4.00, pH 6.86, pH 9.18")
    print()
    print("Reminder:")
    print("- Rinse the probe with distilled water before each buffer.")
    print("- Gently blot dry.")
    print("- Do not rub the glass bulb hard.")
    print("- Wait for the voltage to stabilize before pressing ENTER.")
    print()

    channel = setup_ads1115()

    calibration_points = []

    for buffer_ph in BUFFER_POINTS:
        print(f"\nPrepare pH {buffer_ph:.2f} buffer.")
        input(f"Place the probe in pH {buffer_ph:.2f} buffer, wait 1-2 minutes, then press ENTER...")

        print("Reading voltage...")
        voltage = read_average_voltage(channel)

        print(f"Recorded: pH {buffer_ph:.2f} = {voltage:.4f} V")

        calibration_points.append({
            "ph": buffer_ph,
            "voltage": voltage
        })

        input("Rinse the probe with distilled water, then press ENTER to continue...")

    # Sort by voltage for interpolation
    calibration_points.sort(key=lambda point: point["voltage"])

    data = {
        "calibration_points": calibration_points,
        "sample_count": SAMPLE_COUNT,
        "sample_delay": SAMPLE_DELAY
    }

    with open(CALIBRATION_FILE, "w") as file:
        json.dump(data, file, indent=4)

    print("\nCalibration complete!")
    print(f"Saved calibration file: {CALIBRATION_FILE}")
    print("\nSaved calibration points:")

    for point in calibration_points:
        print(f"Voltage: {point['voltage']:.4f} V  ->  pH {point['ph']:.2f}")


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"No calibration file found: {CALIBRATION_FILE}\n"
            "Run calibration first:\n"
            "python calibrate_dfrobot_ph.py --calibrate"
        )

    with open(CALIBRATION_FILE, "r") as file:
        data = json.load(file)

    points = data["calibration_points"]

    if len(points) < 2:
        raise ValueError("Calibration file must contain at least 2 calibration points.")

    points.sort(key=lambda point: point["voltage"])
    return points


# ============================================================
# VOLTAGE TO PH CONVERSION
# ============================================================

def interpolate_ph(voltage, points):
    """
    Converts ADS1115 voltage to pH using piecewise linear interpolation.

    With 3 buffers:
    - pH 3.99
    - pH 7.40
    - pH 8.80

    The code automatically chooses the correct segment based on voltage.
    """

    # If voltage is below the lowest recorded voltage,
    # use the first two calibration points.
    if voltage <= points[0]["voltage"]:
        p1 = points[0]
        p2 = points[1]

    # If voltage is above the highest recorded voltage,
    # use the last two calibration points.
    elif voltage >= points[-1]["voltage"]:
        p1 = points[-2]
        p2 = points[-1]

    # If voltage is between calibration points,
    # find the surrounding two points.
    else:
        p1 = points[0]
        p2 = points[1]

        for i in range(len(points) - 1):
            low = points[i]
            high = points[i + 1]

            if low["voltage"] <= voltage <= high["voltage"]:
                p1 = low
                p2 = high
                break

    v1 = p1["voltage"]
    v2 = p2["voltage"]
    ph1 = p1["ph"]
    ph2 = p2["ph"]

    if v1 == v2:
        raise ValueError("Two calibration voltages are the same. Please recalibrate.")

    slope = (ph2 - ph1) / (v2 - v1)
    intercept = ph1 - slope * v1

    ph_value = slope * voltage + intercept
    return ph_value


# ============================================================
# LIVE READING MODE
# ============================================================

def read_ph_live():
    print("\nDFRobot pH Sensor Live Reading")
    print("-----------------------------")

    points = load_calibration()
    channel = setup_ads1115()

    print("Loaded calibration points:")
    for point in points:
        print(f"Voltage: {point['voltage']:.4f} V  ->  pH {point['ph']:.2f}")

    print("\nPress CTRL+C to stop.\n")

    try:
        while True:
            voltage = read_average_voltage(channel, sample_count=10, delay=0.1)
            ph_value = interpolate_ph(voltage, points)

            print(f"Voltage: {voltage:.4f} V | pH: {ph_value:.2f}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopped.")


# ============================================================
# TEST RAW VOLTAGE MODE
# ============================================================

def read_raw_voltage():
    print("\nRaw ADS1115 Voltage Reading")
    print("---------------------------")
    print("Use this to check if the DFRobot sensor is detected.")
    print("Press CTRL+C to stop.\n")

    channel = setup_ads1115()

    try:
        while True:
            voltage = channel.voltage
            print(f"Voltage: {voltage:.4f} V")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopped.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DFRobot pH Sensor Calibration for Raspberry Pi 4 + ADS1115"
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run 3-point calibration using pH 4.00, 6.86, and 9.18 buffers"
    )

    parser.add_argument(
        "--read",
        action="store_true",
        help="Read live calibrated pH value"
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Read raw ADS1115 voltage only"
    )

    args = parser.parse_args()

    if args.calibrate:
        calibrate()

    elif args.read:
        read_ph_live()

    elif args.raw:
        read_raw_voltage()

    else:
        print("Choose a mode:")
        print("  python calibrate_dfrobot_ph.py --raw")
        print("  python calibrate_dfrobot_ph.py --calibrate")
        print("  python calibrate_dfrobot_ph.py --read")


if __name__ == "__main__":
    main()