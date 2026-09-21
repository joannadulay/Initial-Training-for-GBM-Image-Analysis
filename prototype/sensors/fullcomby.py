import glob
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime

import adafruit_ads1x15.ads1115 as ads1115
import adafruit_bh1750
import board
import busio
from adafruit_ads1x15.analog_in import AnalogIn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../AquaponicsSystem/sensors
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../AquaponicsSystem

MODE = "log"
DB_PATH = os.path.join(PROJECT_ROOT, "database", "sensor_readings.db")
CSV_PATH = os.path.join(PROJECT_ROOT, "database", "sensor_readings.csv")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "captured_images")
READ_INTERVAL_SECONDS = 60
# Simple 2-point pH fallback calibration used only by safe_read_sensors()
# when the richer multi-point UI calibration (calibration/ph_calibration.json,
# produced by calibrate_dfrobot_ph.py) isn't available. Kept as a separate
# file/format on purpose since the two calibration schemes aren't compatible.
CAL_FILE = os.path.join(BASE_DIR, "calibration", "ph_cal.json")
PH_ADC_CHANNEL = 0
DS18B20_BASE_DIR = "/sys/bus/w1/devices"
CAMERA_CMD = "rpicam-still"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)

EXPECTED_COLS = {
    "timestamp",
    "light_lux",
    "water_temp_c",
    "ph_level",
    "ph_voltage",
    "image_path",
    "ml_prediction",
    "prob_healthy",
    "prob_diseased",
    "prob_discolored",
}


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            light_lux REAL,
            water_temp_c REAL,
            ph_level REAL,
            ph_voltage REAL,
            image_path TEXT,
            ml_prediction TEXT,
            prob_healthy REAL,
            prob_diseased REAL,
            prob_discolored REAL
        )
        """
    )
    conn.commit()
    cur.execute("PRAGMA table_info(sensor_log)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()

    missing = EXPECTED_COLS - cols
    if missing:
        raise RuntimeError(
            "Your existing sensor_readings.db has an old schema.\n"
            f"Missing columns: {sorted(missing)}\n\n"
            "Delete or migrate the old DB before running again."
        )


def insert_row(ts, lux, temp_c, ph, ph_v, img_path, pred, p_health, p_dis, p_disc) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sensor_log (
            timestamp, light_lux, water_temp_c, ph_level, ph_voltage,
            image_path, ml_prediction, prob_healthy, prob_diseased, prob_discolored
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, lux, temp_c, ph, ph_v, img_path, pred, p_health, p_dis, p_disc),
    )
    conn.commit()
    conn.close()


def append_csv(ts, lux, temp_c, ph, ph_v, img_path, pred, p_health, p_dis, p_disc) -> None:
    exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        if not exists:
            f.write(
                "timestamp,light_lux,water_temp_c,ph_level,ph_voltage,image_path,ml_prediction,prob_healthy,prob_diseased,prob_discolored\n"
            )
        f.write(f"{ts},{lux},{temp_c},{ph},{ph_v},{img_path},{pred},{p_health},{p_dis},{p_disc}\n")


def capture_image() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(IMAGE_DIR, f"img_{ts}.jpg")
    cmd = [CAMERA_CMD, "-o", path, "--nopreview", "-t", "500", "-q", "90"]
    subprocess.run(cmd, check=True)

    if not os.path.exists(path) or os.path.getsize(path) < 10_000:
        try:
            os.remove(path)
        except Exception:
            pass
        raise RuntimeError("Camera capture failed (image missing or too small).")

    return path


def read_water_temperature_c() -> float:
    device_folders = glob.glob(os.path.join(DS18B20_BASE_DIR, "28-*"))
    if not device_folders:
        raise RuntimeError("No DS18B20 found. Enable 1-Wire and check wiring.")

    device_file = os.path.join(device_folders[0], "w1_slave")

    def _raw():
        with open(device_file, "r", encoding="utf-8") as f:
            return f.readlines()

    lines = _raw()
    retries = 6
    while retries > 0 and not lines[0].strip().endswith("YES"):
        time.sleep(0.2)
        lines = _raw()
        retries -= 1

    if not lines[0].strip().endswith("YES"):
        raise RuntimeError("DS18B20 CRC check failed.")

    pos = lines[1].find("t=")
    if pos == -1:
        raise RuntimeError("DS18B20 missing t=")

    return float(lines[1][pos + 2:]) / 1000.0


def avg_voltage(chan, samples=30, delay=0.1) -> float:
    vals = []
    for _ in range(samples):
        vals.append(float(chan.voltage))
        time.sleep(delay)
    return sum(vals) / len(vals)


def run_ph_calibration(ph_chan) -> None:
    print("pH Calibration Helper (ADS1115 A0)")
    print("1) Put probe in pH 7.00 buffer, wait stable, then press ENTER.")
    input()
    v7 = avg_voltage(ph_chan)
    print(f"Saved pH 7 voltage: {v7:.3f} V\n")

    print("2) Rinse probe. Put probe in pH 4.00 buffer, wait stable, then press ENTER.")
    input()
    v4 = avg_voltage(ph_chan)
    print(f"Saved pH 4 voltage: {v4:.3f} V\n")

    cal = {"V_PH7": v7, "V_PH4": v4, "PH7": 7.0, "PH4": 4.0}
    with open(CAL_FILE, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)

    print(f"Calibration written to {CAL_FILE}")


def load_ph_cal_optional():
    if not os.path.isfile(CAL_FILE):
        return None
    try:
        with open(CAL_FILE, "r", encoding="utf-8") as f:
            cal = json.load(f)
        for key in ("V_PH7", "V_PH4", "PH7", "PH4"):
            if key not in cal:
                return None
        return cal
    except Exception:
        return None


def voltage_to_ph(v: float, cal) -> float:
    v7, v4 = float(cal["V_PH7"]), float(cal["V_PH4"])
    ph7, ph4 = float(cal["PH7"]), float(cal["PH4"])
    m = (ph7 - ph4) / (v7 - v4)
    b = ph7 - m * v7
    ph = m * v + b
    return max(0.0, min(14.0, ph))


def voltage_to_ph_fallback(v: float) -> float:
    return max(0.0, min(14.0, 3.5 * v))


def init_hardware():
    i2c = busio.I2C(board.SCL, board.SDA)
    light = adafruit_bh1750.BH1750(i2c)
    ads = ads1115.ADS1115(i2c)
    ads.gain = 1
    ph_chan = AnalogIn(ads, PH_ADC_CHANNEL)
    return light, ph_chan


def safe_read_sensors(light_sensor, ph_chan, cal=None):
    values = {
        "lux": None,
        "temp_c": None,
        "ph": None,
        "ph_v": None,
        "errors": [],
    }

    if light_sensor is not None:
        try:
            values["lux"] = float(light_sensor.lux)
        except Exception as exc:
            values["errors"].append(f"BH1750: {exc}")

    try:
        values["temp_c"] = read_water_temperature_c()
    except Exception as exc:
        values["errors"].append(f"DS18B20: {exc}")

    if ph_chan is not None:
        try:
            # Calibration points were built from a 30-sample average
            # (see avg_voltage / ph_calibration.json's sample_count).
            # A single-shot chan.voltage read is noisy enough on a
            # high-impedance pH probe to swing outside the calibrated
            # voltage band and get clamped to 0 or 14. Average several
            # quick samples here so live readings are consistent with
            # how the probe was calibrated.
            values["ph_v"] = avg_voltage(ph_chan, samples=10, delay=0.02)
            values["ph"] = voltage_to_ph(values["ph_v"], cal) if cal else voltage_to_ph_fallback(values["ph_v"])
        except Exception as exc:
            values["errors"].append(f"pH: {exc}")

    return values
