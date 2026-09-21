import json
import os
import subprocess
import sys
import time
import tkinter as tk
import threading
from collections import deque
from datetime import datetime
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from sensors import fullcomby

import csv_utils
from theme import (
    COLORS,
    TEMP_LOW_C,
    TEMP_HIGH_C,
    PH_LOW,
    PH_HIGH,
    LIGHT_LOW_LUX,
    make_card,
    make_button,
    status_color,
)
from history_window import open_history_window
from details_window import open_recommendations_window

try:
    from picamera2 import Picamera2
except Exception as exc:
    Picamera2 = None
    print(f"Picamera2 import warning: {exc}")

# ui_main.py now lives in AquaponicsSystem/ui/, so the project root is one
# level up from this file's own directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICT_RUNNER = os.path.join(BASE_DIR, "ml", "predict_runner.py")
DEFAULT_ML_PYTHON = os.path.join(BASE_DIR, ".venv_ml", "bin", "python")
ANALYZE_TIMEOUT_SECONDS = 90

# pH calibration file created by calibrate_dfrobot_ph.py
PH_CALIBRATION_FILE = os.path.join(BASE_DIR, "sensors", "calibration", "ph_calibration.json")

# Temperature calibration file created by calibrate_temp.py
TEMP_CALIBRATION_FILE = os.path.join(BASE_DIR, "sensors", "calibration", "temp_calibration.json")

# Temperature calibration safety limits.
TEMP_CAL_MIN_RAW_GAP_C = 2.0       # two calibration points must be at least 2°C apart
TEMP_CAL_MIN_SLOPE = 0.70
TEMP_CAL_MAX_SLOPE = 1.30
TEMP_CAL_MIN_OUTPUT_C = 0.0
TEMP_CAL_MAX_OUTPUT_C = 50.0


class AquaponicsUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Kangkong Aquaponics Monitor")
        self.root.geometry("800x480")
        self.root.configure(bg=COLORS["bg"])

        self.preview_size = (480, 360)
        self.display_size = (540, 390)
        self.preview_interval_ms = 30  # Sped up the camera refresh rate

        self.camera_active = True
        self.analysis_running = False
        self.picam2 = None
        self.current_frame_rgb = None
        self.captured_image_path = None
        self.last_prediction_payload = None
        self.last_detail_image_path = None
        self.last_prediction_label = None
        self.last_prediction_reason = None

        # Holds the results of the most recent analysis until the user
        # decides to save them (or discards them via Retake).
        self.pending_analysis = None

        self.last_sensor_values = {
            "lux": None,
            "temp_c": None,
            "ph": None,
            "ph_v": None,
            "errors": [],
        }

        # A single noisy ADC sample can swing the raw pH reading by several
        # units even when the probe hasn't moved (this is what caused the
        # displayed pH to jump around, e.g. 8 -> 14, in still water). These
        # rolling buffers feed a median filter (_smooth_sensor_value) that
        # damps out single-sample spikes while still tracking real changes.
        self.ph_reading_history = deque(maxlen=5)
        self.temp_reading_history = deque(maxlen=5)
        self.lux_reading_history = deque(maxlen=5)

        self.ml_python = self._resolve_ml_python()
        self.ui_ph_cal_points = self._load_ui_ph_calibration()
        self.ui_temp_calibration = self._load_ui_temp_calibration()

        try:
            fullcomby.init_db()
        except Exception as exc:
            print(f"DB init warning: {exc}")

        try:
            self.light_sensor, self.ph_chan = fullcomby.init_hardware()
            self.cal = fullcomby.load_ph_cal_optional()
            print("Hardware initialized successfully.")
        except Exception as exc:
            print(f"Hardware init warning: {exc}")
            self.light_sensor, self.ph_chan, self.cal = None, None, None

        self._init_camera()
        self.setup_ui()
        self.update_live_feed()
        self.update_sensor_readings()

    def _resolve_ml_python(self) -> str:
        env_python = os.environ.get("GBM_PYTHON")
        candidates = [env_python, DEFAULT_ML_PYTHON, sys.executable]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return sys.executable

    def _init_camera(self) -> None:
        if Picamera2 is None:
            print("Picamera2 is not available.")
            return

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": self.preview_size, "format": "RGB888"},
                buffer_count=2,
                queue=False,
                controls={
                    "FrameDurationLimits": (50000, 50000),
                    "AwbEnable": True,
                    "AeEnable": True,
                    "Brightness": 0.0,
                    "Contrast": 1.0,
                    "Saturation": 1.0,
                    "Sharpness": 1.0,
                },
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.6)
            print("Picamera2 started successfully.")
        except Exception as exc:
            print(f"Camera init warning: {exc}")
            self.picam2 = None

    def setup_ui(self) -> None:
        # ---- Bottom action bar (full width) ------------------------------
        # Packed first (side=BOTTOM) so it reserves its own strip at the
        # bottom of the window; the camera preview + side panel below then
        # split whatever vertical space is left above it.
        self.actions_bar = tk.Frame(self.root, bg=COLORS["bg"])
        self.actions_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(6, 10))
        for col in range(7):
            self.actions_bar.grid_columnconfigure(col, weight=1, uniform="actions")

        unified_btn_font = ("Helvetica", 10, "bold")
        label_font = ("Helvetica", 10)
        small_font = ("Helvetica", 9)

        self.btn_capture = make_button(
            self.actions_bar, "CAPTURE", COLORS["blue"], COLORS["blue_hover"],
            self.capture_frame, font=unified_btn_font,
        )
        self.btn_capture.grid(row=0, column=0, sticky="ew", padx=3)

        self.btn_analyze = make_button(
            self.actions_bar, "ANALYZE", COLORS["orange"], COLORS["orange_hover"],
            self.analyze_data, font=unified_btn_font, state=tk.DISABLED,
        )
        self.btn_analyze.grid(row=0, column=1, sticky="ew", padx=3)

        self.btn_save = make_button(
            self.actions_bar, "SAVE", COLORS["green"], COLORS["green_hover"],
            self.save_analysis, font=unified_btn_font, state=tk.DISABLED,
        )
        self.btn_save.grid(row=0, column=2, sticky="ew", padx=3)

        self.btn_retake = make_button(
            self.actions_bar, "RETAKE", COLORS["gray"], COLORS["gray_hover"],
            self.retake_photo, font=unified_btn_font, state=tk.DISABLED,
        )
        self.btn_retake.grid(row=0, column=3, sticky="ew", padx=3)

        self.btn_more_details = make_button(
            self.actions_bar, "DETAILS", COLORS["teal"], COLORS["teal_hover"],
            self.show_more_details, font=unified_btn_font, state=tk.DISABLED,
        )
        self.btn_more_details.grid(row=0, column=4, sticky="ew", padx=3)

        # Only enabled once MORE DETAILS has swapped the preview over to the
        # segmented/annotated view - lets the user flip back to the plain
        # captured photo without needing to Retake.
        self.btn_back = make_button(
            self.actions_bar, "BACK", COLORS["blue"], COLORS["blue_hover"],
            self.show_captured_photo, font=unified_btn_font, state=tk.DISABLED,
        )
        self.btn_back.grid(row=0, column=5, sticky="ew", padx=3)

        self.btn_view_data = make_button(
            self.actions_bar, "HISTORY", COLORS["purple"], COLORS["purple_hover"],
            self.view_csv_data, font=unified_btn_font,
        )
        self.btn_view_data.grid(row=0, column=6, sticky="ew", padx=3)

        # ---- Camera / image preview (left) ---------------------------------
        # No fixed width/height here (unlike before) -- it expands to fill
        # whatever space is left once the side panel and bottom action bar
        # have claimed theirs, which maximizes the preview area.
        self.left_frame = tk.Frame(
            self.root,
            bg="black",
            highlightthickness=2,
            highlightbackground=COLORS["border"],
        )
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 8), pady=10)

        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # ---- Side panel (right): status + live readings only ---------------
        # The action buttons now live in the bottom bar, so this panel only
        # needs to hold the two info cards, letting it sit narrower and
        # further right -- which is what frees up the extra camera width.
        self.right_frame = tk.Frame(self.root, width=230, bg=COLORS["bg"])
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        self.right_frame.pack_propagate(False)

        # --- Status card ---
        status_card = make_card(self.right_frame)
        status_card.pack(fill=tk.X, pady=(0, 8))
        status_inner = tk.Frame(status_card, bg=COLORS["bg_card"])
        status_inner.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            status_inner, text="STATUS", font=("Helvetica", 8, "bold"),
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
        ).pack(anchor="w", pady=(0, 4))

        # height=5 gives room for a longer wrapped reason (e.g. "Shadow near
        # stem was excluded from analysis.") without getting cut off.
        self.lbl_status = tk.Label(
            status_inner,
            text="Starting...",
            font=("Helvetica", 10, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["text"],
            wraplength=200,
            justify="left",
            height=5,
            anchor="nw",
        )
        self.lbl_status.pack(anchor="w", fill=tk.X)

        # --- Sensor readings card ---
        sensors_card = make_card(self.right_frame)
        sensors_card.pack(fill=tk.X, pady=(0, 8))
        sensors_inner = tk.Frame(sensors_card, bg=COLORS["bg_card"])
        sensors_inner.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(
            sensors_inner, text="LIVE READINGS", font=("Helvetica", 8, "bold"),
            bg=COLORS["bg_card"], fg=COLORS["text_muted"],
        ).pack(anchor="w", pady=(0, 3))

        self.lbl_light = tk.Label(
            sensors_inner, text="Light: -- lux", font=label_font,
            bg=COLORS["bg_card"], fg=COLORS["text"], anchor="w",
            wraplength=200, justify="left",
        )
        self.lbl_light.pack(fill=tk.X, pady=1)

        self.lbl_temp = tk.Label(
            sensors_inner, text="Water Temp: -- \u00b0C", font=label_font,
            bg=COLORS["bg_card"], fg=COLORS["text"], anchor="w",
            wraplength=200, justify="left",
        )
        self.lbl_temp.pack(fill=tk.X, pady=1)

        self.lbl_ph = tk.Label(
            sensors_inner, text="pH Level: --", font=label_font,
            bg=COLORS["bg_card"], fg=COLORS["text"], anchor="w",
            wraplength=200, justify="left",
        )
        self.lbl_ph.pack(fill=tk.X, pady=1)

        if self.picam2 is not None:
            self.lbl_status.config(text="Live camera ready", fg=COLORS["text"])
        else:
            self.lbl_status.config(text="Camera not available", fg=COLORS["red"])

    def _load_ui_ph_calibration(self):
        if not os.path.exists(PH_CALIBRATION_FILE):
            print(f"pH calibration file not found: {PH_CALIBRATION_FILE}")
            return None

        try:
            with open(PH_CALIBRATION_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            points = data.get("calibration_points", [])
            cleaned_points = []

            for point in points:
                cleaned_points.append({
                    "ph": float(point["ph"]),
                    "voltage": float(point["voltage"]),
                })

            if len(cleaned_points) < 2:
                print("pH calibration warning: at least 2 calibration points are required.")
                return None

            cleaned_points.sort(key=lambda item: item["voltage"])
            print(f"Loaded UI pH calibration from {PH_CALIBRATION_FILE}")
            return cleaned_points

        except Exception as exc:
            print(f"pH calibration load warning: {exc}")
            return None

    def _interpolate_ph_from_voltage(self, voltage):
        if voltage is None or not self.ui_ph_cal_points:
            return None

        try:
            voltage = float(voltage)
        except Exception:
            return None

        points = self.ui_ph_cal_points

        if voltage <= points[0]["voltage"]:
            p1 = points[0]
            p2 = points[1]
        elif voltage >= points[-1]["voltage"]:
            p1 = points[-2]
            p2 = points[-1]
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
            return None

        slope = (ph2 - ph1) / (v2 - v1)
        intercept = ph1 - slope * v1
        ph_value = slope * voltage + intercept

        return max(0.0, min(14.0, ph_value))

    def _apply_ui_ph_calibration(self, sensor_values):
        if not isinstance(sensor_values, dict):
            return sensor_values

        calibrated_ph = self._interpolate_ph_from_voltage(sensor_values.get("ph_v"))
        if calibrated_ph is not None:
            sensor_values = dict(sensor_values)
            sensor_values["ph"] = calibrated_ph

        return sensor_values

    def _load_ui_temp_calibration(self):
        if not os.path.exists(TEMP_CALIBRATION_FILE):
            print(f"Temperature calibration file not found: {TEMP_CALIBRATION_FILE}")
            return None

        try:
            with open(TEMP_CALIBRATION_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            if data.get("type") == "two_point":
                slope = float(data.get("slope", 1.0))
                intercept = float(data.get("intercept", 0.0))

                point_1 = data.get("point_1", {}) or {}
                point_2 = data.get("point_2", {}) or {}
                raw1 = point_1.get("raw_temp_c")
                raw2 = point_2.get("raw_temp_c")

                raw_min = None
                raw_max = None
                if raw1 is not None and raw2 is not None:
                    raw1 = float(raw1)
                    raw2 = float(raw2)
                    raw_min = min(raw1, raw2)
                    raw_max = max(raw1, raw2)
                    raw_gap = abs(raw2 - raw1)

                    if raw_gap < TEMP_CAL_MIN_RAW_GAP_C:
                        print(
                            "Temperature calibration ignored: "
                            f"calibration points are too close ({raw_gap:.2f} °C apart). "
                            "Recalibrate using two water samples at least 2°C apart."
                        )
                        return None

                if not (TEMP_CAL_MIN_SLOPE <= slope <= TEMP_CAL_MAX_SLOPE):
                    print(
                        "Temperature calibration ignored: "
                        f"unrealistic slope {slope:.6f}. "
                        "Recalibrate the DS18B20 temperature sensor."
                    )
                    return None

                print(f"Loaded UI temperature 2-point calibration from {TEMP_CALIBRATION_FILE}")
                return {
                    "type": "two_point",
                    "slope": slope,
                    "intercept": intercept,
                    "raw_min": raw_min,
                    "raw_max": raw_max,
                }

            if "temp_offset_c" in data:
                offset = float(data.get("temp_offset_c", 0.0))
                if abs(offset) > 5.0:
                    print(
                        "Temperature offset calibration ignored: "
                        f"offset {offset:+.2f} °C is too large."
                    )
                    return None

                print(f"Loaded UI temperature offset calibration from {TEMP_CALIBRATION_FILE}")
                return {
                    "type": "offset",
                    "offset": offset,
                }

            print("Temperature calibration warning: unsupported temp_calibration.json format.")
            return None

        except Exception as exc:
            print(f"Temperature calibration load warning: {exc}")
            return None

    def _apply_ui_temp_calibration_value(self, raw_temp_c):
        if raw_temp_c is None:
            return None

        try:
            raw_temp_c = float(raw_temp_c)
        except Exception:
            return None

        cal = self.ui_temp_calibration

        if not cal:
            return raw_temp_c

        calibrated_temp_c = raw_temp_c

        if cal.get("type") == "two_point":
            slope = float(cal.get("slope", 1.0))
            intercept = float(cal.get("intercept", 0.0))
            calibrated_temp_c = slope * raw_temp_c + intercept

            raw_min = cal.get("raw_min")
            raw_max = cal.get("raw_max")
            if raw_min is not None and raw_max is not None:
                margin = 5.0
                if raw_temp_c < raw_min - margin or raw_temp_c > raw_max + margin:
                    print(
                        "Temperature note: current raw reading is outside the calibrated range; "
                        "using calibrated extrapolation carefully. "
                        f"Raw={raw_temp_c:.2f} °C, calibrated range={raw_min:.2f}-{raw_max:.2f} °C"
                    )

        elif cal.get("type") == "offset":
            offset = float(cal.get("offset", 0.0))
            calibrated_temp_c = raw_temp_c + offset

        if not (TEMP_CAL_MIN_OUTPUT_C <= calibrated_temp_c <= TEMP_CAL_MAX_OUTPUT_C):
            print(
                "Temperature calibration result ignored: "
                f"raw={raw_temp_c:.2f} °C, calibrated={calibrated_temp_c:.2f} °C. "
                "Using raw DS18B20 value instead."
            )
            return raw_temp_c

        return calibrated_temp_c

    def _apply_ui_temp_calibration(self, sensor_values):
        if not isinstance(sensor_values, dict):
            return sensor_values

        sensor_values = dict(sensor_values)

        try:
            raw_temp_c = fullcomby.read_water_temperature_c()
            calibrated_temp_c = self._apply_ui_temp_calibration_value(raw_temp_c)
            if calibrated_temp_c is not None:
                sensor_values["raw_temp_c"] = raw_temp_c
                sensor_values["temp_c"] = calibrated_temp_c
        except Exception as exc:
            sensor_values.setdefault("errors", [])
            sensor_values["errors"].append(f"DS18B20: {exc}")

        return sensor_values

    def _apply_ui_sensor_calibrations(self, sensor_values):
        sensor_values = self._apply_ui_ph_calibration(sensor_values)
        sensor_values = self._apply_ui_temp_calibration(sensor_values)
        return sensor_values

    def _smooth_sensor_value(self, history: deque, value):
        """Rolling-median filter over the last few readings.

        A single noisy ADC sample can swing the raw pH (or temp/lux) value
        by a lot even in perfectly still water. Taking the median of a
        short rolling window removes that kind of single-sample spike
        while still following genuine changes within a second or two.
        """
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        history.append(value)
        ordered = sorted(history)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _normalize_preview_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

    def _show_rgb_on_label(self, frame_rgb: np.ndarray) -> None:
        display_rgb = cv2.resize(frame_rgb, self.display_size, interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(display_rgb, mode="RGB")
        imgtk = ImageTk.PhotoImage(image=pil_img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_live_feed(self) -> None:
        if self.camera_active and self.picam2 is not None and not self.analysis_running:
            try:
                frame_rgb = self.picam2.capture_array("main")
                frame_rgb = self._normalize_preview_rgb(frame_rgb)
                self.current_frame_rgb = frame_rgb
                self._show_rgb_on_label(frame_rgb)
            except Exception as exc:
                self.lbl_status.config(text="Camera read failed", fg=COLORS["red"])
                print(f"Camera read error: {exc}")

        self.root.after(self.preview_interval_ms, self.update_live_feed)

    def update_sensor_readings(self) -> None:
        def fetch_and_update():
            try:
                sensor_values = fullcomby.safe_read_sensors(self.light_sensor, self.ph_chan, self.cal)
                sensor_values = self._apply_ui_sensor_calibrations(sensor_values)

                def update_ui():
                    lux = self._smooth_sensor_value(self.lux_reading_history, sensor_values["lux"])
                    temp_c = self._smooth_sensor_value(self.temp_reading_history, sensor_values["temp_c"])
                    ph = self._smooth_sensor_value(self.ph_reading_history, sensor_values["ph"])

                    self.last_sensor_values = dict(sensor_values)
                    self.last_sensor_values["lux"] = lux
                    self.last_sensor_values["temp_c"] = temp_c
                    self.last_sensor_values["ph"] = ph

                    self.lbl_light.config(
                        text=f"Light: {lux:.2f} lux" if lux is not None else "Light: -- lux",
                        fg=status_color(lux, LIGHT_LOW_LUX, float("inf")),
                    )
                    self.lbl_temp.config(
                        text=f"Water Temp: {temp_c:.2f} \u00b0C" if temp_c is not None else "Water Temp: -- \u00b0C",
                        fg=status_color(temp_c, TEMP_LOW_C, TEMP_HIGH_C),
                    )
                    self.lbl_ph.config(
                        text=f"pH Level: {ph:.2f}" if ph is not None else "pH Level: --",
                        fg=status_color(ph, PH_LOW, PH_HIGH),
                    )

                    self.root.after(1000, self.update_sensor_readings)

                self.root.after(0, update_ui)

            except Exception as exc:
                print(f"Live sensor update error: {exc}")
                self.root.after(1000, self.update_sensor_readings)

        threading.Thread(target=fetch_and_update, daemon=True).start()

    def capture_frame(self) -> None:
        if self.current_frame_rgb is None:
            messagebox.showwarning("Camera", "No camera frame available yet.")
            return

        os.makedirs(fullcomby.IMAGE_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.captured_image_path = os.path.join(fullcomby.IMAGE_DIR, f"ui_capture_{ts}.jpg")

        save_bgr = cv2.cvtColor(self.current_frame_rgb, cv2.COLOR_RGB2BGR)
        saved = cv2.imwrite(self.captured_image_path, save_bgr)
        if not saved:
            messagebox.showerror("Save Error", "Failed to save captured image.")
            return

        self.camera_active = False
        self._show_rgb_on_label(self.current_frame_rgb)

        self.btn_capture.config(state=tk.DISABLED, bg=COLORS["gray"])
        self.btn_analyze.config(state=tk.NORMAL, bg=COLORS["orange"])
        self.btn_retake.config(state=tk.NORMAL, bg=COLORS["gray"])
        self.btn_back.config(state=tk.DISABLED, bg=COLORS["gray"])

        self.lbl_status.config(text="Image captured.\nAnalyze or Retake.", fg=COLORS["yellow"])

    def retake_photo(self) -> None:
        self.captured_image_path = None
        self.camera_active = True

        self.btn_capture.config(state=tk.NORMAL, bg=COLORS["blue"])
        self.btn_analyze.config(state=tk.DISABLED, bg=COLORS["gray"])
        self.btn_retake.config(state=tk.DISABLED, bg=COLORS["gray"])
        self._reset_save_button()
        self.last_prediction_payload = None
        self.last_detail_image_path = None
        self.last_prediction_label = None
        self.last_prediction_reason = None
        self.pending_analysis = None
        self.btn_more_details.config(state=tk.DISABLED, bg=COLORS["gray"])
        self.btn_back.config(state=tk.DISABLED, bg=COLORS["gray"])

        if self.picam2 is not None:
            self.lbl_status.config(text="Live camera ready", fg=COLORS["text"])
        else:
            self.lbl_status.config(text="Camera not available", fg=COLORS["red"])

    def _reset_save_button(self) -> None:
        self.btn_save.config(state=tk.DISABLED, text="SAVE", bg=COLORS["gray"])

    def _prediction_text_color(self, label: str) -> str:
        label_text = str(label or "").strip().lower()

        if "no kangkong" in label_text or "not kangkong" in label_text:
            return COLORS["red"]
        if "healthy" in label_text:
            return COLORS["green"]
        if "discolored" in label_text or "discolour" in label_text:
            return COLORS["yellow"]
        if "diseased" in label_text or "disease" in label_text:
            return COLORS["orange"]

        return COLORS["text"]

    # ------------------------------------------------------------------
    # History window (Search / Filter / View Image / Delete / Export /
    # Refresh all live in history_window.py; this just opens it).
    # ------------------------------------------------------------------
    def view_csv_data(self) -> None:
        csv_path = fullcomby.CSV_PATH
        if not os.path.exists(csv_path):
            messagebox.showinfo("No Data", "No CSV data found yet. Please analyze an image first.")
            return
        open_history_window(self.root, csv_path, fullcomby.IMAGE_DIR, BASE_DIR)

    def _resolve_possible_path(self, value):
        if not value:
            return None

        if not isinstance(value, str):
            return None

        value = value.strip()
        if not value:
            return None

        candidates = [value]
        if not os.path.isabs(value):
            candidates.append(os.path.join(BASE_DIR, value))
            candidates.append(os.path.join(fullcomby.IMAGE_DIR, value))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    def _find_detail_image_path(self, payload: dict):
        if not isinstance(payload, dict):
            return None

        possible_keys = [
            "details_image_path",
            "detail_image_path",
            "analysis_image_path",
            "visualization_path",
            "visualisation_path",
            "visual_path",
            "debug_image_path",
            "output_image_path",
            "result_image_path",
            "figure_path",
            "plot_path",
        ]

        for key in possible_keys:
            path = self._resolve_possible_path(payload.get(key))
            if path:
                return path

        for nested_key in ["details", "debug", "outputs", "files"]:
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                for key in possible_keys:
                    path = self._resolve_possible_path(nested.get(key))
                    if path:
                        return path

        return None

    def _get_payload_value(self, payload: dict, keys, default="--"):
        if not isinstance(payload, dict):
            return default
        for key in keys:
            if key in payload and payload[key] not in [None, ""]:
                return payload[key]
        return default

    def _show_pil_on_label(self, pil_img: Image.Image) -> None:
        canvas_w, canvas_h = self.display_size
        pil_img = pil_img.convert("RGB")
        pil_img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)

        canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["bg"])
        x = (canvas_w - pil_img.width) // 2
        y = (canvas_h - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))

        imgtk = ImageTk.PhotoImage(image=canvas)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _show_image_file_on_label(self, image_path: str) -> bool:
        try:
            pil_img = Image.open(image_path)
            self._show_pil_on_label(pil_img)
            return True
        except Exception as exc:
            print(f"Failed to display detail image: {exc}")
            return False

    def _build_fallback_details_image(self) -> Image.Image:
        from PIL import ImageDraw, ImageFont

        payload = self.last_prediction_payload or {}
        probabilities = payload.get("probabilities", payload.get("probs", {})) if isinstance(payload, dict) else {}

        # predict_core.py nests leaf_ratio/component_count/reason inside
        # payload["details"], not at the top level. Merge the two (top-level
        # wins on conflicts) so lookups below find them either way.
        nested_details = payload.get("details") if isinstance(payload, dict) else None
        nested_details = nested_details if isinstance(nested_details, dict) else {}
        lookup = {**nested_details, **payload} if isinstance(payload, dict) else nested_details

        label = self.last_prediction_label or self._get_payload_value(lookup, ["label", "prediction"], "--")
        reason = self.last_prediction_reason or self._get_payload_value(lookup, ["reason", "message"], "No extra reason returned by predict_runner.py")
        leaf_ratio = self._get_payload_value(lookup, ["leaf_ratio", "leafRatio", "leaf_area_ratio"], "--")
        components = self._get_payload_value(lookup, ["components", "component_count", "leaf_components", "scene_component_count"], "--")
        confidence = self._get_payload_value(lookup, ["confidence", "max_probability", "max_confidence"], "--")

        accent = self._prediction_text_color(label)

        w, h = self.display_size
        img = Image.new("RGB", (w, h), COLORS["bg"])
        draw = ImageDraw.Draw(img)

        def rrect(xy, radius, **kw):
            try:
                draw.rounded_rectangle(xy, radius=radius, **kw)
            except Exception:
                draw.rectangle(xy, **kw)

        try:
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
            font_text = ImageFont.truetype("DejaVuSans.ttf", 13)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
        except Exception:
            font_title = font_bold = font_text = font_small = ImageFont.load_default()

        # Header bar
        rrect((14, 12, w - 14, 46), 10, fill=COLORS["bg_card"])
        draw.text((26, 20), "Detailed GBM Analysis", fill=COLORS["text"], font=font_title)

        # Result badge
        badge_text = str(label)
        badge_w = draw.textlength(badge_text, font=font_bold) + 24
        rrect((w - 14 - badge_w, 58, w - 14, 84), 12, fill=accent)
        draw.text((w - 14 - badge_w + 12, 63), badge_text, fill="#10151f", font=font_bold)

        # Info card (reason / metrics)
        info_top = 58
        info_bottom = 178
        rrect((14, info_top, w - 26 - badge_w, info_bottom), 10, fill=COLORS["bg_card"])
        draw.text((26, info_top + 8), f"Reason: {reason}", fill=COLORS["text_dim"], font=font_small)
        draw.text((26, info_top + 32), f"Leaf Ratio: {leaf_ratio}", fill=COLORS["text"], font=font_text)
        draw.text((26, info_top + 54), f"Components: {components}", fill=COLORS["text"], font=font_text)
        draw.text((26, info_top + 76), f"Confidence: {confidence}", fill=COLORS["text"], font=font_text)

        # Original image card
        orig_top = info_bottom + 14
        rrect((14, orig_top, 234, h - 14), 10, fill=COLORS["bg_card"])
        draw.text((26, orig_top + 8), "Original Image", fill=COLORS["text"], font=font_bold)
        if self.captured_image_path and os.path.exists(self.captured_image_path):
            try:
                original = Image.open(self.captured_image_path).convert("RGB")
                original.thumbnail((196, h - orig_top - 44), Image.LANCZOS)
                img.paste(original, (26, orig_top + 32))
            except Exception as exc:
                draw.text((26, orig_top + 32), f"Could not load image: {exc}", fill=COLORS["text_dim"], font=font_small)

        # Chart card
        chart_left = 246
        chart_top = orig_top
        chart_right = w - 14
        chart_bottom = h - 14
        rrect((chart_left, chart_top, chart_right, chart_bottom), 10, fill=COLORS["bg_card"])
        draw.text((chart_left + 12, chart_top + 8), "Prediction Confidence", fill=COLORS["text"], font=font_bold)

        plot_x0 = chart_left + 20
        plot_y0 = chart_top + 40
        plot_x1 = chart_right - 20
        plot_y1 = chart_bottom - 34

        # gridlines
        for frac in (0.25, 0.5, 0.75, 1.0):
            gy = plot_y1 - int((plot_y1 - plot_y0) * frac)
            draw.line((plot_x0, gy, plot_x1, gy), fill=COLORS["border"], width=1)

        names = ["Healthy", "Discolored", "Diseased"]
        bar_colors = [COLORS["green"], COLORS["yellow"], COLORS["orange"]]
        values = []
        for name in names:
            try:
                values.append(float(probabilities.get(name, 0.0)))
            except Exception:
                values.append(0.0)

        n = len(names)
        plot_w = plot_x1 - plot_x0
        slot_w = plot_w / n
        bar_w = min(60, slot_w * 0.5)

        for i, (name, value, bcolor) in enumerate(zip(names, values, bar_colors)):
            slot_center = plot_x0 + slot_w * i + slot_w / 2
            x0 = slot_center - bar_w / 2
            x1 = slot_center + bar_w / 2
            bar_h = max(2, int(max(0.0, min(1.0, value)) * (plot_y1 - plot_y0)))
            y0 = plot_y1 - bar_h
            rrect((x0, y0, x1, plot_y1), 4, fill=bcolor)
            draw.text((slot_center - 14, y0 - 16), f"{value:.2f}", fill=COLORS["text"], font=font_small)
            draw.text((slot_center - 28, plot_y1 + 6), name, fill=COLORS["text_dim"], font=font_small)

        return img

    # ------------------------------------------------------------------
    # More Details / Recommendations popup (built in details_window.py)
    # ------------------------------------------------------------------
    def show_more_details(self) -> None:
        """Shows the detailed analysis on the main preview area AND opens a recommendations popup."""
        if not self.last_prediction_payload:
            messagebox.showinfo("No Details", "Please analyze an image first.")
            return

        detail_path = self.last_detail_image_path or self._find_detail_image_path(self.last_prediction_payload)
        self.last_detail_image_path = detail_path

        if detail_path and self._show_image_file_on_label(detail_path):
            self.lbl_status.config(text="Detailed analysis shown.", fg=COLORS["teal"])
        else:
            fallback_img = self._build_fallback_details_image()
            self._show_pil_on_label(fallback_img)
            self.lbl_status.config(text="Detailed analysis shown.", fg=COLORS["teal"])

        # The preview now shows the segmented/annotated view instead of the
        # plain photo - let the user flip back to it without retaking.
        if self.captured_image_path and os.path.exists(self.captured_image_path):
            self.btn_back.config(state=tk.NORMAL, bg=COLORS["blue"])

        open_recommendations_window(self)

    def show_captured_photo(self) -> None:
        """Swaps the preview back to the plain captured photo (no
        segmentation/annotation overlay), so the user can re-check what was
        actually captured after viewing MORE DETAILS."""
        if not self.captured_image_path or not os.path.exists(self.captured_image_path):
            messagebox.showinfo("No Image", "No captured image available.")
            return

        if self._show_image_file_on_label(self.captured_image_path):
            self.lbl_status.config(text="Captured image shown.", fg=COLORS["text"])

    def _run_prediction_subprocess(self, image_path: str):
        runners = []
        for candidate in [os.environ.get("GBM_PYTHON"), self.ml_python, sys.executable]:
            if candidate and candidate not in runners and os.path.exists(candidate):
                runners.append(candidate)

        last_error = "No Python interpreter available for ML runner."
        for python_exec in runners:
            try:
                completed = subprocess.run(
                    [python_exec, PREDICT_RUNNER, image_path],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=ANALYZE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                last_error = str(exc)
                continue

            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()

            if completed.returncode == 0:
                try:
                    payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
                except Exception as exc:
                    raise RuntimeError(f"Prediction output was not valid JSON: {exc}") from exc

                label = payload.get("label", payload.get("prediction", "Error"))
                probabilities = payload.get("probabilities", payload.get("probs", {}))
                return label, probabilities, payload

            lines = [line.strip() for line in stderr.splitlines() if line.strip()]
            if lines:
                last_error = lines[-1]
            elif stdout:
                last_error = stdout.splitlines()[-1]
            else:
                last_error = f"Prediction runner failed with code {completed.returncode}."

        raise RuntimeError(last_error)

    def analyze_data(self) -> None:
        if not self.captured_image_path or not os.path.exists(self.captured_image_path):
            messagebox.showwarning("No Image", "Please press CAPTURE first.")
            return

        self.analysis_running = True
        self.btn_analyze.config(state=tk.DISABLED, bg=COLORS["gray"])
        self.btn_retake.config(state=tk.DISABLED, bg=COLORS["gray"])
        self.btn_more_details.config(state=tk.DISABLED, bg=COLORS["gray"])
        self._reset_save_button()
        self.lbl_status.config(text="Analyzing...", fg=COLORS["yellow"])
        self.root.update_idletasks()

        try:
            sensor_values = fullcomby.safe_read_sensors(self.light_sensor, self.ph_chan, self.cal)
            sensor_values = self._apply_ui_sensor_calibrations(sensor_values)
        except Exception as exc:
            print(f"Sensor read during analysis failed: {exc}")
            sensor_values = self.last_sensor_values

        lux_val = self._smooth_sensor_value(self.lux_reading_history, sensor_values.get("lux"))
        temp_val = self._smooth_sensor_value(self.temp_reading_history, sensor_values.get("temp_c"))
        ph_val = self._smooth_sensor_value(self.ph_reading_history, sensor_values.get("ph"))
        phv_val = sensor_values["ph_v"] if sensor_values.get("ph_v") is not None else 0.0

        self.last_sensor_values = dict(sensor_values)
        self.last_sensor_values["lux"] = lux_val
        self.last_sensor_values["temp_c"] = temp_val
        self.last_sensor_values["ph"] = ph_val

        lux_val = lux_val if lux_val is not None else 0.0
        temp_val = temp_val if temp_val is not None else 0.0
        ph_val = ph_val if ph_val is not None else 0.0

        self.lbl_light.config(
            text=f"Light: {lux_val:.2f} lux" if self.last_sensor_values["lux"] is not None else "Light: -- lux",
            fg=status_color(self.last_sensor_values["lux"], LIGHT_LOW_LUX, float("inf")),
        )
        self.lbl_temp.config(
            text=f"Water Temp: {temp_val:.2f} \u00b0C" if self.last_sensor_values["temp_c"] is not None else "Water Temp: -- \u00b0C",
            fg=status_color(self.last_sensor_values["temp_c"], TEMP_LOW_C, TEMP_HIGH_C),
        )
        self.lbl_ph.config(
            text=f"pH Level: {ph_val:.2f}" if self.last_sensor_values["ph"] is not None else "pH Level: --",
            fg=status_color(self.last_sensor_values["ph"], PH_LOW, PH_HIGH),
        )

        pred_label = "ML Error"
        p_h = p_disc = p_d = 0.0

        try:
            pred_label, prob_dict, payload = self._run_prediction_subprocess(self.captured_image_path)
            self.last_prediction_payload = payload
            self.last_prediction_label = pred_label
            # predict_runner.py nests "reason" inside payload["details"], not
            # at the top level, so check there first.
            details_dict = payload.get("details") if isinstance(payload, dict) else None
            details_dict = details_dict if isinstance(details_dict, dict) else {}
            self.last_prediction_reason = (
                payload.get("reason")
                or details_dict.get("reason")
                or payload.get("message", "")
            )
            self.last_detail_image_path = self._find_detail_image_path(payload)

            p_h = round(float(prob_dict.get("Healthy", 0.0)), 4)
            p_disc = round(float(prob_dict.get("Discolored", 0.0)), 4)
            p_d = round(float(prob_dict.get("Diseased", 0.0)), 4)

            result_color = self._prediction_text_color(pred_label)
            if pred_label.lower() == "no kangkong detected":
                self.lbl_status.config(text="No Kangkong Detected\nRetake the image.", fg=result_color)
            else:
                reason_text = self.last_prediction_reason
                if reason_text:
                    short_reason = str(reason_text).strip().splitlines()[0][:60]
                    self.lbl_status.config(text=f"{pred_label}\n{short_reason}", fg=result_color)
                else:
                    self.lbl_status.config(text=pred_label, fg=result_color)

            self.btn_more_details.config(state=tk.NORMAL, bg=COLORS["teal"])
        except Exception as ml_err:
            short_error = str(ml_err).strip().splitlines()[0][:80]
            self.lbl_status.config(text=f"ML Error:\n{short_error}", fg=COLORS["red"])
            self.btn_more_details.config(state=tk.DISABLED, bg=COLORS["gray"])
            print(f"ML Error: {ml_err}")

        # Analysis results are held in memory only. Nothing is written to
        # the database or CSV until the user explicitly taps SAVE.
        ts = datetime.now().isoformat(timespec="seconds")
        self.pending_analysis = {
            "timestamp": ts,
            "lux": lux_val,
            "temp_c": temp_val,
            "ph": ph_val,
            "ph_v": phv_val,
            "image_path": self.captured_image_path,
            "prediction": pred_label,
            "prob_healthy": p_h,
            "prob_discolored": p_disc,
            "prob_diseased": p_d,
        }
        self.btn_save.config(state=tk.NORMAL, text="SAVE", bg=COLORS["green"])

        self.analysis_running = False
        self.btn_analyze.config(state=tk.NORMAL, bg=COLORS["orange"])
        self.btn_retake.config(state=tk.NORMAL, bg=COLORS["gray"])

    def save_analysis(self) -> None:
        """Writes the pending analysis (from the last ANALYZE run) to the
        database and CSV log. Only runs when the user taps SAVE, giving them
        the choice to discard a result (e.g. by hitting RETAKE instead)."""
        data = self.pending_analysis
        if not data:
            messagebox.showinfo("Nothing to Save", "Analyze an image first.")
            return

        try:
            fullcomby.insert_row(
                data["timestamp"],
                round(data["lux"], 2),
                round(data["temp_c"], 2),
                round(data["ph"], 2),
                round(data["ph_v"], 3),
                data["image_path"],
                data["prediction"],
                data["prob_healthy"],
                data["prob_diseased"],
                data["prob_discolored"],
            )
            csv_utils.append_sensor_csv(
                fullcomby.CSV_PATH,
                data["timestamp"],
                data["lux"],
                data["temp_c"],
                data["ph"],
                data["ph_v"],
                data["prediction"],
                data["prob_healthy"],
                data["prob_discolored"],
                data["prob_diseased"],
                data["image_path"],
            )
            print(f"[{data['timestamp']}] Saved: {data['prediction']}")
            self.btn_save.config(state=tk.DISABLED, text="SAVED", bg=COLORS["gray"])
            self.pending_analysis = None
        except Exception as exc:
            print(f"Failed to save analysis: {exc}")
            messagebox.showerror("Save Error", f"Failed to save analysis:\n{exc}")

    def on_close(self) -> None:
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AquaponicsUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
