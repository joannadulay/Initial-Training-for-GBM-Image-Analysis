"""
Shared visual theme + small constants used across the UI modules.

Centralizing this here means ui_main.py (main window), history_window.py
(History tab / log viewer), and details_window.py (More Details /
recommendations popup) all stay visually consistent and only need one
import instead of copy-pasted color dictionaries.
"""

import tkinter as tk

# ---------------------------------------------------------------------------
# Visual theme.
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#161d29",            # app background
    "bg_panel": "#1e2736",      # right-side panel background
    "bg_card": "#232e42",       # card / grouping background
    "bg_card_alt": "#2a3650",   # slightly lighter card (rows, hover)
    "border": "#33415c",        # card border / separators
    "text": "#eef2f7",          # primary text
    "text_dim": "#9fb0c9",      # secondary text
    "text_muted": "#6b7c96",    # tertiary / hint text

    "blue": "#3b82f6",
    "blue_hover": "#5b93f7",
    "orange": "#f0973a",
    "orange_hover": "#f5ab5c",
    "teal": "#17b3a3",
    "teal_hover": "#2fcabb",
    "purple": "#9b6bd6",
    "purple_hover": "#ac82df",
    "gray": "#5b6a82",
    "gray_hover": "#71829c",
    "green": "#2fbf6e",
    "green_hover": "#41d17f",
    "red": "#e6584f",
    "red_hover": "#ec746c",
    "yellow": "#e0b93d",
}

# ---------------------------------------------------------------------------
# Sensor notification thresholds. Adjust these based on your thesis standard.
# ---------------------------------------------------------------------------
TEMP_LOW_C = 21.0
TEMP_HIGH_C = 30.0

# Tilapia + plants pH standard.
PH_LOW = 6.7
PH_HIGH = 7.8

# Minimum useful sunlight/light level. Adjust based on your thesis standard.
LIGHT_LOW_LUX = 1000.0

# Temperature calibration safety limits.
TEMP_CAL_MIN_RAW_GAP_C = 2.0       # two calibration points must be at least 2°C apart
TEMP_CAL_MIN_SLOPE = 0.70
TEMP_CAL_MAX_SLOPE = 1.30
TEMP_CAL_MIN_OUTPUT_C = 0.0
TEMP_CAL_MAX_OUTPUT_C = 50.0

# ---------------------------------------------------------------------------
# CSV schema for the sensor/prediction log.
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "timestamp",
    "light_lux",
    "water_temp_c",
    "ph_level",
    "ph_voltage",
    "ml_prediction",
    "prob_healthy",
    "prob_discolored",
    "prob_diseased",
    "image_path",
]

CSV_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime"],
    "light_lux": ["light_lux", "light lux", "light"],
    "water_temp_c": ["water_temp_c", "water_temp", "water temp", "temperature", "temp_c"],
    "ph_level": ["ph_level", "ph", "ph level"],
    "ph_voltage": ["ph_voltage", "ph_v", "ph voltage"],
    "ml_prediction": ["ml_prediction", "prediction", "gbm_analysis", "gbm prediction"],
    "prob_healthy": ["prob_healthy", "healthy_prob", "healthy"],
    "prob_discolored": ["prob_discolored", "discolored_prob", "discolored"],
    "prob_diseased": ["prob_diseased", "diseased_prob", "diseased"],
    "image_path": ["image_path", "image path", "path", "captured_image_path"],
}


# ---------------------------------------------------------------------------
# Small style helpers (free functions, not tied to any window/class).
# ---------------------------------------------------------------------------
def make_card(parent, **kwargs):
    """A bordered 'card' frame used to visually group related widgets."""
    defaults = dict(
        bg=COLORS["bg_card"],
        highlightthickness=1,
        highlightbackground=COLORS["border"],
        highlightcolor=COLORS["border"],
    )
    defaults.update(kwargs)
    return tk.Frame(parent, **defaults)


def _add_hover(widget, normal_bg, hover_bg):
    def on_enter(_):
        if str(widget["state"]) != "disabled":
            widget.config(bg=hover_bg)

    def on_leave(_):
        if str(widget["state"]) != "disabled":
            widget.config(bg=normal_bg)

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def make_button(parent, text, bg, hover_bg, command, font=None, state=tk.NORMAL, fg="white"):
    btn = tk.Button(
        parent,
        text=text,
        font=font or ("Helvetica", 11, "bold"),
        bg=bg,
        fg=fg,
        activebackground=hover_bg,
        activeforeground=fg,
        bd=0,
        relief=tk.FLAT,
        cursor="hand2",
        padx=6,
        pady=7,
        state=state,
        command=command,
    )
    _add_hover(btn, bg, hover_bg)
    return btn


def status_color(value, low, high):
    if value is None:
        return COLORS["text_muted"]
    if value < low or value > high:
        return COLORS["red"]
    return COLORS["green"]


def center_window_over(win, parent):
    """Centers a Toplevel over its parent window (best-effort)."""
    try:
        win.update_idletasks()
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        ww = win.winfo_width()
        wh = win.winfo_height()
        x = px + (pw - ww) // 2
        y = py + (ph - wh) // 2
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    except Exception:
        pass
