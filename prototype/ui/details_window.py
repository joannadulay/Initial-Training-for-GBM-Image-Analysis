"""
"More Details" recommendations popup for the Kangkong Aquaponics Monitor.

Pulled out of ui_main.py so the main file stays focused on
camera/sensor/analyze logic. Takes the running AquaponicsUI instance so it
can read the latest sensor values / prediction status without duplicating
that state.
"""

import tkinter as tk
from tkinter import ttk

from theme import COLORS, TEMP_LOW_C, TEMP_HIGH_C, PH_LOW, PH_HIGH, LIGHT_LOW_LUX, make_button


def _merge_payload(app):
    """
    predict_core.py nests reason/leaf_ratio/component_count/timings/etc.
    inside payload["details"], while label/probabilities live at the top
    level. Merge them into one lookup dict (nested details take priority
    on key collisions, since they're the more specific/authoritative
    source) so callers here don't need to know which layer something
    lives in.
    """
    payload = app.last_prediction_payload if isinstance(app.last_prediction_payload, dict) else {}
    nested_details = payload.get("details")
    nested_details = nested_details if isinstance(nested_details, dict) else {}
    top_level = {k: v for k, v in payload.items() if k != "details"}
    return {**top_level, **nested_details}


STAGE_ORDER = [
    "Image Loading", "Segmentation", "Feature Extraction",
    "Gate Model", "GBM Predict", "Predict Probability", "Visualization",
]


# ---------------------------------------------------------------------------
# Colour helpers (used for card hover tint - no new theme keys required)
# ---------------------------------------------------------------------------

def _clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))


def _shade(hex_color: str, amount: int) -> str:
    """Lighten (amount > 0) or darken (amount < 0) a #rrggbb color."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#" + hex_color
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = _clamp(r + amount), _clamp(g + amount), _clamp(b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_analysis_findings(app):
    """Findings specific to the last image analysis - not sensor readings,
    which already have their own section below."""
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)

    label = app.last_prediction_label or merged.get("label") or merged.get("prediction") or "--"
    reason = app.last_prediction_reason or merged.get("reason") or "No reason returned."
    leaf_ratio = merged.get("leaf_ratio")
    components = merged.get("component_count")
    gate_prob = merged.get("gate_leaf_probability")
    probs = merged.get("probabilities")
    probs = probs if isinstance(probs, dict) else {}

    accent = app._prediction_text_color(label)
    items = [(f"Prediction: {label}", accent)]
    items.append((f"Reason: {reason}", COLORS["text_muted"]))

    if isinstance(leaf_ratio, (int, float)):
        items.append((f"Leaf Area Ratio: {leaf_ratio:.4f}", COLORS["yellow"]))
    if isinstance(components, (int, float)):
        items.append((f"Leaf-like Regions Detected: {int(components)}", COLORS["yellow"]))
    if isinstance(gate_prob, (int, float)):
        items.append((f"Gate Leaf Probability: {gate_prob:.4f}", COLORS["yellow"]))

    prob_colors = {"Healthy": COLORS["green"], "Discolored": COLORS["yellow"], "Diseased": COLORS["red"]}
    for cls_name in ["Healthy", "Discolored", "Diseased"]:
        if cls_name in probs:
            try:
                val = float(probs[cls_name])
            except (TypeError, ValueError):
                continue
            items.append((f"{cls_name} Confidence: {val:.4f}", prob_colors[cls_name]))

    return items


def _first_present(merged, *keys):
    """Return the first key in `keys` that exists in `merged` with a usable
    numeric/string value, else None. Lets us stay agnostic to whatever
    naming the feature extractor happens to use."""
    for k in keys:
        if k in merged and merged[k] is not None:
            return merged[k]
    return None


def _build_visual_findings(app):
    """
    Human-readable description of *what the image itself looked like*,
    built from the engineered features in feature_extractor.py:
    engineered_features() -> [yellow_ratio, green_ratio, lesion_ratio,
    dark_ratio, mean_h, std_h, mean_s, std_s, mean_v, std_gray, leaf_ratio,
    seg_rejected, lesion_count, mean_area_ratio, std_area_ratio,
    mean_circularity, fragmentation], plus edge_density_features().

    NOTE: this assumes predict_core.py exposes these under payload["details"]
    using the same names as feature_extractor.py. If it instead only passes
    the raw concatenated feature vector (no dict), none of the keys below
    will be found and this falls back to a single label-driven line - update
    the key names here (or predict_core.py's output) to match once confirmed.

    Also note: the pipeline characterizes colored spots/lesions (yellow,
    brown, dark/necrotic) by hue + shape, not literal holes in the leaf
    tissue - there's no hole/puncture detector in feature_extractor.py, so
    that's intentionally not claimed here.
    """
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)
    label = app.last_prediction_label or merged.get("label") or merged.get("prediction") or "--"
    probs = merged.get("probabilities")
    probs = probs if isinstance(probs, dict) else {}
    confidence = None
    if label in probs:
        try:
            confidence = float(probs[label])
        except (TypeError, ValueError):
            confidence = None

    items = []

    yellow_ratio = _first_present(merged, "yellow_ratio")
    green_ratio = _first_present(merged, "green_ratio")
    if isinstance(yellow_ratio, (int, float)):
        if yellow_ratio > 0.12:
            items.append((
                f"Yellowing: {yellow_ratio:.1%} of the leaf area reads as yellow-hued, consistent with "
                "chlorosis or early nutrient deficiency rather than healthy green tissue.",
                COLORS["yellow"],
            ))
        elif yellow_ratio > 0.04:
            items.append((
                f"Yellowing: A small amount of yellow-hued area ({yellow_ratio:.1%}) was detected - "
                "worth watching but not yet a strong signal on its own.",
                COLORS["yellow"],
            ))
        elif isinstance(green_ratio, (int, float)) and green_ratio > 0.5:
            items.append((
                f"Yellowing: Minimal yellow area detected ({yellow_ratio:.1%}); the leaf reads as "
                f"predominantly green ({green_ratio:.1%}).",
                COLORS["green"],
            ))

    lesion_ratio = _first_present(merged, "lesion_ratio")
    dark_ratio = _first_present(merged, "dark_ratio")
    lesion_count = _first_present(merged, "lesion_count")
    mean_circularity = _first_present(merged, "mean_circularity")
    fragmentation = _first_present(merged, "fragmentation")

    if isinstance(lesion_ratio, (int, float)) and lesion_ratio > 0.02:
        shape_note = ""
        color = COLORS["yellow"]
        if isinstance(lesion_count, (int, float)) and lesion_count >= 1:
            n = int(lesion_count)
            if n >= 3 and isinstance(mean_circularity, (int, float)) and mean_circularity > 0.55:
                shape_note = (
                    f" Detected as {n} small, roughly circular spots - a pattern more typical of disease "
                    "lesions than general discoloration."
                )
                color = COLORS["red"]
            elif n <= 2:
                shape_note = (
                    f" Detected as {n} larger, less circular patch(es) - more typical of general "
                    "discoloration/chlorosis than active disease spotting."
                )
                color = COLORS["yellow"]
            else:
                shape_note = f" Detected as {n} distinct brown/necrotic region(s)."
                color = COLORS["yellow"] if n < 3 else COLORS["red"]
        items.append((
            f"Brown/Necrotic Spotting: {lesion_ratio:.1%} of the leaf area shows brown discoloration.{shape_note}",
            color,
        ))
    elif isinstance(lesion_ratio, (int, float)):
        items.append((
            f"Brown/Necrotic Spotting: Negligible brown discoloration detected ({lesion_ratio:.1%}).",
            COLORS["green"],
        ))

    if isinstance(dark_ratio, (int, float)) and dark_ratio > 0.03:
        items.append((
            f"Dark/Necrotic Centers: {dark_ratio:.1%} of the leaf reads as very dark or near-black, "
            "which often marks true necrotic tissue at the center of disease lesions rather than "
            "simple discoloration.",
            COLORS["red"],
        ))

    if isinstance(fragmentation, (int, float)) and fragmentation > 0 and isinstance(lesion_ratio, (int, float)) and lesion_ratio > 0.02:
        if fragmentation > 8:
            items.append((
                "Spot Pattern: Many small, scattered spots relative to total affected area - a "
                "fragmented pattern typically associated with active disease rather than a single "
                "stress patch.",
                COLORS["red"],
            ))

    edge_density = _first_present(merged, "edge_density")
    if isinstance(edge_density, (int, float)):
        if edge_density > 0.15:
            items.append((
                f"Edge Sharpness: Edge density ({edge_density:.3f}) is elevated, meaning the surface has "
                "sharp, high-contrast boundaries typical of distinct spots rather than a smooth color "
                "gradient.",
                COLORS["yellow"],
            ))

    # Fallback: nothing specific enough was available, so give a single
    # label-driven description instead of leaving the section empty.
    if not items:
        conf_txt = f" ({confidence * 100:.1f}% confidence)" if isinstance(confidence, (int, float)) else ""
        if label == "Diseased":
            text = (
                f"The captured image was classified as Diseased{conf_txt}. Per-region breakdown (yellow/"
                "brown/dark area, spot count and shape) isn't reaching this window yet - check that "
                "predict_core.py forwards the engineered feature values under payload['details'] using "
                "the same names as feature_extractor.py."
            )
            color = COLORS["red"]
        elif label == "Discolored":
            text = (
                f"The captured image was classified as Discolored{conf_txt}, but the underlying color/"
                "spot signals aren't reaching this window yet - same predict_core.py wiring note as above."
            )
            color = COLORS["yellow"]
        elif label == "Healthy":
            text = (
                f"The captured image was classified as Healthy{conf_txt}, with no discoloration or "
                "lesion signal strong enough to flag."
            )
            color = COLORS["green"]
        else:
            text = f"No detailed visual signal available for prediction '{label}'{conf_txt}."
            color = COLORS["text_muted"]
        items.append((text, color))

    return items


def _build_performance_lines(app):
    """Per-stage timing breakdown from the last analysis run, if present."""
    if not app.last_prediction_payload:
        return None

    merged = _merge_payload(app)
    timings = merged.get("timings")
    if not isinstance(timings, dict) or not timings:
        return None

    items = []
    total_ms = 0.0
    for stage in STAGE_ORDER:
        if stage in timings:
            try:
                ms = float(timings[stage])
            except (TypeError, ValueError):
                continue
            total_ms += ms
            items.append((f"{stage}: {ms:.2f} ms", COLORS["text"]))

    if not items:
        return None

    items.append((f"Total Pipeline: {total_ms:.2f} ms", COLORS["teal"]))
    return items


def _rec_severity_color(text: str) -> str:
    lowered = text.lower()
    if "optimal" in lowered or "good light" in lowered or "keep up" in lowered:
        return COLORS["green"]
    if "unavailable" in lowered or "no data" in lowered:
        return COLORS["text_muted"]
    if "too cold" in lowered or "too hot" in lowered or "too acidic" in lowered \
            or "too alkaline" in lowered or "too low" in lowered or "possible discoloration" in lowered \
            or "disease" in lowered:
        return COLORS["red"]
    return COLORS["yellow"]


def _rec_severity_icon(text: str) -> str:
    lowered = text.lower()
    if "optimal" in lowered or "good light" in lowered or "keep up" in lowered:
        return "\N{CHECK MARK}"
    if "unavailable" in lowered or "no data" in lowered:
        return "\N{BULLET}"
    if "too cold" in lowered or "too hot" in lowered or "too acidic" in lowered \
            or "too alkaline" in lowered or "too low" in lowered or "possible discoloration" in lowered \
            or "disease" in lowered:
        return "\N{HEAVY BALLOT X}"
    return "\N{WARNING SIGN}"


def _build_plant_health_recommendation(app):
    """
    Builds a description of what the last image analysis actually found
    (label, reason, confidence, leaf coverage) plus a matching care
    recommendation - rather than just telling the user to go capture an
    image.
    """
    if not app.last_prediction_payload:
        return "Plant Health: Capture and Analyze an image to get ML health recommendations."

    merged = _merge_payload(app)
    label = app.last_prediction_label or merged.get("label") or merged.get("prediction") or "--"
    reason = app.last_prediction_reason or merged.get("reason")
    leaf_ratio = merged.get("leaf_ratio")
    probs = merged.get("probabilities")
    probs = probs if isinstance(probs, dict) else {}

    confidence = None
    if label in probs:
        try:
            confidence = float(probs[label])
        except (TypeError, ValueError):
            confidence = None

    findings = []
    if isinstance(confidence, (int, float)):
        findings.append(f"{confidence * 100:.1f}% confidence")
    if reason:
        findings.append(reason)
    if isinstance(leaf_ratio, (int, float)):
        findings.append(f"leaf coverage {leaf_ratio:.1%}")
    findings_text = f" ({'; '.join(findings)})" if findings else ""

    if label == "Diseased":
        return (
            f"Plant Health: The last image analysis found signs of Disease{findings_text}. "
            "Check for pests, root rot, or nutrient deficiencies (such as nitrogen/iron), "
            "and consider isolating the affected plant."
        )
    if label == "Discolored":
        return (
            f"Plant Health: The last image analysis found Discoloration{findings_text}. "
            "This can point to early nutrient deficiency or light stress - review recent "
            "pH/temperature/lux readings and watch for progression."
        )
    if label == "Healthy":
        return (
            f"Plant Health: The last image analysis found the leaves Healthy{findings_text}. "
            "Keep up the current watering, lighting, and nutrient routine."
        )

    return f"Plant Health: Last analysis result was '{label}'{findings_text}."


def _build_recommendations(app):
    lux = app.last_sensor_values.get("lux")
    temp = app.last_sensor_values.get("temp_c")
    ph = app.last_sensor_values.get("ph")

    recs = []

    if temp is None:
        recs.append("Temperature: No data available yet.")
    elif temp < TEMP_LOW_C:
        recs.append(f"Temperature ({temp:.2f} °C): Water is too cold. Kangkong prefers {TEMP_LOW_C}-{TEMP_HIGH_C}°C. Consider adding an aquarium heater.")
    elif temp > TEMP_HIGH_C:
        recs.append(f"Temperature ({temp:.2f} °C): Water is too hot. Consider a partial water change with cooler water, adding shade, or improving ventilation.")
    else:
        recs.append(f"Temperature ({temp:.2f} °C): Optimal! The water temperature is perfect for Kangkong.")

    if ph is None:
        recs.append("pH Level: No data available yet.")
    elif ph < PH_LOW:
        recs.append(f"pH Level ({ph:.2f}): Water is too acidic. Add a pH UP solution (or agricultural lime) to safely raise it to {PH_LOW}-{PH_HIGH}.")
    elif ph > PH_HIGH:
        recs.append(f"pH Level ({ph:.2f}): Water is too alkaline. Add a pH DOWN solution (like phosphoric acid) to safely lower it to {PH_LOW}-{PH_HIGH}.")
    else:
        recs.append(f"pH Level ({ph:.2f}): Optimal! This range is excellent for nutrient absorption.")

    if lux is None:
        recs.append("Sunlight: No data available yet.")
    elif lux < LIGHT_LOW_LUX:
        recs.append(f"Sunlight ({lux:.2f} lux): Light intensity is low. Turn on grow lights or move the setup to a sunnier location.")
    else:
        recs.append(f"Sunlight ({lux:.2f} lux): Good light intensity detected. The plants are receiving adequate light.")

    recs.append(_build_plant_health_recommendation(app))

    return recs


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------

def _add_section_title(parent, text, icon=""):
    wrap = tk.Frame(parent, bg=COLORS["bg"])
    wrap.pack(fill=tk.X, pady=(18, 6))

    label_text = f"{icon}  {text}" if icon else text
    tk.Label(
        wrap, text=label_text, font=("Helvetica", 12, "bold"),
        bg=COLORS["bg"], fg=COLORS["text"], anchor="w",
    ).pack(side=tk.LEFT)

    # thin accent rule filling the remaining width, gives sections a
    # clearer visual break than plain whitespace did before
    rule = tk.Frame(wrap, bg=COLORS["bg_card"], height=1)
    rule.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0), pady=(2, 0))


def _add_card(parent, text, color, icon=None):
    outer = tk.Frame(parent, bg=COLORS["bg"])
    outer.pack(fill=tk.X, pady=4)

    card = tk.Frame(
        outer, bg=COLORS["bg_card"],
        highlightbackground=_shade(COLORS["bg_card"], 18),
        highlightthickness=1, bd=0,
    )
    card.pack(fill=tk.X, ipady=4)

    strip = tk.Frame(card, bg=color, width=5)
    strip.pack(side=tk.LEFT, fill=tk.Y)

    inner = tk.Frame(card, bg=COLORS["bg_card"])
    inner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 12), pady=10)

    if icon:
        row = tk.Frame(inner, bg=COLORS["bg_card"])
        row.pack(fill=tk.X)
        tk.Label(
            row, text=icon, font=("Helvetica", 11, "bold"),
            bg=COLORS["bg_card"], fg=color,
        ).pack(side=tk.LEFT, padx=(0, 8), anchor="n")
        tk.Label(
            row, text=text, font=("Helvetica", 11), bg=COLORS["bg_card"], fg=COLORS["text"],
            wraplength=580, justify="left", anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    else:
        tk.Label(
            inner, text=text, font=("Helvetica", 11), bg=COLORS["bg_card"], fg=COLORS["text"],
            wraplength=610, justify="left", anchor="w",
        ).pack(fill=tk.X, expand=True)

    # subtle hover highlight so the list doesn't feel static
    hover_bg = _shade(COLORS["bg_card"], 10)

    def _on_enter(_e):
        card.configure(bg=hover_bg)
        inner.configure(bg=hover_bg)
        for w in inner.winfo_children():
            w.configure(bg=hover_bg)
            for c in w.winfo_children():
                c.configure(bg=hover_bg)

    def _on_leave(_e):
        card.configure(bg=COLORS["bg_card"])
        inner.configure(bg=COLORS["bg_card"])
        for w in inner.winfo_children():
            w.configure(bg=COLORS["bg_card"])
            for c in w.winfo_children():
                c.configure(bg=COLORS["bg_card"])

    for widget in (card, inner):
        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)


def _add_tab_panel(notebook, label):
    """Creates one notebook tab ("panel") with its own independently
    scrollable, padded content area.

    Returns (content_frame, unbind_wheel): pack cards into content_frame;
    call unbind_wheel() during window cleanup so a stray bind_all() from
    this panel's mousewheel handler doesn't outlive the window.
    """
    page = tk.Frame(notebook, bg=COLORS["bg"])
    notebook.add(page, text=label)

    canvas = tk.Canvas(page, bg=COLORS["bg"], highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll_y = tk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scroll_y.set)

    inner = tk.Frame(canvas, bg=COLORS["bg"])
    canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_configure(_event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_resize(event):
        # keep the inner frame the same width as the canvas so cards
        # stretch/wrap correctly instead of leaving dead space on resize
        canvas.itemconfigure(canvas_window, width=event.width)

    inner.bind("<Configure>", on_configure)
    canvas.bind("<Configure>", on_canvas_resize)

    # Bind the mousewheel only while the pointer is over this panel's
    # canvas (bind + unbind on Enter/Leave), same pattern as before but
    # now per-tab since each tab has its own canvas.
    def on_mousewheel(event):
        canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def bind_wheel(_e):
        canvas.bind_all("<MouseWheel>", on_mousewheel)

    def unbind_wheel(_e=None):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", bind_wheel)
    canvas.bind("<Leave>", unbind_wheel)

    # padding wrapper so cards aren't flush against the tab's edges
    content = tk.Frame(inner, bg=COLORS["bg"])
    content.pack(fill=tk.BOTH, expand=True, padx=14, pady=(12, 16))

    return content, unbind_wheel


def open_recommendations_window(app):
    """Builds and shows the 'System Recommendations' popup.

    `app` is the running AquaponicsUI instance (needs .root, .lbl_status,
    and .last_sensor_values).
    """
    details_window = tk.Toplevel(app.root)
    details_window.title("Analysis Details & Recommendations")
    details_window.geometry("760x600")
    details_window.minsize(620, 420)
    details_window.configure(bg=COLORS["bg"])
    details_window.transient(app.root)

    # --- Header -----------------------------------------------------
    header = tk.Frame(details_window, bg=COLORS["bg_panel"])
    header.pack(fill=tk.X)

    header_text = tk.Frame(header, bg=COLORS["bg_panel"])
    header_text.pack(side=tk.LEFT, padx=16, pady=12)
    tk.Label(
        header_text, text="Aquaponic System Management",
        font=("Helvetica", 15, "bold"), bg=COLORS["bg_panel"], fg=COLORS["text"],
    ).pack(anchor="w")
    tk.Label(
        header_text, text="Image analysis, sensor status, and care recommendations",
        font=("Helvetica", 9), bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
    ).pack(anchor="w")

    close_button_slot = {}
    make_button(
        header, "CLOSE", COLORS["red"], COLORS["red_hover"],
        lambda: close_button_slot["close"](),
        font=("Helvetica", 10, "bold"),
    ).pack(side=tk.RIGHT, padx=16)

    # thin divider under the header for a cleaner separation from the body
    tk.Frame(details_window, bg=_shade(COLORS["bg_panel"], -12), height=2).pack(fill=tk.X)

    analysis_items = _build_analysis_findings(app)
    visual_items = _build_visual_findings(app)
    performance_items = _build_performance_lines(app)
    recs = _build_recommendations(app)

    # Panels instead of one long stacked/scrolling list - each section
    # gets its own tab, Recommendations shown first by default, so nothing
    # requires scrolling past unrelated sections to be seen.
    body = tk.Frame(details_window, bg=COLORS["bg"])
    body.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

    # Restyle ttk's notebook/tabs to match the dark theme instead of the
    # default OS look, which would otherwise clash with everything else.
    style = ttk.Style(details_window)
    style.theme_use("clam")
    style.configure("Details.TNotebook", background=COLORS["bg"], borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure(
        "Details.TNotebook.Tab",
        background=COLORS["bg_panel"],
        foreground=COLORS["text_muted"],
        padding=(16, 9),
        font=("Helvetica", 10, "bold"),
        borderwidth=0,
    )
    style.map(
        "Details.TNotebook.Tab",
        background=[("selected", COLORS["bg_card"])],
        foreground=[("selected", COLORS["text"])],
    )

    notebook = ttk.Notebook(body, style="Details.TNotebook")
    notebook.pack(fill=tk.BOTH, expand=True)

    wheel_unbinders = []

    # Recommendations panel is added (and shown) first - this is what the
    # user actually opens the window to see.
    recs_panel, unbind = _add_tab_panel(notebook, "\N{ELECTRIC LIGHT BULB} Recommendations")
    wheel_unbinders.append(unbind)
    for rec in recs:
        color = _rec_severity_color(rec)
        icon = _rec_severity_icon(rec)
        _add_card(recs_panel, rec, color, icon=icon)

    if performance_items:
        perf_panel, unbind = _add_tab_panel(notebook, "\N{STOPWATCH} Performance")
        wheel_unbinders.append(unbind)
        for text, color in performance_items:
            _add_card(perf_panel, text, color)

    if visual_items:
        visual_panel, unbind = _add_tab_panel(notebook, "\N{HERB} Visual Findings")
        wheel_unbinders.append(unbind)
        for text, color in visual_items:
            _add_card(visual_panel, text, color)

    # Raw inference output goes last - it's the most technical/lowest-level
    # panel and the least likely thing someone wants to see first.
    if analysis_items:
        inference_panel, unbind = _add_tab_panel(notebook, "\N{MICROSCOPE} Inference")
        wheel_unbinders.append(unbind)
        for text, color in analysis_items:
            _add_card(inference_panel, text, color)

    notebook.select(0)

    def _cleanup_on_close():
        for unbind in wheel_unbinders:
            unbind()
        details_window.grab_release()
        details_window.destroy()

    close_button_slot["close"] = _cleanup_on_close
    details_window.protocol("WM_DELETE_WINDOW", _cleanup_on_close)
    details_window.bind("<Escape>", lambda _e: _cleanup_on_close())

    # Make sure this popup actually receives clicks/focus (fixes the
    # CLOSE button appearing unresponsive on some window managers/kiosks).
    details_window.update_idletasks()
    details_window.lift()
    details_window.focus_force()
    details_window.grab_set()

    return details_window
