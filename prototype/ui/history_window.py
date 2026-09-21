"""
History tab (Toplevel window) for the Kangkong Aquaponics Monitor.

Shows every saved sensor reading + ML prediction as a scrollable, color
coded table, with:
  - a live text Search box
  - a Prediction Filter dropdown
  - VIEW IMAGE, DELETE SELECTED, EXPORT CSV, and REFRESH actions
  - multi-row selection (Ctrl/Shift click) for bulk delete
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import csv_utils
from theme import COLORS, CSV_COLUMNS, make_button


FILTER_OPTIONS = ["All", "Healthy", "Discolored", "Diseased", "No Kangkong", "Unclassified"]


class HistoryWindow:
    def __init__(self, parent_root, csv_path: str, image_dir: str, base_dir: str = None):
        self.parent_root = parent_root
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.base_dir = base_dir

        # self.all_rows keeps file order (oldest -> newest). row_lookup maps
        # a Treeview iid to an index into self.all_rows so delete/view/export
        # can always find the real underlying record, regardless of the
        # current search/filter/sort-by-newest-first display order.
        self.all_rows = []
        self.row_lookup = {}
        self.displayed_rows = []  # rows currently shown, in display order

        self.win = tk.Toplevel(parent_root)
        self.win.title("Sensor Readings Log")
        self.win.geometry("800x480")
        self.win.configure(bg=COLORS["bg"])
        self.win.resizable(False, False)
        self.win.transient(parent_root)

        self._build_ui()
        self.win.protocol("WM_DELETE_WINDOW", self._go_back_to_main)

        if not os.path.exists(csv_path):
            messagebox.showinfo("No Data", "No CSV data found yet. Please analyze an image first.", parent=self.win)
            self._go_back_to_main()
            return

        self._load_data()

        # Grab focus/raise so the window (and its buttons) reliably receives
        # clicks even on the touchscreen kiosk setup.
        self.win.lift()
        self.win.focus_force()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        header = tk.Frame(self.win, bg=COLORS["bg_panel"])
        header.pack(fill=tk.X)

        header_inner = tk.Frame(header, bg=COLORS["bg_panel"])
        header_inner.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(
            header_inner,
            text="History",
            font=("Helvetica", 13, "bold"),
            bg=COLORS["bg_panel"],
            fg=COLORS["text"],
        ).pack(side=tk.LEFT)

        self.lbl_history_count = tk.Label(
            header_inner,
            text="",
            font=("Helvetica", 9),
            bg=COLORS["bg_panel"],
            fg=COLORS["text_muted"],
        )
        self.lbl_history_count.pack(side=tk.LEFT, padx=(8, 0))

        make_button(
            header_inner, "◀ BACK", COLORS["red"], COLORS["red_hover"],
            self._go_back_to_main, font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        # Action buttons live in the header (next to BACK) instead of a
        # separate bottom bar, so they're always visible even on short
        # displays where the table + horizontal scrollbar would otherwise
        # push a bottom action bar off-screen.
        make_button(
            header_inner, "Refresh", COLORS["gray"], COLORS["gray_hover"],
            self._refresh, font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        make_button(
            header_inner, "Export CSV", COLORS["teal"], COLORS["teal_hover"],
            self._export_csv, font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        make_button(
            header_inner, "Delete Selected", COLORS["red"], COLORS["red_hover"],
            self._delete_selected, font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        make_button(
            header_inner, "View Analysis", COLORS["orange"], COLORS["orange_hover"],
            self._view_selected_analysis, font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        make_button(
            header_inner, "View Image", COLORS["blue"], COLORS["blue_hover"],
            self._view_selected_image, font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        tk.Frame(self.win, bg=COLORS["border"], height=1).pack(fill=tk.X)

        # --- Search + Filter row -----------------------------------------
        toolbar = tk.Frame(self.win, bg=COLORS["bg"])
        toolbar.pack(fill=tk.X, padx=14, pady=(10, 0))

        tk.Label(
            toolbar, text="Search:", font=("Helvetica", 10),
            bg=COLORS["bg"], fg=COLORS["text_dim"],
        ).pack(side=tk.LEFT)

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._apply_filters())
        search_entry = tk.Entry(
            toolbar, textvariable=self.search_var, font=("Helvetica", 10),
            bg=COLORS["bg_card"], fg=COLORS["text"], insertbackground=COLORS["text"],
            relief=tk.FLAT, highlightthickness=1, highlightbackground=COLORS["border"],
        )
        search_entry.pack(side=tk.LEFT, padx=(6, 18), ipady=3, fill=tk.X, expand=True)
        search_entry.bind("<FocusIn>", lambda _e: search_entry.config(highlightbackground=COLORS["blue"], highlightcolor=COLORS["blue"]))
        search_entry.bind("<FocusOut>", lambda _e: search_entry.config(highlightbackground=COLORS["border"], highlightcolor=COLORS["border"]))

        tk.Label(
            toolbar, text="Filter:", font=("Helvetica", 10),
            bg=COLORS["bg"], fg=COLORS["text_dim"],
        ).pack(side=tk.LEFT)

        self.filter_var = tk.StringVar(value="All")
        filter_menu = ttk.Combobox(
            toolbar, textvariable=self.filter_var, values=FILTER_OPTIONS,
            state="readonly", width=14, font=("Helvetica", 10),
        )
        filter_menu.pack(side=tk.LEFT, padx=(6, 0))
        filter_menu.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        # --- Legend --------------------------------------------------------
        legend = tk.Frame(self.win, bg=COLORS["bg"])
        legend.pack(fill=tk.X, padx=14, pady=(10, 0))
        legend_items = [
            ("Healthy", COLORS["green"]),
            ("Discolored", COLORS["yellow"]),
            ("Diseased", COLORS["orange"]),
            ("No Kangkong", COLORS["red"]),
        ]
        for name, color in legend_items:
            tk.Label(legend, text="●", font=("Helvetica", 11), bg=COLORS["bg"], fg=color).pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(
                legend, text=name, font=("Helvetica", 9), bg=COLORS["bg"], fg=COLORS["text_dim"],
            ).pack(side=tk.LEFT, padx=(0, 16))

        # --- Table -----------------------------------------------------------
        main_frame = tk.Frame(
            self.win, bg=COLORS["bg_card"],
            highlightthickness=1, highlightbackground=COLORS["border"],
        )
        main_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(10, 10))
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "History.Treeview",
            rowheight=30,
            font=("Helvetica", 10),
            background=COLORS["bg_card"],
            fieldbackground=COLORS["bg_card"],
            foreground=COLORS["text"],
            borderwidth=0,
        )
        style.configure(
            "History.Treeview.Heading",
            font=("Helvetica", 10, "bold"),
            background=COLORS["bg_panel"],
            foreground=COLORS["text"],
            relief="flat",
        )
        style.map("History.Treeview.Heading", background=[("active", COLORS["bg_card_alt"])])
        style.map("History.Treeview", background=[("selected", COLORS["blue"])])

        self.tree = ttk.Treeview(main_frame, style="History.Treeview", selectmode="extended")
        self.tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, width=18, command=self.tree.yview)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")

        tree_scroll_x = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, width=18, command=self.tree.xview)
        tree_scroll_x.grid(row=1, column=0, sticky="ew")

        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        self.tree.tag_configure("row_even", background=COLORS["bg_card"])
        self.tree.tag_configure("row_odd", background=COLORS["bg_card_alt"])
        self.tree.tag_configure("status_green", background="#1c3a2c", foreground=COLORS["text"])
        self.tree.tag_configure("status_yellow", background="#3a3319", foreground=COLORS["text"])
        self.tree.tag_configure("status_orange", background="#3a2a19", foreground=COLORS["text"])
        self.tree.tag_configure("status_red", background="#3a1e1e", foreground=COLORS["text"])

        self.tree["columns"] = CSV_COLUMNS
        self.tree["show"] = "headings"

        col_widths = {
            "timestamp": 165,
            "light_lux": 90,
            "water_temp_c": 110,
            "ph_level": 90,
            "ph_voltage": 100,
            "ml_prediction": 140,
            "prob_healthy": 115,
            "prob_discolored": 130,
            "prob_diseased": 120,
            "image_path": 300,
        }

        for col in CSV_COLUMNS:
            heading_text = col.replace("_", " ").title()
            self.tree.heading(col, text=heading_text)
            c_width = col_widths.get(col, 120)
            self.tree.column(col, width=c_width, minwidth=c_width, stretch=False, anchor="center")

        self._bind_scroll_and_drag()

        # --- Bottom tip bar -----------------------------------------------
        # (Action buttons now live in the header next to BACK, so they're
        # always visible regardless of screen height.)
        actions = tk.Frame(self.win, bg=COLORS["bg"])
        actions.pack(fill=tk.X, padx=14, pady=(0, 10))

        tk.Label(
            actions, text="Tip: Hold Ctrl or Shift to select multiple rows",
            font=("Helvetica", 9, "italic"), bg=COLORS["bg"], fg=COLORS["text_muted"],
        ).pack(side=tk.RIGHT)

    def _bind_scroll_and_drag(self):
        def on_mousewheel(event):
            self.tree.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        def on_linux_scroll_up(_event):
            self.tree.yview_scroll(-1, "units")
            return "break"

        def on_linux_scroll_down(_event):
            self.tree.yview_scroll(1, "units")
            return "break"

        drag_data = {"x": 0, "y": 0}

        def start_drag(event):
            drag_data["x"] = event.x
            drag_data["y"] = event.y

        def drag_scroll(event):
            dx = event.x - drag_data["x"]
            dy = event.y - drag_data["y"]

            if abs(dx) > 20:
                self.tree.xview_scroll(1 if dx < 0 else -1, "units")
                drag_data["x"] = event.x
            if abs(dy) > 20:
                self.tree.yview_scroll(1 if dy < 0 else -1, "units")
                drag_data["y"] = event.y

        self.tree.bind("<MouseWheel>", on_mousewheel)
        self.tree.bind("<Button-4>", on_linux_scroll_up)
        self.tree.bind("<Button-5>", on_linux_scroll_down)
        self.tree.bind("<ButtonPress-1>", start_drag, add="+")
        self.tree.bind("<B1-Motion>", drag_scroll, add="+")

    # ------------------------------------------------------------------
    # Data loading / filtering / rendering
    # ------------------------------------------------------------------
    def _row_status_tag(self, prediction: str) -> str:
        label_text = str(prediction or "").strip().lower()
        if "no kangkong" in label_text or "not kangkong" in label_text:
            return "status_red"
        if "healthy" in label_text:
            return "status_green"
        if "discolored" in label_text or "discolour" in label_text:
            return "status_yellow"
        if "diseased" in label_text or "disease" in label_text:
            return "status_orange"
        return ""

    def _load_data(self):
        try:
            self.all_rows = csv_utils.read_sensor_csv_rows(self.csv_path)
        except Exception as exc:
            messagebox.showerror("Read Error", f"Failed to read CSV: {exc}", parent=self.win)
            self.all_rows = []
        self._apply_filters()

    def _row_matches_filter(self, row, filter_value):
        if filter_value == "All":
            return True
        prediction = str(row.get("ml_prediction", "")).strip().lower()
        if filter_value == "Unclassified":
            return prediction in ("", "unclassified", "no clear kangkong")
        return filter_value.lower() in prediction

    def _row_matches_search(self, row, search_text):
        if not search_text:
            return True
        search_text = search_text.strip().lower()
        return any(search_text in str(value).lower() for value in row.values())

    def _apply_filters(self):
        search_text = self.search_var.get()
        filter_value = self.filter_var.get() if hasattr(self, "filter_var") else "All"

        # Keep original (index, row) pairs so deletes/exports can always
        # trace back to the true row in self.all_rows.
        indexed_rows = list(enumerate(self.all_rows))
        filtered = [
            (idx, row) for idx, row in indexed_rows
            if self._row_matches_filter(row, filter_value) and self._row_matches_search(row, search_text)
        ]
        # Most recent entries first.
        filtered.reverse()
        self._populate_tree(filtered)

    def _populate_tree(self, indexed_rows):
        self.tree.delete(*self.tree.get_children())
        self.row_lookup = {}
        self.displayed_rows = indexed_rows

        for display_pos, (original_idx, row) in enumerate(indexed_rows):
            status_tag = self._row_status_tag(row.get("ml_prediction", ""))
            tag = status_tag if status_tag else ("row_even" if display_pos % 2 == 0 else "row_odd")
            iid = self.tree.insert("", tk.END, values=[row.get(col, "") for col in CSV_COLUMNS], tags=(tag,))
            self.row_lookup[iid] = original_idx

        total = len(self.all_rows)
        shown = len(indexed_rows)
        if shown == total:
            self.lbl_history_count.config(text=f"({total} records)")
        else:
            self.lbl_history_count.config(text=f"({shown} of {total} records)")

    def _selected_original_indices(self):
        iids = self.tree.selection()
        return sorted(self.row_lookup[iid] for iid in iids if iid in self.row_lookup)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _resolve_image_path(self, value):
        if not value:
            return None
        value = str(value).strip()
        if not value:
            return None

        candidates = [value]
        if not os.path.isabs(value):
            if self.base_dir:
                candidates.append(os.path.join(self.base_dir, value))
            if self.image_dir:
                candidates.append(os.path.join(self.image_dir, value))
                candidates.append(os.path.join(self.image_dir, os.path.basename(value)))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    def _view_selected_image(self):
        indices = self._selected_original_indices()
        if not indices:
            messagebox.showinfo("View Image", "Select a row first.", parent=self.win)
            return
        if len(indices) > 1:
            messagebox.showinfo("View Image", "Select just one row to view its image.", parent=self.win)
            return

        row = self.all_rows[indices[0]]
        image_path = self._resolve_image_path(row.get("image_path"))
        if not image_path:
            messagebox.showwarning("View Image", "Image file could not be found on disk.", parent=self.win)
            return

        self._open_image_viewer(image_path, row)

    def _view_selected_analysis(self):
        indices = self._selected_original_indices()
        if not indices:
            messagebox.showinfo("View Analysis", "Select a row first.", parent=self.win)
            return
        if len(indices) > 1:
            messagebox.showinfo("View Analysis", "Select just one row to view its analysis.", parent=self.win)
            return

        row = self.all_rows[indices[0]]
        self._open_analysis_viewer(row)

    def _prediction_accent_color(self, label: str) -> str:
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

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def _build_analysis_chart(self, row):
        from PIL import ImageDraw, ImageFont

        label = row.get("ml_prediction", "--") or "--"
        accent = self._prediction_accent_color(label)

        names = ["Healthy", "Discolored", "Diseased"]
        prob_keys = {"Healthy": "prob_healthy", "Discolored": "prob_discolored", "Diseased": "prob_diseased"}
        bar_colors = [COLORS["green"], COLORS["yellow"], COLORS["orange"]]
        values = [self._safe_float(row.get(prob_keys[name])) for name in names]

        w, h = 640, 420
        img = Image.new("RGB", (w, h), COLORS["bg"])
        draw = ImageDraw.Draw(img)

        def rrect(xy, radius, **kw):
            try:
                draw.rounded_rectangle(xy, radius=radius, **kw)
            except Exception:
                draw.rectangle(xy, **kw)

        try:
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
            font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
            font_text = ImageFont.truetype("DejaVuSans.ttf", 12)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
        except Exception:
            font_title = font_bold = font_text = font_small = ImageFont.load_default()

        # Header bar
        rrect((14, 12, w - 14, 46), 10, fill=COLORS["bg_card"])
        draw.text((26, 20), "Saved Analysis", fill=COLORS["text"], font=font_title)

        # Result badge
        badge_text = str(label)
        badge_w = draw.textlength(badge_text, font=font_bold) + 24
        rrect((w - 14 - badge_w, 58, w - 14, 84), 12, fill=accent)
        draw.text((w - 14 - badge_w + 12, 63), badge_text, fill="#10151f", font=font_bold)

        # Readings card
        info_top, info_bottom = 58, 178
        info_right = w - 26 - badge_w
        rrect((14, info_top, info_right, info_bottom), 10, fill=COLORS["bg_card"])
        draw.text((26, info_top + 8), f"Timestamp: {row.get('timestamp', '--')}", fill=COLORS["text_dim"], font=font_small)
        draw.text((26, info_top + 32), f"Light: {row.get('light_lux', '--')} lux", fill=COLORS["text"], font=font_text)
        draw.text((26, info_top + 54), f"Water Temp: {row.get('water_temp_c', '--')} \u00b0C", fill=COLORS["text"], font=font_text)
        draw.text(
            (26, info_top + 76),
            f"pH Level: {row.get('ph_level', '--')}  (V: {row.get('ph_voltage', '--')})",
            fill=COLORS["text"], font=font_text,
        )

        # Captured-image thumbnail card
        thumb_top = info_bottom + 14
        rrect((14, thumb_top, 234, h - 14), 10, fill=COLORS["bg_card"])
        draw.text((26, thumb_top + 8), "Captured Image", fill=COLORS["text"], font=font_bold)
        image_path = self._resolve_image_path(row.get("image_path"))
        if image_path:
            try:
                original = Image.open(image_path).convert("RGB")
                original.thumbnail((196, h - thumb_top - 44), Image.LANCZOS)
                img.paste(original, (26, thumb_top + 32))
            except Exception:
                draw.text((26, thumb_top + 32), "Could not load image", fill=COLORS["text_dim"], font=font_small)
        else:
            draw.text((26, thumb_top + 32), "Image not found on disk", fill=COLORS["text_dim"], font=font_small)

        # Chart card
        chart_left, chart_top = 246, thumb_top
        chart_right, chart_bottom = w - 14, h - 14
        rrect((chart_left, chart_top, chart_right, chart_bottom), 10, fill=COLORS["bg_card"])
        draw.text((chart_left + 12, chart_top + 8), "Prediction Confidence", fill=COLORS["text"], font=font_bold)

        plot_x0, plot_y0 = chart_left + 20, chart_top + 40
        plot_x1, plot_y1 = chart_right - 20, chart_bottom - 34

        for frac in (0.25, 0.5, 0.75, 1.0):
            gy = plot_y1 - int((plot_y1 - plot_y0) * frac)
            draw.line((plot_x0, gy, plot_x1, gy), fill=COLORS["border"], width=1)

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

    def _open_analysis_viewer(self, row):
        try:
            chart_img = self._build_analysis_chart(row)
        except Exception as exc:
            messagebox.showerror("View Analysis", f"Failed to build analysis view:\n{exc}", parent=self.win)
            return

        viewer = tk.Toplevel(self.win)
        viewer.title("Saved Analysis")
        viewer.configure(bg=COLORS["bg"])
        viewer.transient(self.win)

        header = tk.Frame(viewer, bg=COLORS["bg_panel"])
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=f"{row.get('timestamp', '')}  •  {row.get('ml_prediction', '')}",
            font=("Helvetica", 11, "bold"), bg=COLORS["bg_panel"], fg=COLORS["text"],
        ).pack(side=tk.LEFT, padx=12, pady=8)
        make_button(
            header, "CLOSE", COLORS["red"], COLORS["red_hover"], viewer.destroy,
            font=("Helvetica", 10, "bold"),
        ).pack(side=tk.RIGHT, padx=12, pady=6)

        imgtk = ImageTk.PhotoImage(chart_img)
        img_label = tk.Label(viewer, image=imgtk, bg=COLORS["bg"])
        img_label.image = imgtk  # keep a reference
        img_label.pack(padx=10, pady=10)

        viewer.update_idletasks()
        viewer.lift()
        viewer.focus_force()
        viewer.grab_set()
        viewer.bind("<Escape>", lambda _e: viewer.destroy())

    def _open_image_viewer(self, image_path, row):
        try:
            pil_img = Image.open(image_path)
        except Exception as exc:
            messagebox.showerror("View Image", f"Failed to open image:\n{exc}", parent=self.win)
            return

        viewer = tk.Toplevel(self.win)
        viewer.title(os.path.basename(image_path))
        viewer.configure(bg=COLORS["bg"])
        viewer.transient(self.win)

        header = tk.Frame(viewer, bg=COLORS["bg_panel"])
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=f"{row.get('timestamp', '')}  •  {row.get('ml_prediction', '')}",
            font=("Helvetica", 11, "bold"), bg=COLORS["bg_panel"], fg=COLORS["text"],
        ).pack(side=tk.LEFT, padx=12, pady=8)
        make_button(
            header, "CLOSE", COLORS["red"], COLORS["red_hover"], viewer.destroy,
            font=("Helvetica", 10, "bold"),
        ).pack(side=tk.RIGHT, padx=12, pady=6)

        max_w, max_h = 640, 420
        display_img = pil_img.copy().convert("RGB")
        display_img.thumbnail((max_w, max_h), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(display_img)

        img_label = tk.Label(viewer, image=imgtk, bg=COLORS["bg"])
        img_label.image = imgtk  # keep a reference
        img_label.pack(padx=10, pady=10)

        viewer.update_idletasks()
        viewer.lift()
        viewer.focus_force()
        viewer.grab_set()
        viewer.bind("<Escape>", lambda _e: viewer.destroy())

    def _delete_selected(self):
        indices = self._selected_original_indices()
        if not indices:
            messagebox.showinfo("Delete Selected", "Select at least one row first.", parent=self.win)
            return

        count = len(indices)
        confirm = messagebox.askyesno(
            "Delete Selected",
            f"Delete {count} selected log entr{'y' if count == 1 else 'ies'}? This cannot be undone.",
            parent=self.win,
        )
        if not confirm:
            return

        also_delete_images = messagebox.askyesno(
            "Delete Images Too?",
            "Also delete the associated captured image file(s) from disk?",
            parent=self.win,
        )

        indices_set = set(indices)
        rows_to_delete = [self.all_rows[i] for i in indices]

        # Rebuild self.all_rows without the deleted indices.
        self.all_rows = [row for i, row in enumerate(self.all_rows) if i not in indices_set]

        try:
            csv_utils.write_sensor_csv_rows(self.csv_path, self.all_rows)
        except Exception as exc:
            messagebox.showerror("Delete Error", f"Failed to update CSV file:\n{exc}", parent=self.win)
            return

        if also_delete_images:
            for row in rows_to_delete:
                image_path = self._resolve_image_path(row.get("image_path"))
                if image_path:
                    try:
                        os.remove(image_path)
                    except Exception as exc:
                        print(f"Could not delete image {image_path}: {exc}")

        self._apply_filters()
        messagebox.showinfo("Deleted", f"{count} entr{'y' if count == 1 else 'ies'} deleted.", parent=self.win)

    def _export_csv(self):
        if not self.displayed_rows:
            messagebox.showinfo("Export CSV", "There is nothing to export with the current search/filter.", parent=self.win)
            return

        dest_path = filedialog.asksaveasfilename(
            parent=self.win,
            title="Export History CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="kangkong_history_export.csv",
        )
        if not dest_path:
            return

        rows_in_display_order = [row for _idx, row in self.displayed_rows]
        try:
            csv_utils.write_sensor_csv_rows(dest_path, rows_in_display_order)
        except Exception as exc:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{exc}", parent=self.win)
            return

        messagebox.showinfo("Export CSV", f"Exported {len(rows_in_display_order)} rows to:\n{dest_path}", parent=self.win)

    def _refresh(self):
        self._load_data()

    def _go_back_to_main(self):
        self.win.destroy()
        self.parent_root.deiconify()
        self.parent_root.lift()
        self.parent_root.focus_force()


def open_history_window(parent_root, csv_path: str, image_dir: str, base_dir: str = None):
    """Convenience entry point mirroring the old view_csv_data() function."""
    return HistoryWindow(parent_root, csv_path, image_dir, base_dir)
