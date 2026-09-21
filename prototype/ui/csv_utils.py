"""
Shared helpers for reading/writing the sensor + ML prediction CSV log.

Used by ui_main.py (appending a new row after SAVE) and by
history_window.py (reading all rows for the History tab, and rewriting
the file after a delete).
"""

import csv
import os

from theme import CSV_COLUMNS, CSV_COLUMN_ALIASES


def csv_key_lookup(record: dict, aliases: list) -> str:
    normalized_record = {
        str(key).strip().lower().replace(" ", "_"): value
        for key, value in record.items()
    }

    for alias in aliases:
        key = str(alias).strip().lower().replace(" ", "_")
        if key in normalized_record:
            return normalized_record[key]

    return ""


def normalize_csv_record(record: dict) -> dict:
    return {
        column: csv_key_lookup(record, CSV_COLUMN_ALIASES[column])
        for column in CSV_COLUMNS
    }


def read_sensor_csv_rows(csv_path: str):
    """Reads every row of the CSV log into a list of normalized dicts."""
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []

        rows = []
        for row in reader:
            raw_record = {}
            for index, column_name in enumerate(header):
                raw_record[column_name] = row[index] if index < len(row) else ""
            rows.append(normalize_csv_record(raw_record))

        return rows


def write_sensor_csv_rows(csv_path: str, rows) -> None:
    """Overwrites the CSV log with exactly the given rows (used after a
    delete from the History tab)."""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def ensure_sensor_csv_format(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        return

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        current_header = next(reader, None)

    if current_header == CSV_COLUMNS:
        return

    rows = read_sensor_csv_rows(csv_path)
    write_sensor_csv_rows(csv_path, rows)


def append_sensor_csv(
    csv_path: str,
    timestamp: str,
    light_lux: float,
    water_temp_c: float,
    ph_level: float,
    ph_voltage: float,
    ml_prediction: str,
    prob_healthy: float,
    prob_discolored: float,
    prob_diseased: float,
    image_path: str,
) -> None:
    ensure_sensor_csv_format(csv_path)

    row = {
        "timestamp": timestamp,
        "light_lux": round(light_lux, 2),
        "water_temp_c": round(water_temp_c, 2),
        "ph_level": round(ph_level, 2),
        "ph_voltage": round(ph_voltage, 3),
        "ml_prediction": ml_prediction,
        "prob_healthy": round(prob_healthy, 4),
        "prob_discolored": round(prob_discolored, 4),
        "prob_diseased": round(prob_diseased, 4),
        "image_path": image_path,
    }

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
