"""
database/db.py

Shared module for:
  1. Loading / saving the authorized plates whitelist (JSON)
  2. Matching a recognized (OCR'd) plate against that whitelist,
     tolerant of small OCR mistakes
  3. Logging every access attempt to a CSV file

Import this from anywhere in the project:
    from database.db import is_authorized, log_entry, add_plate, remove_plate
"""

import json
import csv
import os
import difflib
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Paths (resolved relative to THIS file, so it works no matter where
# the calling script is run from)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PLATES_FILE = BASE_DIR / "authorized_plates.json"
LOG_FILE = BASE_DIR / "access_log.csv"

LOG_HEADERS = ["timestamp", "plate_detected", "matched_plate", "owner",
               "status", "confidence", "gate_action"]


# ------------------------------------------------------------------
# Whitelist management
# ------------------------------------------------------------------
def load_authorized_plates():
    """Return the list of authorized-plate records (dicts)."""
    if not PLATES_FILE.exists():
        return []
    with open(PLATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_authorized_plates(plates):
    with open(PLATES_FILE, "w", encoding="utf-8") as f:
        json.dump(plates, f, indent=2, ensure_ascii=False)


def add_plate(plate, owner="", vehicle="", active=True):
    plates = load_authorized_plates()
    plate = normalize_plate(plate)

    for p in plates:
        if normalize_plate(p["plate"]) == plate:
            p.update({"owner": owner, "vehicle": vehicle, "active": active})
            save_authorized_plates(plates)
            return p

    record = {"plate": plate, "owner": owner, "vehicle": vehicle, "active": active}
    plates.append(record)
    save_authorized_plates(plates)
    return record


def remove_plate(plate):
    plates = load_authorized_plates()
    plate = normalize_plate(plate)
    new_plates = [p for p in plates if normalize_plate(p["plate"]) != plate]
    save_authorized_plates(new_plates)
    return len(plates) != len(new_plates)


# ------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------
def normalize_plate(text):
    """Uppercase, strip spaces, drop characters that aren't A-Z 0-9 or '-'."""
    text = text.upper().strip()
    return "".join(ch for ch in text if ch.isalnum() or ch == "-")


def is_authorized(plate_text, fuzzy_threshold=0.85):
    """
    Check a recognized plate string against the whitelist.

    Uses exact match first; falls back to fuzzy matching so a single
    OCR misread (e.g. '2A-1284' vs '2A-1234') doesn't wrongly deny
    someone who IS on the list. Returns:

        (authorized: bool, matched_record: dict|None, score: float)
    """
    plate_text = normalize_plate(plate_text)
    plates = load_authorized_plates()
    active_plates = [p for p in plates if p.get("active", True)]

    # 1. Exact match
    for p in active_plates:
        if normalize_plate(p["plate"]) == plate_text:
            return True, p, 1.0

    # 2. Fuzzy match (guards against OCR noise, e.g. 0/O, 1/I confusion)
    best_record, best_score = None, 0.0
    for p in active_plates:
        score = difflib.SequenceMatcher(
            None, plate_text, normalize_plate(p["plate"])
        ).ratio()
        if score > best_score:
            best_score, best_record = score, p

    if best_record and best_score >= fuzzy_threshold:
        return True, best_record, best_score

    return False, None, best_score


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
def _ensure_log_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADERS)


def log_entry(plate_detected, status, confidence, matched_record=None, gate_action=""):
    """
    Append one row to access_log.csv.

    status: "AUTHORIZED" or "DENIED"
    gate_action: "OPENED" / "STAYED CLOSED" / "" etc.
    """
    _ensure_log_file()
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        plate_detected,
        matched_record["plate"] if matched_record else "",
        matched_record["owner"] if matched_record else "",
        status,
        f"{confidence:.2f}",
        gate_action,
    ]
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def get_recent_logs(limit=50):
    """Return the most recent log rows (list of dicts), newest first."""
    _ensure_log_file()
    with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return list(reversed(rows))[:limit]
