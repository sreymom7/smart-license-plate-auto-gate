"""
dashboard/app.py

Simple Flask dashboard for the Smart License Plate Recognition & Auto Gate
System. Shows live access logs, current gate status, and lets you manage
the authorized-plates whitelist. Includes a "Simulate Scan" box so you can
test the full match -> gate -> log flow without a camera or hardware.

Run with:
    cd dashboard
    python app.py
Then open http://localhost:5000
"""

import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from database.db import (
    load_authorized_plates, add_plate, remove_plate,
    is_authorized, log_entry, get_recent_logs
)
from gate.gate_controller import GateController
from plate_scanner import scan_image_bytes

app = Flask(__name__)
gate = GateController(open_seconds=5, cooldown_seconds=10)


@app.route("/")
def index():
    return render_template(
        "index.html",
        plates=load_authorized_plates(),
        logs=get_recent_logs(50),
        gate_open=gate.is_open,
    )


# ------------------------------------------------------------
# Status / logs (polled by the front end every few seconds)
# ------------------------------------------------------------
@app.route("/api/status")
def api_status():
    return jsonify({
        "gate_open": gate.is_open,
        "logs": get_recent_logs(50),
    })


# ------------------------------------------------------------
# Whitelist management
# ------------------------------------------------------------
@app.route("/api/plates")
def api_plates():
    return jsonify(load_authorized_plates())


@app.route("/api/plates/add", methods=["POST"])
def api_add_plate():
    data = request.get_json(force=True)
    record = add_plate(
        plate=data.get("plate", ""),
        owner=data.get("owner", ""),
        vehicle=data.get("vehicle", ""),
        active=data.get("active", True),
    )
    return jsonify({"success": True, "record": record})


@app.route("/api/plates/delete", methods=["POST"])
def api_delete_plate():
    data = request.get_json(force=True)
    removed = remove_plate(data.get("plate", ""))
    return jsonify({"success": removed})


# ------------------------------------------------------------
# Manual scan simulator - lets you test the pipeline without a
# camera by typing in a plate number as if OCR had just read it.
# ------------------------------------------------------------
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True)
    plate_text = data.get("plate", "").strip()

    if not plate_text:
        return jsonify({"success": False, "error": "No plate text provided"}), 400

    authorized, matched_record, score = is_authorized(plate_text)
    status = "AUTHORIZED" if authorized else "DENIED"

    gate_action = "-"
    if authorized:
        gate_action = gate.trigger(matched_record["plate"])

    log_entry(
        plate_detected=plate_text,
        status=status,
        confidence=1.0,
        matched_record=matched_record,
        gate_action=gate_action,
    )

    return jsonify({
        "success": True,
        "status": status,
        "matched": matched_record,
        "match_score": round(score, 2),
        "gate_action": gate_action,
        "gate_open": gate.is_open,
    })


# ------------------------------------------------------------
# Real image scan - runs actual YOLO + OCR on an uploaded photo,
# then matches / gates / logs each plate it finds
# ------------------------------------------------------------
@app.route("/api/scan_image", methods=["POST"])
def api_scan_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"success": False, "error": "Empty file"}), 400

    try:
        detections, annotated_b64 = scan_image_bytes(image_bytes)
    except FileNotFoundError as e:
        # Model hasn't been trained / weights not in place yet
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Scan failed: {e}"}), 500

    if not detections:
        return jsonify({
            "success": True,
            "detections": [],
            "annotated_image": annotated_b64,
            "gate_open": gate.is_open,
            "message": "No plate detected in this image.",
        })

    results = []
    for det in detections:
        plate_text = det["plate_text"]

        if not plate_text:
            results.append({**det, "status": "UNREADABLE", "gate_action": "-"})
            continue

        authorized, matched_record, score = is_authorized(plate_text)
        status = "AUTHORIZED" if authorized else "DENIED"

        gate_action = "-"
        if authorized:
            gate_action = gate.trigger(matched_record["plate"])

        log_entry(
            plate_detected=plate_text,
            status=status,
            confidence=det["detector_confidence"],
            matched_record=matched_record,
            gate_action=gate_action,
        )

        results.append({
            **det,
            "status": status,
            "gate_action": gate_action,
            "match_score": round(score, 2),
        })

    return jsonify({
        "success": True,
        "detections": results,
        "annotated_image": annotated_b64,
        "gate_open": gate.is_open,
    })


if __name__ == "__main__":
    # use_reloader=False matters here: the reloader watches the whole
    # project folder, and every log/whitelist write would otherwise be
    # mistaken for a code change and restart the server mid-request.
    app.run(debug=True, use_reloader=False, port=5000)
