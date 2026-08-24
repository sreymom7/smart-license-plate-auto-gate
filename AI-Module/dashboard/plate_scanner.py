"""
dashboard/plate_scanner.py

Runs the REAL detection + OCR pipeline (same logic as app/main_upload_img.py)
on an uploaded image, for use by the dashboard's "Scan Image" feature.

Model and OCR reader are loaded once and cached, so repeated scans are fast.
"""

import base64
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "plate_detection" / "runs" / "detect" / "train" / "weights" / "best.pt"

_model = None
_reader = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                f"Train it first with models/plate_detection/tranning.py, "
                f"then make sure runs/detect/train/weights/best.pt exists."
            )
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


def _load_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'])
    return _reader


def scan_image_bytes(image_bytes, conf=0.5):
    """
    Run detection + OCR on raw image bytes (e.g. from a file upload).

    Returns:
        (detections, annotated_image_base64)

    detections: list of dicts, one per plate found:
        {
          "bbox": [x1, y1, x2, y2],
          "detector_confidence": float,
          "plate_text": str,       # "" if OCR found nothing
          "ocr_confidence": float,
        }
    """
    model = _load_model()
    reader = _load_reader()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded file as an image.")

    result_image = image.copy()
    results = model(image, conf=conf)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_conf = float(box.conf[0])

            pad = 10
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(image.shape[1], x2 + pad)
            cy2 = min(image.shape[0], y2 + pad)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # OCR preprocess (same as main_upload_img.py)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            texts = reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

            best_text, best_score = "", 0.0
            for t in texts:
                text = t[1].upper().strip()
                score = float(t[2])
                if score > best_score:
                    best_text, best_score = text, score

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "detector_confidence": round(det_conf, 3),
                "plate_text": best_text,
                "ocr_confidence": round(best_score, 3),
            })

            # Draw box + label on the annotated copy
            color = (0, 255, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            label = f"{best_text or '?'} | {det_conf:.2f}"
            cv2.putText(
                result_image, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

    ok, buf = cv2.imencode('.jpg', result_image)
    annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8') if ok else None

    return detections, annotated_b64
