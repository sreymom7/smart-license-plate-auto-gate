import cv2
import easyocr
import os
from detector import detect
import re

# ==========================
# OCR
# ==========================
reader = easyocr.Reader(['en'])

# ==========================
# OUTPUT
# ==========================
os.makedirs("../output", exist_ok=True)

# ==========================
# CAMBODIAN PLATE VALIDATION
# ==========================
def is_valid_plate(text):
    text = text.upper().replace(" ", "")
    return re.match(r"^[1-5][A-Z]{1,2}-\d{4}$", text) is not None


# ==========================
# VIDEO INPUT
# ==========================
video_path = "../data/001.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

# ==========================
# VIDEO OUTPUT
# ==========================
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "../output/result.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

print("Processing video...")

# ==========================
# FRAME LOOP
# ==========================
frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # OPTIONAL: skip frames for speed (CCTV style)
    if frame_id % 2 != 0:
        continue

    result_image = frame.copy()

    plates = detect(frame)

    for plate in plates:

        x1, y1 = plate["x1"], plate["y1"]
        x2, y2 = plate["x2"], plate["y2"]
        conf = float(plate["confidence"])

        if conf < 0.5:
            continue

        # padding
        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # OCR preprocess
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        texts = reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

        best_text = ""
        best_score = 0

        for t in texts:
            text = t[1].upper().strip()
            score = float(t[2])

            if score > best_score:
                best_text = text
                best_score = score

        if not best_text:
            continue

        # validate plate
        if not is_valid_plate(best_text):
            continue

        print(f"Frame {frame_id} → Plate: {best_text}")

        # ==========================
        # DRAW CCTV BOX
        # ==========================
        color = (0, 255, 0)

        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        label = f"{best_text} | {conf:.2f}"

        cv2.putText(
            result_image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # save frame to video
    out.write(result_image)

    # OPTIONAL: show live preview
    cv2.imshow("CCTV Plate Detection", result_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("DONE → ../output/result.mp4")