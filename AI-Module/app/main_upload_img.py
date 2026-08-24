import cv2
import easyocr
import os
from detector import detect
import re

# OCR
reader = easyocr.Reader(['en'])

# Create output folders
os.makedirs("../output", exist_ok=True)

# Load image
image = cv2.imread("../data/test4.jpg")

if image is None:
    print("Image not found")
    exit()

# Keep original image
result_image = image.copy()

# Detect plates
plates = detect(image)

if len(plates) == 0:
    print("No plate detected")

for i, plate in enumerate(plates):

    x1 = plate["x1"]
    y1 = plate["y1"]
    x2 = plate["x2"]
    y2 = plate["y2"]

    conf = plate["confidence"]

    # Crop plate
    plate_img = image[y1:y2, x1:x2]


    # ==========================
    # OCR PREPROCESS
    # ==========================
    gray = cv2.cvtColor(
        plate_img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # OCR
    texts = reader.readtext(
        gray,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    )

    plate_text = ""

    if texts:
        plate_text = texts[0][1].strip()
        print(f"Plate: {plate_text}")
    else:
        print("Plate: Not detected")

    print(f"Confidence: {conf:.2f}")

    # ==========================
    # CCTV STYLE BOX
    # ==========================
    color = (0, 255, 0)
    thickness = 2
    line = 25

    cv2.line(result_image, (x1, y1), (x1 + line, y1), color, thickness)
    cv2.line(result_image, (x1, y1), (x1, y1 + line), color, thickness)

    cv2.line(result_image, (x2, y1), (x2 - line, y1), color, thickness)
    cv2.line(result_image, (x2, y1), (x2, y1 + line), color, thickness)

    cv2.line(result_image, (x1, y2), (x1 + line, y2), color, thickness)
    cv2.line(result_image, (x1, y2), (x1, y2 - line), color, thickness)

    cv2.line(result_image, (x2, y2), (x2 - line, y2), color, thickness)
    cv2.line(result_image, (x2, y2), (x2, y2 - line), color, thickness)

    # ==========================
    # LABEL
    # ==========================
    label = f"{plate_text} | {conf:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    (w, h), _ = cv2.getTextSize(
        label,
        font,
        font_scale,
        font_thickness
    )

    cv2.rectangle(
        result_image,
        (x1, y1 - h - 15),
        (x1 + w + 10, y1),
        (0, 180, 0),
        -1
    )

    cv2.putText(
        result_image,
        label,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness
    )

# Save result
cv2.imwrite("../output/result.jpg", result_image)