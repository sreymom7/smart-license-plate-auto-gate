from ultralytics import YOLO

model = YOLO("../models/plate_detection/runs/detect/train/weights/best.pt")

def detect(image, conf=0.5):

    results = model(image, conf=conf)

    plates = []

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plate_crop = image[y1:y2, x1:x2]

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "image": plate_crop
            })

    return plates