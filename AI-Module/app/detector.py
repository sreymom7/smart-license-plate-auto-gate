from ultralytics import YOLO

# Load model once when module starts
model = YOLO("../models/plate_detection/runs/detect/train/weights/best.pt")

def detect(image, conf=0.5):
    """
    image: OpenCV image (numpy array)

    Returns:
        list of detected plates
    """

    results = model(image, conf=conf)

    plates = []

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            plates.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(box.conf[0])
            })

    return plates