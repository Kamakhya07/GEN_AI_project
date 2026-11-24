from ultralytics import YOLO

# Load the trained road damage detection model
MODEL_PATH = "best.pt"

model = YOLO(MODEL_PATH)

DAMAGE_CLASSES = [
    "Longitudinal_Crack",
    "Transverse_Crack",
    "Alligator_Crack",
    "Other_Corruption",
    "Pothole"
]

def detect_road_damage(image_path, conf=0.25):
    """
    Detect road damage in an image using trained YOLO model.

    Args:
        image_path: Path to the input image.
        conf: Confidence threshold.

    Returns:
        List of detections with class, confidence and bounding box.
    """
    results = model.predict(source=image_path, conf=conf, verbose=False)

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        detections.append({
            "class_id": cls_id,
            "class_name": DAMAGE_CLASSES[cls_id],
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })

    return detections


if __name__ == "__main__":
    image_path = "road_image.jpg"  # Replace with your test image

    detections = detect_road_damage(image_path)

    for detection in detections:
        print(detection)
