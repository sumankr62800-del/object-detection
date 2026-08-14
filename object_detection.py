import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Map fixed random colors for classes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype="uint8")


def draw_custom_boxes(frame, results):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_text = f"{model.names[cls_id]} {conf:.2f}"
            color = [int(c) for c in colors[cls_id]]

            # Bounding box & label text background
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x1, max(y1 - h - 10, 0)), (x1 + w + 10, y1), color, -1
            )
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    return frame


cap = cv2.VideoCapture(0)

# 2. Initialize Resizable Window
window_name = "YOLOv8 Detection (Press 'F' for Fullscreen)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Fullscreen state tracker
is_fullscreen = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.35, verbose=False)
    frame = draw_custom_boxes(frame, results)

    cv2.imshow(window_name, frame)

    # 3. Handle Key Presses
    key = cv2.waitKey(1) & 0xFF

    # Toggle full screen on 'f' or 'F' keypress
    if key in (ord("f"), ord("F")):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )

    # Exit program on 'q' or ESC
    elif key in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()