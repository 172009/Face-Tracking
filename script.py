import cv2
from ultralytics import YOLO

model = YOLO("yolov12n-face.pt")  #yolo-version 12

#webcam
webcam = cv2.VideoCapture(0)

#resize (in work)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Detect face
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
        # Lấy tọa độ hộp
            x1, y1, x2, y2 = map(int, box.xyxy[0])

        
            cls_id = int(box.cls[0])     # AI debug
            label = model.names[cls_id]  # AI debug

            conf = float(box.conf[0])    # AI debug
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)  # box
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("YOLO_FD -- (Currently In Development)", frame)

    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

webcam.release()
cv2.destroyAllWindows()
