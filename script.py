import cv2
import os, sys
from ultralytics import YOLO

# (resource use to load exe files)
def resource_path(relative_path):
    if getattr(sys, '_MEIPASS', False): 
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_path = resource_path("yolov12n-face.pt")
model = YOLO(model_path)

# started with webcam
webcam = cv2.VideoCapture(0)
use_webcam = True  



image_path = None # for save images path
use_webcam = True # for set default is camera
while True:
    from tkinter import Tk, filedialog
    if use_webcam:
        ret, frame = webcam.read()
        if not ret:
            break
    else:
        if image_path:
            frame = cv2.imread(image_path)
        else:
            continue

    # Detect face
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("YOLO Detect", frame)

    key = cv2.waitKey(1) & 0xFF
    #change picture
    if key == ord("p"): 
        Tk().withdraw()
        image_path = filedialog.askopenfilename(
            title="Choose picture to load",
            filetypes=[("Images or Video files", "*.jpg *.jpeg *.png *.gif *.mp4")]
        )
        if image_path:
            use_webcam = False
            webcam.release()
    #change camera
    elif key == ord("c"): 
        if not use_webcam:  
            webcam = cv2.VideoCapture(0)
            use_webcam = True
    # exits
    elif key == ord("e"):
        break

# clear
if webcam.isOpened():
    webcam.release()
cv2.destroyAllWindows()


