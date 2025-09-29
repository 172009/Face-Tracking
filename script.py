from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os, sys
from ultralytics import YOLO
import time
from tkinter import Tk, simpledialog, filedialog

# (resource use to load exe files)
def resource_path(relative_path):
    if getattr(sys, "_MEIPASS", False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


face_model_path = resource_path("yolov12n-face.pt")

face_model = YOLO(face_model_path)

current_model = face_model
current_label = "face"


# started with webcam
webcam = cv2.VideoCapture(0)
use_webcam = True


detect_enable = True


# load data if exists ( AI Debug)
if os.path.exists("known_face_embeddings.npy"):
    known_face_embeddings = list(
        np.load("known_face_embeddings.npy", allow_pickle=True)
    )
else:
    known_face_embeddings = []

if os.path.exists("known_face_names.npy"):
    known_face_names = list(np.load("known_face_names.npy", allow_pickle=True))
else:
    known_face_names = []


# add new people
def add_new_person(app):
    Tk().withdraw()  # new window
    new_name = simpledialog.askstring("Input", "Import Name: ")
    if not new_name:
        return

    image_paths = filedialog.askopenfilenames(
        title="Choose this picture: ",
        filetypes=[("Images files", "*.jpg *.jpeg *.png")],
    )
    if not image_paths:
        return

    added_count = 0
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read {image_path}")
            continue

        faces = app.get(img)
        if len(faces) > 0:
            for face in faces:
                embedding = face.embedding
                known_face_embeddings.append(embedding)
                known_face_names.append(new_name)
                added_count += 1
                print(f"Added {os.path.basename(image_path)} for {new_name}")

        else:
            print(f"Not found images in picture {os.path.basename(image_path)}")

    np.save("known_face_embeddings.npy", np.array(known_face_embeddings))
    np.save("known_face_names.npy", np.array(known_face_names))
    print(f"All images of {new_name} has been save to database i guess")


def init_insightface():
    try:
        model_directory = resource_path("buffalo_l")
        app = FaceAnalysis(
                name="buffalo_l",
                root=model_directory,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace init success!!")
        return app
    except Exception as e:
        print(f"Error while trying to download {e}")
        return None


app = init_insightface()
if app is None:
    print("InsightFace init failed. Exiting...")
    sys.exit(1)
# for fps
time_prev = 0


image_path = None  # for save images path
use_webcam = True  # for set default is camera
while True:

    if use_webcam:
        ret, frame = webcam.read()
        if not ret:
            print("retrying...")
            break
    else:
        if image_path:
            frame = cv2.imread(image_path[0])
        else:
            continue
    if detect_enable and app is not None:
        # using insightface to detect
        faces = app.get(frame)
        for face in faces:
            if not hasattr(face, "bbox") or face.bbox is None:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            if not hasattr(face, "embedding") or face.embedding is None:
                continue
            embedding = face.embedding
            sims = [
                np.dot(embedding, ref)
                / (np.linalg.norm(embedding) * np.linalg.norm(ref))
                for ref in known_face_embeddings
            ]
            if len(sims) > 0:
                best_match = np.argmax(sims)
                if sims[best_match] > 0.5:
                    name = known_face_names[best_match]
                    cv2.putText(
                        frame,
                        f"cls_id: {name}",
                        (x1, y2 + 25),
                        cv2.FONT_ITALIC,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    print(f"Found target!!!")
                else:
                    cv2.putText(
                        frame,
                        "cls_id: unknown",
                        (x1, y2 + 25),
                        cv2.FONT_ITALIC,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "cls_id: unknown",
                    (x1, y2 + 25),
                    cv2.FONT_ITALIC,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        # using yolo to detect new face ( AI Debug instead using results -> yolo_result)
        yolo_results = face_model(frame, device=0)
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = face_model.names[cls_id]
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_ITALIC,
                    0.8,
                    (255, 255, 255),
                    2,
                )
    curr_time = time.time()
    fps = 1 / (curr_time - time_prev) if time_prev != 0 else 0
    time_prev = curr_time

    cv2.putText(
        frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_ITALIC, 0.9, (255, 0, 255), 2
    )

    cv2.imshow("YOLO Detect - Still In Development", frame)

    key = cv2.waitKey(1) & 0xFF
    # change picture
    if key == ord("p"):
        Tk().withdraw()
        image_path = filedialog.askopenfilenames(
            title="Choose picture to load",
            filetypes=[("Images or Video files", "*.jpg *.jpeg *.png *.gif *.mp4")],
        )
        if image_path:
            image_path = list(image_path)
            use_webcam = False
            webcam.release()
    # change camera
    elif key == ord("c"):
        if not use_webcam:
            webcam = cv2.VideoCapture(0)
            use_webcam = True
    elif key == ord("i"):
        detect_enable = not detect_enable
    # exits
    elif key == ord("e"):
        break
    elif key == ord("a"):
        add_new_person(app)

# clear
if webcam.isOpened():
    webcam.release()
# hands.close()
cv2.destroyAllWindows()
