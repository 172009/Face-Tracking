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
    new_name = simpledialog.askstring("Input", "Import Name: ") # ask for name
    if not new_name: # if that not new_name, return nothing
        return

    image_paths = filedialog.askopenfilenames(
        title="Choose this picture: ",
        filetypes=[("Images files", "*.jpg *.jpeg *.png")], 
    ) # import the picture contain face
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
                known_face_embeddings.append(embedding) # add new embedding face after import picture
                known_face_names.append(new_name) # add new name after import the name
                added_count += 1 
                print(f"Added {os.path.basename(image_path)} for {new_name}")

        else:
            print(f"Not found images in picture {os.path.basename(image_path)}")

    np.save("known_face_embeddings.npy", np.array(known_face_embeddings)) # database, maybe should change in future to get more security
    np.save("known_face_names.npy", np.array(known_face_names)) # same, the database save the name, could change to security
    print(f"All images of {new_name} has been save to database i guess")


def init_insightface(): #very simple init insightface def, this is the main brain, where it will using ai model to detect the face
    try:
        model_directory = resource_path("buffalo_l")
        app = FaceAnalysis(
                name="buffalo_l", #model using for project, the strongest model i known
                root=model_directory,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"], #what the hardware insightface will using to, GPU or CPU ( recommend using GPU for more frame )
            )
        app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id = 0 mean using GPU ( if available ), -1 mean CPU , 1 mean another GPU if you have
        print("InsightFace init success!!") #  if it init succesfully
        return app
    except Exception as e:
        print(f"Error while trying init insightface {e}") 
        return None


app = init_insightface()
if app is None:
    print("InsightFace init failed. Exiting...") # if not init complete, it gone
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
        # get insight face and paint the box to the face if it found one and of course - yolo will doing better than insight face since it used to detect face
        faces = app.get(frame)
        for face in faces:
            if not hasattr(face, "bbox") or face.bbox is None:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int) # the box direction ( x, y, weight, height )
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
                if sims[best_match] > 0.5: # the value that match with the images recommend about ( 0.3 ~ 0.6 ), but i choose 0.5 because it work pretties well
                    name = known_face_names[best_match]
                    cv2.putText(frame, f"cls_id: {name}", (x1, y2 + 25), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2, ) #if it found correct entity when user import name and provide picutre
                    print(f"Found target!!!")
                else:
                    cv2.putText( frame, "cls_id: unknown", (x1, y2 + 25), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2,) # if not found, return value unknow
            else:
                cv2.putText(frame, "cls_id: unknown", (x1, y2 + 25), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2, )
        # using yolo to detect new face ( AI Debug )
        yolo_results = face_model(frame, device=0)
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = face_model.names[cls_id] # default is face
                conf = box.conf[0].item() # confident
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText( frame, f"{label} {conf:.2f}", (x1, y1 - 10),  cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2, ) #print the label ( default model is "face") and the conf is mean confident - how much truth point to that face
    # very simple fps calculator
    curr_time = time.time()
    fps = 1 / (curr_time - time_prev) if time_prev != 0 else 0
    time_prev = curr_time

    cv2.putText(
        frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_ITALIC, 0.9, (255, 0, 255), 2
    )
    # window name called
    cv2.imshow("YOLO Detect - Still In Development", frame)

    key = cv2.waitKey(1) & 0xFF
    # import picture and get the file
    if key == ord("p"):
        Tk().withdraw()
        image_path = filedialog.askopenfilenames(
            title="Choose picture to load",
            filetypes=[("Images or Video files", "*.jpg *.jpeg *.png *.gif *.mp4")], # type files
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
    #add new person name and picture
    elif key == ord("a"):
        add_new_person(app)

# clear
if webcam.isOpened():
    webcam.release()
# hands.close()
cv2.destroyAllWindows()

