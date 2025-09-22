# TRACKING ( CURRENTLY IN DEVELOPMENT)
# This script is written based on Python

# ![title](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh1.png)
- [Cà_Phê----MIN](https://www.youtube.com/watch?v=7m8ek8D9me0&list=RD7m8ek8D9me0&start_radio=1)
# ![title](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh2.png)
- [ Cứ_Chill_Thôi----Chillies](https://www.youtube.com/watch?v=LZN4I3K8SC0&list=RDLZN4I3K8SC0&start_radio=1)
# ![tittle](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh3.png)
- [Bad----Michael_Jackson](https://www.youtube.com/watch?v=dsUXAEzaC3Q&list=RDdsUXAEzaC3Q&start_radio=1)
# ![title](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh4.png)
- [Genshin_Impact_Nod_Krai_Performance----Genshin_Impact](https://www.youtube.com/watch?v=RuXa_yxZMGI&list=RDRuXa_yxZMGI&start_radio=1)




#### NOTE: we are trying to make it better to identify orther object
#### NOTE: This application using your webcam to detect, you can change value inside source code if you want


##### TODO LIST:

-[x] Detected Face
-[ ] Detected Logo School
-[ ] Compare faces and Flags Name


_(may add more in furture)_


# Getting started
### Requirements
- [Python-3.13] (https://www.python.org/downloads/)
```sh
pip install ultralytics
pip install opencv-python
```

### a) using from build
1. Head to release
2. Download latest release

### b) using from source code
1. Navigate to "Code" button
2. Download zip and extract
3. Open script.py


### BUILD 
```sh
git clone https://github.com/172009/Face-Tracking.git
cd Face-Tracking
```
_you can build by using Visual Studio Code or any programs can edit code_

### Indetify your own object using YOLO

##### - We will help you how to build your own yolo detect object

- 1.) You need to gather more than 2000 images of the object
- 2.) Download [label-studio](https://labelstud.io/) and upload all your image
- 3.) Mark your object using label-studio and export it With yolo and images 
###### The path should be
        images/   (contain all of your images)
        labels/   ( mark file. contain number)
        notes.json 
        classes.txt   ( your mark object names)
- 4.) make your own data.yml
###### For example
        ```sh
        train: images/train
        val: images/val

        nc: 3  # class name, you can check on your classes.txt and added to here
        names:
           - person
           - car
           - dog
        ```
- 5.) Open command promt on the folder contain all the step
```sh
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```
    _model = yolov8n best for detect human body or some small object, yolov12n for best performance - great for started, or other model version you can check at yolo github_

- 6.) The best model is in the weights folder. Called "best.pt"



### WARNING: You need a highly GPU to doing that action, and it can take really long to complete the train. I recommend using Google Collab for this action

- [Google_Collab](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=EMEDk5byzxY5)

_You can find step in step inside the Google Collab_


### USAGE
_Exit: using "e" button on keyboard to escape_
_Load picture and Images: using "p", choose the picture or videos you like from file_
_Camera: using "c" to open camera when you want to change from picture, videos -> camera_



### Support
_Fix your self_






