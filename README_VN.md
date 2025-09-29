# Theo dõi Khuôn Mặt ( Hiện tại đang trong giai đoạn phát triển)
# Đoạn script này được viết bằng Python

# ![title](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh1.png)
- [Cà_Phê----MIN](https://www.youtube.com/watch?v=7m8ek8D9me0&list=RD7m8ek8D9me0&start_radio=1)
# ![title](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh2.png)
- [ Cứ_Chill_Thôi----Chillies](https://www.youtube.com/watch?v=LZN4I3K8SC0&list=RDLZN4I3K8SC0&start_radio=1)
# ![tittle](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh3.png)
- [Bad----Michael_Jackson](https://www.youtube.com/watch?v=dsUXAEzaC3Q&list=RDdsUXAEzaC3Q&start_radio=1)
# ![title](https://github.com/172009/Face-Tracking/blob/main/assets/image/sh4.png)
- [Genshin_Impact_Nod_Krai_Performance----Genshin_Impact](https://www.youtube.com/watch?v=RuXa_yxZMGI&list=RDRuXa_yxZMGI&start_radio=1)




#### NOTE: Tôi đang cố gắng làm nó tốt hơn trong việc xác định danh tính vật thể
#### NOTE: Chương trình này có sử dụng webcam của bạn. Kiểm tra các nút bên dưới để sử dụng


##### Nhiệm Vụ:

###### - [x] Nhận diện khuôn mặt
###### -[ ] Nhận diện logo của Trường
###### -[ ] So sánh khuôn mặt và xác định danh tính


_(có thể sẽ thêm mục tiêu trong tương lai)_


# Bắt đầu sử dụng
### Yêu cầu tối thiểu
- [Python-3.13](https://www.python.org/downloads/)
- [CUDA_v11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN_v8.5.0_CUDA_v11.x](https://developer.nvidia.com/compute/cudnn/secure/8.5.0/local_installers/11.7/cudnn-windows-x86_64-8.5.0.96_cuda11-archive.zip)
  #### xin hãy giải nén file cuDNN vào nơi mà CUDA đã được cài đặt, file cuDNN bao gồm bin, include và một số tệp khác....
```sh
pip install -r requirements.txt
```

### a) sử dụng từ release
1. Xác định Release trong Github
2. Tải Release mới nhất để sử dụng

### b) sử dụng từ mã nguồn
1. Tìm nút "Code" ở trên cùng
2. Chọn và tải về "Download ZIP"
3. Mở script.py


### Xây dựng
```sh
git clone https://github.com/172009/Face-Tracking.git
cd Face-Tracking
pip install -r requirements.txt
```
_bạn có thể xây dựng đoạn mã cho riêng bạn bằng Visual Studio Code hoặc các ứng dụng chỉnh sửa lệnh_

### Nhận dạng vật thể bằng YOLO

##### - Tôi sẽ giúp bạn xây dựng một model YOLO để nhận dạng vật thể bạn mong muốn

- 1.) Bạn cần ít nhất 2000 ảnh về vật thể đó ( càng rõ nét và từ nhiều góc ánh sáng )
- 2.) Tải [label-studio](https://labelstud.io/) và tải lên toàn bộ ảnh
- 3.) Đánh dấu vật thể và export bằng "YOLO with Images" để có file lẫn ảnh
###### Bên trong file chính sát phát có
        images/   (Chứa ảnh)
        labels/   ( File chứa các số đánh giá ảnh, con số)
        notes.json 
        classes.txt   ( tên vật thể bạn đã đánh dấu trong label-studio)
- 4.) tạo cho riêng bạn tệp data.yml
###### Ví dụ
        ```sh
        train: images/train
        val: images/val

        nc: 3  # class name, you can check on your classes.txt and added to here
        names:
           - person
           - car
           - dog
        ```
- 5.) Mở Command Promt từ thư mục hoàn thành các bước trên
```sh
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```
    _model = yolov8 rất tốt cho việc đánh giá cơ thể người cũng như các vật nhỏ, trong khi đó yolov12 lại có hiệu năng tốt và nhanh - rất tốt để khởi đầu, bạn có thể kiểm tra các model khác qua github của họ_

- 6.) Model tốt nhất nằm trong thư mục weight. Được gọi là "best.pt"



### Cảnh báo: Nếu bạn đang có ý định xây dựng model cho riêng bạn, hãy biết rằng nó rất tốn thời gian cũng như điều kiện cấu hình đồ họa cao. Việc VRAM quá thấp có thể dẫn đến crash trong lúc Train.
### NOTE: Tôi khuyên bạn hãy sử dụng Google Collab, trong đây có sẳn step-in-step chi tiết để hướng dẫn. Và tất nhiên nó sẽ build trên máy ảo, không phải của bạn, tất nhiên là hoàn toàn miễn phí

- [Google_Collab](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=EMEDk5byzxY5)



### Cách dùng
- _thoát: nhấn nút "e", hãy chuyển bàn phím sang ENG_


- _Tải ảnh: nhấn nút "p" để xuất hiện hộp thoại tải ảnh (có thể load cả video )_


- _Camera: nhấn nút "c" để quay lại camera_


- _Thêm người: nhấn nút "a" để thêm tên, và chọn ảnh có chứa người đó ( ảnh chỉ chứa 1 người đó )_



### Hỗ trợ
_Chịu_

###Credits
- [InsightFace](https://github.com/deepinsight/insightface)
- [YOLO](https://github.com/ultralytics/ultralytics)






