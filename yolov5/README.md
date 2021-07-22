# Giới thiệu
Bài toán phát hiện phương tiện giao thông trên cao tốc sử dụng Yolov5

## Training
Dữ liệu và file trọng số sau khi đào tạo được lưu ở [Google Driver](https://drive.google.com/drive/folders/1fLe6qXnOxgnH2CQOmKFG1ikh16VRtjyG?usp=sharing)

## Demo
```
python detect.py --weights weights/best.pt --img 640 --conf 0.25 --source data/images/train/000d68e42b71d3eac10ccc077aba07c1.jpg
```
Kết quả được lưu trong runs/detect/exp
