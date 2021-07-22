# Giới thiệu
Đây là bài toán đo kích thước vật thể (chuột, điện thoại, sách, thước) dựa trên bài toán phân đoạn ảnh (image segmentation). Phân đoạn ảnh để tách ra các vùng của các vật thể khác
nhau, sau đó dùng các thuật toán xử lý ảnh đơn giản để vẽ hộp giới hạn của mỗi vật thể. Dựa vào kích thước cố định của vật làm tham chiếu (thước) để tính toán kích thước của
các vật thể còn lại

## Yêu cầu
- Python3
- PyTorch
- Cuda
- opencv-python
- pillow
- imutils

## Models
Tải các trọng số đã được train sẵn tại [Google Drive](https://drive.google.com/drive/folders/1Bi6B_DMa3CLWXRcjvRJHNkS3HsiRxyry?usp=sharing).

## Demo
```python main.py```
Chương trình thử nghiệm trên video và kết quả được lưu ở file project.mp4
