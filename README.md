---
title: Dog Cat Classifier
emoji: 🐶
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.41.0
app_file: app.py
pinned: false
license: mit
---

# 🐱🐶 Dog vs Cat Classifier

Ứng dụng phân loại Chó/Mèo sử dụng **ConvMixer** - một kiến trúc CNN hiện đại.

## ✨ Tính năng
- Upload ảnh và nhận kết quả phân loại ngay lập tức
- Hiển thị GradCAM để giải thích vùng model tập trung
- Giao diện Streamlit thân thiện

## 🏗️ Model
- **Architecture:** ConvMixer-768/32
- **Paper:** "Patches Are All You Need?"
- **Dataset:** Dogs vs Cats

## � Chạy Local

### Yêu cầu
- Python 3.8+
- pip

### Cài đặt

1. **Clone repository:**
```bash
git clone <repository-url>
cd dog-cat-classifier
```

2. **Tạo môi trường ảo (khuyến nghị):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ: `http://localhost:8501`

## �🚀 Sử dụng
Upload một ảnh chó hoặc mèo để xem kết quả phân loại!
---

## 🎬 Kịch bản Demo (2 phút)

### ⏱️ Phần 1: Giới thiệu (30 giây)
> *Mở ứng dụng tại http://localhost:8501*

**Lời thoại:**
> "Xin chào! Đây là ứng dụng **Dog vs Cat Classifier** - một hệ thống phân loại chó mèo sử dụng kiến trúc **ConvMixer** từ paper 'Patches Are All You Need?'. Ứng dụng được xây dựng bằng Streamlit và PyTorch."

**Thao tác:**
- Chỉ vào sidebar: "Ở đây hiển thị thông tin về model ConvMixer-768/32 với input size 224x224"
- "Model được huấn luyện trên dataset Dogs vs Cats và đạt độ chính xác gần 100%"

---

### ⏱️ Phần 2: Demo phân loại ảnh chó (40 giây)

**Lời thoại:**
> "Bây giờ mình sẽ demo với một ảnh chó"

**Thao tác:**
1. Click **"Browse files"** hoặc kéo thả ảnh chó vào
2. Chờ model load (lần đầu sẽ download từ Hugging Face Hub)
3. Chỉ vào kết quả:
   - "Model dự đoán đây là **Dog** với độ tin cậy **XX%**"
   - "Bên phải là **Grad-CAM heatmap** - vùng màu đỏ/vàng cho thấy model đang tập trung vào đâu để đưa ra quyết định"
   - "Như các bạn thấy, model tập trung vào vùng mặt/tai của chó - đây là những đặc trưng quan trọng để phân biệt"
   - "2 thanh progress bar bên dưới thể hiện xác suất của từng class"

---

### ⏱️ Phần 3: Demo phân loại ảnh mèo (40 giây)

**Lời thoại:**
> "Tiếp theo với ảnh mèo"

**Thao tác:**
1. Upload ảnh mèo
2. Chỉ vào kết quả:
   - "Model dự đoán chính xác đây là **Cat** với độ tin cậy **XX%**"
   - "Grad-CAM cho thấy model nhìn vào vùng đặc trưng của mèo như mắt, tai, râu"
   - "Điều này chứng tỏ model đã học được các đặc điểm quan trọng để phân biệt chó và mèo"

**Lời thoại bổ sung (nếu còn thời gian):**
> "Các bạn có thể thử với nhiều ảnh khác nhau - model hoạt động tốt nhất với ảnh rõ ràng, có chủ thể là chó hoặc mèo"

---

### ⏱️ Phần 4: Kết thúc (10 giây)

**Lời thoại:**
> "Tóm lại, ứng dụng demo thành công model ConvMixer cho bài toán phân loại chó mèo với độ chính xác cao. Model được deploy trên Hugging Face Hub và có thể dễ dàng tích hợp. Cảm ơn các bạn đã theo dõi!"

---

### 📝 Checklist trước khi demo
- [ ] Chuẩn bị 2-3 ảnh chó rõ ràng
- [ ] Chuẩn bị 2-3 ảnh mèo rõ ràng  
- [ ] Đảm bảo kết nối internet (để download model lần đầu)
- [ ] Chạy thử app trước để model đã được cache
- [ ] Mở sẵn http://localhost:8501 trên browser