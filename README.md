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

## Loading model from HF: https://huggingface.co/vtdung23/dog-cat-model/tree/main
