"""
Model Utilities for Dog vs Cat Classification
==============================================
Chứa các hàm và class cần thiết để load và sử dụng model ConvMixer.

Tái hiện từ paper: "Patches Are All You Need?" (ConvMixer)
Dataset: Dog vs Cat
"""

import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
import os

# ============================================================================
# CẤU HÌNH - REPO_ID TRỎ ĐẾN MODEL REPOSITORY
# ============================================================================
REPO_ID = "vtdung23/dog-cat-model"  # Model được lưu ở đây
MODEL_FILENAME = "model.pt"

# Labels cho classification
CLASS_NAMES = ["Cat", "Dog"]  # Index 0 = Cat, Index 1 = Dog

# ============================================================================
# TRANSFORM - Giống với test_transform trong notebook
# ============================================================================
def get_transform():
    """
    Trả về transform dùng cho inference.
    Được trích xuất từ notebook training.
    """
    transform = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


# ============================================================================
# LOAD MODEL TỪ HUGGING FACE HUB
# ============================================================================
def create_model():
    """
    Tạo kiến trúc model ConvMixer giống với notebook training.
    
    Model: convmixer_768_32 (768 channels, 32 layers)
    - Dựa trên paper "Patches Are All You Need?"
    - Fine-tuned cho binary classification (Dog vs Cat)
    """
    model = timm.create_model(
        'convmixer_768_32.in1k',  # ConvMixer với 768 channels, 32 layers
        pretrained=False,         # Không load pretrained weights từ ImageNet
        num_classes=1             # Binary classification (output 1 node)
    )
    return model


def load_model_from_hub(repo_id=REPO_ID, filename=MODEL_FILENAME, device="cpu"):
    """
    Load model từ Hugging Face Hub hoặc từ file local (nếu đang test).
    
    Args:
        repo_id: ID của repository trên HF Hub (format: "username/model-name")
        filename: Tên file trọng số model
        device: Device để load model ("cpu" hoặc "cuda")
    
    Returns:
        model: Model đã load trọng số và sẵn sàng inference
    """
    model_path = None
    
    # Kiểm tra nếu đang dùng placeholder -> load từ file local
    if repo_id == "YOUR_USERNAME/YOUR_MODEL_NAME":
        # Thử load từ file local (cho mục đích test)
        local_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(local_path):
            print(f"📂 Đang load model từ file local: {local_path}")
            model_path = local_path
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file local '{filename}'. "
                "Vui lòng cập nhật REPO_ID hoặc đặt file model.pt vào thư mục app."
            )
    else:
        # Download từ HF Hub
        print(f"📥 Đang tải model từ Hugging Face Hub: {repo_id}...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=None  # Sử dụng cache mặc định của HF
        )
        print(f"✅ Đã tải model về: {model_path}")
    
    # Tạo model architecture
    model = create_model()
    
    # Load trọng số
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # Chuyển sang device và đặt chế độ evaluation
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model đã sẵn sàng trên device: {device}")
    return model


# ============================================================================
# HÀM DỰ ĐOÁN
# ============================================================================
def preprocess_image(image: Image.Image, transform=None):
    """
    Tiền xử lý ảnh để đưa vào model.
    
    Args:
        image: PIL Image
        transform: Transform function (nếu None sẽ dùng mặc định)
    
    Returns:
        tensor: Tensor đã transform với shape (1, 3, 224, 224)
    """
    if transform is None:
        transform = get_transform()
    
    # Đảm bảo ảnh là RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply transform và thêm batch dimension
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # (3, 224, 224) -> (1, 3, 224, 224)
    
    return tensor


def predict(model, image: Image.Image, device="cpu"):
    """
    Dự đoán ảnh là Chó hay Mèo.
    
    Args:
        model: Model đã load
        image: PIL Image
        device: Device của model
    
    Returns:
        dict: {
            "class": "Dog" hoặc "Cat",
            "confidence": float (0-100),
            "probabilities": {"Cat": float, "Dog": float}
        }
    """
    # Tiền xử lý ảnh
    transform = get_transform()
    input_tensor = preprocess_image(image, transform)
    input_tensor = input_tensor.to(device)
    
    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        # Output là logit, cần sigmoid để chuyển thành probability
        prob = torch.sigmoid(output).item()
    
    # prob là xác suất của class 1 (Dog)
    # 1 - prob là xác suất của class 0 (Cat)
    prob_dog = prob * 100
    prob_cat = (1 - prob) * 100
    
    # Xác định class
    if prob >= 0.5:
        predicted_class = "Dog"
        confidence = prob_dog
    else:
        predicted_class = "Cat"
        confidence = prob_cat
    
    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "Cat": prob_cat,
            "Dog": prob_dog
        }
    }


# ============================================================================
# GRAD-CAM UTILITIES
# ============================================================================
def get_target_layer(model):
    """
    Lấy layer cuối cùng của ConvMixer để tính Grad-CAM.
    
    Trong ConvMixer, cấu trúc gồm:
    - stem: Patch embedding (Conv2d)
    - blocks: Danh sách các ConvMixer blocks
    - pooling: Global Average Pooling
    - head: Classifier
    
    Ta chọn block cuối cùng của `blocks` để visualize.
    """
    # ConvMixer trong timm có cấu trúc: stem -> blocks -> pooling -> head
    # Ta lấy block cuối cùng trong blocks
    return model.blocks[-1]


# ============================================================================
# TEST FUNCTIONS (chỉ chạy khi test local)
# ============================================================================
if __name__ == "__main__":
    # Test tạo model
    print("🧪 Testing model creation...")
    model = create_model()
    print(f"✅ Model created successfully!")
    print(f"   - Model type: {type(model).__name__}")
    
    # Test transform
    print("\n🧪 Testing transform...")
    transform = get_transform()
    print(f"✅ Transform created successfully!")
    
    # Test với ảnh random
    print("\n🧪 Testing with random tensor...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Forward pass successful!")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output value: {output.item():.4f}")
    
    # Test target layer
    print("\n🧪 Testing target layer for Grad-CAM...")
    target_layer = get_target_layer(model)
    print(f"✅ Target layer: {type(target_layer).__name__}")
