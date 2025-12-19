"""
Dog vs Cat Classification - Streamlit App
==========================================
Ứng dụng demo model ConvMixer cho bài toán phân loại Chó/Mèo.

Tái hiện từ paper: "Patches Are All You Need?" (ConvMixer)
"""

import streamlit as st
import torch
import numpy as np
import json
import io
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

# Import từ model_utils
from model_utils import (
    load_model_from_hub,
    predict,
    preprocess_image,
    get_transform,
    get_target_layer,
    REPO_ID,
    CLASS_NAMES
)

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="🐱🐶 Dog vs Cat Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .dog-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .cat-prediction {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #4ECDC4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE MODEL - Load model một lần và cache
# ============================================================================
@st.cache_resource
def load_cached_model():
    """Load model từ HF Hub và cache để không phải load lại."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_hub(device=device)
    return model, device


# ============================================================================
# GRAD-CAM FUNCTION
# ============================================================================
def generate_gradcam(model, image: Image.Image, device: str):
    """
    Tạo Grad-CAM heatmap để giải thích model đang nhìn vào đâu.
    
    Args:
        model: Model đã load
        image: PIL Image gốc
        device: Device của model
    
    Returns:
        visualization: Ảnh với heatmap overlay
    """
    # Lấy target layer
    target_layer = get_target_layer(model)
    
    # Khởi tạo GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Tiền xử lý ảnh cho model
    transform = get_transform()
    input_tensor = preprocess_image(image, transform)
    input_tensor = input_tensor.to(device)
    
    # Dự đoán trước để biết class nào được predict
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    # Binary classification: output > 0.5 -> Dog (category=1), else Cat (category=0)
    # BinaryClassifierOutputTarget sẽ đảo dấu gradient nếu category=0 (Cat)
    # Điều này đảm bảo Grad-CAM highlight đúng vùng cho cả Dog và Cat
    predicted_category = 1 if prob >= 0.5 else 0
    targets = [BinaryClassifierOutputTarget(predicted_category)]
    
    # Tạo grayscale cam với target phù hợp
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Lấy batch đầu tiên
    
    # Chuẩn bị ảnh gốc (resize về 224x224 và normalize về [0,1])
    image_resized = image.resize((224, 224))
    rgb_img = np.array(image_resized) / 255.0
    
    # Overlay heatmap lên ảnh
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization


# ============================================================================
# TAB 1: DEMO
# ============================================================================
def render_demo_tab():
    """Render nội dung tab Demo."""
    
    st.markdown("### 📸 Upload ảnh để phân loại")
    st.markdown("Hỗ trợ định dạng: JPG, JPEG, PNG")
    
    # Upload ảnh
    uploaded_file = st.file_uploader(
        "Chọn ảnh...",
        type=["jpg", "jpeg", "png"],
        help="Upload ảnh chó hoặc mèo để model dự đoán"
    )
    
    if uploaded_file is not None:
        # Đọc ảnh
        image = Image.open(uploaded_file)
        
        # Load model (cached)
        with st.spinner("🔄 Đang tải model..."):
            try:
                model, device = load_cached_model()
            except Exception as e:
                st.error(f"""
                ❌ **Lỗi khi tải model!**
                
                Vui lòng kiểm tra:
                1. Đã cập nhật `REPO_ID` trong file `model_utils.py` chưa?
                2. Repository trên Hugging Face Hub đã public chưa?
                3. File `model.pt` đã được upload lên repository chưa?
                
                Chi tiết lỗi: {str(e)}
                """)
                return
        
        # Layout 2 cột
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Ảnh gốc")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### 🔥 Grad-CAM Heatmap")
            with st.spinner("Đang tạo Grad-CAM..."):
                try:
                    gradcam_img = generate_gradcam(model, image, device)
                    st.image(gradcam_img, width=224) 
                    st.caption("Vùng màu đỏ/vàng = nơi model tập trung để đưa ra quyết định")
                except Exception as e:
                    st.warning(f"Không thể tạo Grad-CAM: {str(e)}")
        
        # Dự đoán
        st.markdown("---")
        st.markdown("### 🎯 Kết quả dự đoán")
        
        with st.spinner("Đang phân tích..."):
            result = predict(model, image, device)
        
        # Hiển thị kết quả
        predicted_class = result["class"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        
        # Chọn style dựa trên kết quả
        if predicted_class == "Dog":
            emoji = "🐶"
            style_class = "dog-prediction"
        else:
            emoji = "🐱"
            style_class = "cat-prediction"
        
        # Hiển thị prediction box
        st.markdown(f"""
        <div class="prediction-box {style_class}">
            <h1>{emoji} {predicted_class}</h1>
            <h3>Độ tin cậy: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bars cho xác suất
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🐱 **Cat**")
            st.progress(probs["Cat"] / 100)
            st.markdown(f"**{probs['Cat']:.2f}%**")
        
        with col2:
            st.markdown("🐶 **Dog**")
            st.progress(probs["Dog"] / 100)
            st.markdown(f"**{probs['Dog']:.2f}%**")
    
    else:
        # Placeholder khi chưa upload
        st.info("👆 Vui lòng upload ảnh chó hoặc mèo để bắt đầu!")
        
        # Sample images info
        with st.expander("💡 Gợi ý"):
            st.markdown("""
            - Ảnh nên rõ ràng, có chủ thể là chó hoặc mèo
            - Model hoạt động tốt nhất với ảnh có nền đơn giản
            - Hỗ trợ các định dạng phổ biến: JPG, PNG
            """)


# ============================================================================
# TAB 2: REPORT
# ============================================================================
def render_report_tab():
    """Render nội dung tab Report."""
    
    st.markdown("### 📊 Báo cáo kết quả huấn luyện")
    st.markdown("So sánh hiệu suất giữa **ResNet34** và **ConvMixer** trên dataset Dog vs Cat")
    
    # Load results
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file `results.json`!")
        return
    except json.JSONDecodeError:
        st.error("❌ File `results.json` không đúng định dạng!")
        return
    
    # Lấy data
    resnet_data = results.get("result_resnet", {})
    convmixer_data = results.get("result_convmixer", {})
    
    # ===== METRICS OVERVIEW =====
    st.markdown("#### 🏆 Tổng quan hiệu suất")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Best validation accuracy
    resnet_best_acc = max(resnet_data.get("valid_metric", [0])) * 100
    convmixer_best_acc = max(convmixer_data.get("valid_metric", [0])) * 100
    
    # Final validation loss
    resnet_final_loss = resnet_data.get("valid_loss", [0])[-1]
    convmixer_final_loss = convmixer_data.get("valid_loss", [0])[-1]
    
    with col1:
        st.metric(
            label="🎯 ResNet34 - Best Acc",
            value=f"{resnet_best_acc:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🎯 ConvMixer - Best Acc",
            value=f"{convmixer_best_acc:.2f}%",
            delta=f"+{convmixer_best_acc - resnet_best_acc:.2f}%" if convmixer_best_acc > resnet_best_acc else f"{convmixer_best_acc - resnet_best_acc:.2f}%"
        )
    
    with col3:
        st.metric(
            label="📉 ResNet34 - Final Loss",
            value=f"{resnet_final_loss:.4f}"
        )
    
    with col4:
        st.metric(
            label="📉 ConvMixer - Final Loss",
            value=f"{convmixer_final_loss:.4f}",
            delta=f"{convmixer_final_loss - resnet_final_loss:.4f}" if convmixer_final_loss < resnet_final_loss else f"+{convmixer_final_loss - resnet_final_loss:.4f}",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # ===== BIỂU ĐỒ =====
    st.markdown("#### 📈 Biểu đồ quá trình huấn luyện")
    
    # Chuẩn bị data
    epochs = list(range(1, len(resnet_data.get("train_loss", [])) + 1))
    
    # Colors
    resnet_color = '#FF6B6B'
    convmixer_color = '#4ECDC4'
    train_style = '-'
    valid_style = '--'
    
    # Helper function to create download buttons for a chart
    def add_download_buttons(fig, chart_name, key_suffix):
        # Save to PDF
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', dpi=300)
        pdf_buffer.seek(0)
        
        # Save to SVG
        svg_buffer = io.BytesIO()
        fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
        svg_buffer.seek(0)
        
        # 2 nút nằm ngang cạnh nhau
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.download_button(
                label="📥 PDF",
                data=pdf_buffer,
                file_name=f"{chart_name}.pdf",
                mime="application/pdf",
                key=f"download_{key_suffix}_pdf"
            )
        with btn_col2:
            st.download_button(
                label="📥 SVG",
                data=svg_buffer,
                file_name=f"{chart_name}.svg",
                mime="image/svg+xml",
                key=f"download_{key_suffix}_svg"
            )
        
        plt.close(fig)
    
    # Layout 2 cột cho các biểu đồ
    col_chart1, col_chart2 = st.columns(2)
    
    # 1. Training Loss
    with col_chart1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(epochs, resnet_data.get("train_loss", []), train_style, 
                 color=resnet_color, label='ResNet34', linewidth=2)
        ax1.plot(epochs, convmixer_data.get("train_loss", []), train_style, 
                 color=convmixer_color, label='ConvMixer', linewidth=2)
        ax1.set_title("Training Loss", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        add_download_buttons(fig1, "training_loss", "train_loss")
    
    # 2. Validation Loss
    with col_chart2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(epochs, resnet_data.get("valid_loss", []), valid_style, 
                 color=resnet_color, label='ResNet34', linewidth=2, marker='o')
        ax2.plot(epochs, convmixer_data.get("valid_loss", []), valid_style, 
                 color=convmixer_color, label='ConvMixer', linewidth=2, marker='s')
        ax2.set_title("Validation Loss", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        add_download_buttons(fig2, "validation_loss", "valid_loss")
    
    # Layout 2 cột tiếp theo
    col_chart3, col_chart4 = st.columns(2)
    
    # 3. Training Accuracy
    with col_chart3:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        train_acc_resnet = [x * 100 for x in resnet_data.get("train_metric", [])]
        train_acc_convmixer = [x * 100 for x in convmixer_data.get("train_metric", [])]
        ax3.plot(epochs, train_acc_resnet, train_style, 
                 color=resnet_color, label='ResNet34', linewidth=2)
        ax3.plot(epochs, train_acc_convmixer, train_style, 
                 color=convmixer_color, label='ConvMixer', linewidth=2)
        ax3.set_title("Training Accuracy", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([70, 100])
        plt.tight_layout()
        st.pyplot(fig3)
        add_download_buttons(fig3, "training_accuracy", "train_acc")
    
    # 4. Validation Accuracy
    with col_chart4:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        valid_acc_resnet = [x * 100 for x in resnet_data.get("valid_metric", [])]
        valid_acc_convmixer = [x * 100 for x in convmixer_data.get("valid_metric", [])]
        ax4.plot(epochs, valid_acc_resnet, valid_style, 
                 color=resnet_color, label='ResNet34', linewidth=2, marker='o')
        ax4.plot(epochs, valid_acc_convmixer, valid_style, 
                 color=convmixer_color, label='ConvMixer', linewidth=2, marker='s')
        ax4.set_title("Validation Accuracy", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy (%)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([90, 100])
        plt.tight_layout()
        st.pyplot(fig4)
        add_download_buttons(fig4, "validation_accuracy", "valid_acc")
    
    st.markdown("---")
    
    # ===== BẢNG CHI TIẾT =====
    st.markdown("#### 📋 Bảng số liệu chi tiết")
    
    tab_resnet, tab_convmixer = st.tabs(["📊 ResNet34", "📊 ConvMixer"])
    
    with tab_resnet:
        import pandas as pd
        df_resnet = pd.DataFrame({
            "Epoch": epochs,
            "Train Loss": [f"{x:.4f}" for x in resnet_data.get("train_loss", [])],
            "Valid Loss": [f"{x:.4f}" for x in resnet_data.get("valid_loss", [])],
            "Train Acc (%)": [f"{x*100:.2f}" for x in resnet_data.get("train_metric", [])],
            "Valid Acc (%)": [f"{x*100:.2f}" for x in resnet_data.get("valid_metric", [])]
        })
        st.dataframe(df_resnet, use_container_width=True)
    
    with tab_convmixer:
        import pandas as pd
        df_convmixer = pd.DataFrame({
            "Epoch": epochs,
            "Train Loss": [f"{x:.4f}" for x in convmixer_data.get("train_loss", [])],
            "Valid Loss": [f"{x:.4f}" for x in convmixer_data.get("valid_loss", [])],
            "Train Acc (%)": [f"{x*100:.2f}" for x in convmixer_data.get("train_metric", [])],
            "Valid Acc (%)": [f"{x*100:.2f}" for x in convmixer_data.get("valid_metric", [])]
        })
        st.dataframe(df_convmixer, use_container_width=True)
    
    # ===== KẾT LUẬN =====
    st.markdown("---")
    st.markdown("#### 💡 Kết luận")
    
    winner = "ConvMixer" if convmixer_best_acc >= resnet_best_acc else "ResNet34"
    
    st.success(f"""
    🏆 **{winner}** đạt hiệu suất tốt nhất trên validation set!
    
    - **ResNet34**: {resnet_best_acc:.2f}% accuracy
    - **ConvMixer**: {convmixer_best_acc:.2f}% accuracy
    
    ConvMixer - một kiến trúc đơn giản chỉ dùng patch embeddings và depthwise convolutions - 
    đã chứng minh hiệu quả cạnh tranh với ResNet trên bài toán Dog vs Cat, 
    phù hợp với kết luận của paper "Patches Are All You Need?".
    """)


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar():
    """Render sidebar với thông tin bổ sung."""
    
    with st.sidebar:
        st.markdown("## 🐾 Dog vs Cat Classifier")
        st.markdown("---")
        
        st.markdown("### 📖 Về project")
        st.markdown("""
        Đây là ứng dụng demo model **ConvMixer** được huấn luyện 
        trên dataset Dog vs Cat.
        
        **Paper gốc:** *"Patches Are All You Need?"*
        """)
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Thông tin kỹ thuật")
        st.markdown(f"""
        - **Model:** ConvMixer-768/32
        - **Input size:** 224x224
        - **Classes:** Cat, Dog
        - **Framework:** PyTorch + timm
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔗 Links")
        st.markdown("""
        - [ConvMixer Paper](https://arxiv.org/abs/2201.09792)
        - [timm Library](https://github.com/huggingface/pytorch-image-models)
        """)
        
        st.markdown("---")
        
        # Device info
        device = "CUDA 🚀" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device}")


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Hàm chính chạy ứng dụng."""
    
    # Render sidebar
    render_sidebar()
    
    # Header
    st.markdown('<h1 class="main-header">🐱 Dog vs Cat Classifier 🐶</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by ConvMixer - "Patches Are All You Need?"</p>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["🎮 Demo", "📊 Report"])
    
    with tab1:
        render_demo_tab()
    
    with tab2:
        render_report_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Made with ❤️ using Streamlit | Machine Learning Project</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
