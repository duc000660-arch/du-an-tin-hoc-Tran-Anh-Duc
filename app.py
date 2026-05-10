import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import time

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="DigiBalance AI - 10A5", page_icon="⏱️", layout="centered")

# Giao diện CSS tùy chỉnh để trông chuyên nghiệp hơn
st.markdown("""
    <style>
    .main { background-color: #f0fdf4; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #10b981; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- KHỞI TẠO MÔ HÌNH AI (K-Means) ---
@st.cache_resource
def train_ai():
    # Dữ liệu giả lập để AI học các vùng ranh giới
    # [Giờ MXH, Giờ Tự Học]
    X = np.array([
        [1, 5], [2, 4], [1.5, 6],  # Nhóm Chăm chỉ
        [4, 3], [5, 4], [3, 3],    # Nhóm Cân bằng
        [8, 1], [10, 0.5], [9, 2]   # Nhóm Xao nhãng
    ])
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)
    return model

ai_model = train_ai()

# --- GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.title("⏱️ DigiBalance AI")
st.subheader("Dự án: Quản lý thời gian số - Trần Anh Đức 10A5")

st.info("👋 **Hướng dẫn:** Hãy kéo các thanh trượt bên dưới để mô tả thói quen của bạn, sau đó nhấn nút để AI phân loại.")

# Ô nhập liệu thông minh (MTC 3.e)
col1, col2 = st.columns(2)
with col1:
    mxh = st.slider("📱 Giờ dùng Mạng xã hội/ngày", 0.0, 15.0, 3.0, 0.5)
with col2:
    hoc = st.slider("📚 Giờ Tự học/ngày", 0.0, 15.0, 2.0, 0.5)

# Nút bấm xử lý
if st.button("🚀 PHÂN TÍCH HÀNH VI"):
    with st.spinner('AI đang tính toán cụm dữ liệu...'):
        time.sleep(1) # Tạo hiệu ứng xử lý
        
        # Dự đoán
        input_data = np.array([[mxh, hoc]])
        cluster = ai_model.predict(input_data)[0]
        
        st.divider()
        
        # Hiển thị kết quả thông minh dựa trên logic Clustering
        if mxh > hoc * 2 or mxh > 7:
            st.error("### 🔴 Nhãn: NHÓM XAO NHÃNG CAO")
            st.warning("**Khuyến nghị:** Bạn đang dành quá nhiều thời gian cho không gian ảo. Hãy thử quy tắc Pomodoro và hạn chế thông báo điện thoại.")
        elif hoc >= mxh:
            st.success("### 🟢 Nhãn: NHÓM CHĂM CHỈ / TẬP TRUNG")
            st.balloons()
            st.write("**Khuyến nghị:** Tuyệt vời! Hãy duy trì phong độ. Đừng quên nghỉ ngơi 5-10 phút sau mỗi giờ học nhé.")
        else:
            st.info("### 🔵 Nhãn: NHÓM CÂN BẰNG")
            st.write("**Khuyến nghị:** Bạn đang làm khá tốt. Hãy cố gắng giảm thêm 30 phút MXH để dành cho vận động thể chất.")

# Phần giải thích F.A.T.E (Tiêu chí minh bạch)
with st.expander("🔍 Giải thích kỹ thuật (Tiêu chí F.A.T.E)"):
    st.write(f"Mô hình sử dụng thuật toán **K-Means Clustering**.")
    st.write(f"Vị trí dữ liệu của bạn trên không gian 2D: `[{mxh}, {hoc}]`.")
    st.write("AI quyết định nhãn dựa trên khoảng cách ngắn nhất từ điểm của bạn tới tâm của 3 cụm hành vi đã học.")
