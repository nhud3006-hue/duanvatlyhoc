import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import lightkurve as lk

# Tiêu đề
st.set_page_config(page_title="🔭 Thợ săn hành tinh", layout="centered")
st.title("🪐 Dự đoán ngoại hành tinh từ dữ liệu TESS")
st.write("Upload file dữ liệu và để AI phán quyết!")

# Load model và scaler
model = joblib.load('planet_model_v2.pkl')
scaler = joblib.load('scaler_v2.pkl')

# Upload CSV
uploaded_file = st.file_uploader("📁 Chọn file CSV chứa các ứng viên", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Đã tải dữ liệu thành công!")
    st.dataframe(df.head())

    # Dự đoán
    feature_cols = ['period', 'radius', 'mass', 'inclination']
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    df_result = df.copy()
    df_result['Prediction'] = y_pred
    df_result['Confidence'] = y_proba

    # Lọc ứng viên tiềm năng
    candidates = df_result[df_result['Prediction'] == 1]
    st.success(f"✅ Tìm thấy {len(candidates)} ứng viên hành tinh tiềm năng!")

    st.subheader("📊 Kết quả dự đoán")
    st.dataframe(candidates)

    # Vẽ biểu đồ
    fig, ax = plt.subplots()
    ax.scatter(df_result['period'], df_result['radius'], c=df_result['Confidence'], cmap='RdYlGn')
    ax.set_xlabel('Chu kỳ (ngày)')
    ax.set_ylabel('Bán kính (Rjup)')
    st.pyplot(fig)
