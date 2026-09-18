# -*- coding: utf-8 -*-
"""
HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
Sử dụng 8 chỉ số Beneish M-Score & Mô hình Logistic Regression
Streamlit Web Application
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, classification_report
)

# -----------------------------------------------------------------------------
# CẤU HÌNH TRANG WEB STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dự báo Gian lận BCTC | Beneish M-Score",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS giao diện chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-fraud {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #4ADE80;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ĐỊNH NGHĨA BIẾN VÀ THUẬT TOÁN
# -----------------------------------------------------------------------------
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

RATIO_INFO = {
    "DSRI": {
        "name": "Days Sales in Receivables Index",
        "vi": "Chỉ số số ngày thu tiền khách hàng",
        "desc": "Đo lường tốc độ tăng các khoản phải thu so với doanh thu. Tỷ lệ > 1 bất thường cảnh báo ghi nhận doanh thu non hoặc ảo.",
        "default": 1.05, "min": 0.1, "max": 5.0, "step": 0.05
    },
    "GMI": {
        "name": "Gross Margin Index",
        "vi": "Chỉ số tỷ suất lãi gộp",
        "desc": "Tỷ số biên lãi gộp năm trước so với năm nay. Nếu > 1, biên lợi nhuận đang suy giảm, tạo động cơ thao túng lợi nhuận.",
        "default": 1.02, "min": 0.1, "max": 5.0, "step": 0.05
    },
    "AQI": {
        "name": "Asset Quality Index",
        "vi": "Chỉ số chất lượng tài sản",
        "desc": "Đo lường tỷ lệ tài sản phi hiện vật và chi phí hoãn lại. Tỷ lệ > 1 cảnh báo việc vốn hóa chi phí bất hợp lý.",
        "default": 1.01, "min": 0.1, "max": 5.0, "step": 0.05
    },
    "SGI": {
        "name": "Sales Growth Index",
        "vi": "Chỉ số tăng trưởng doanh thu",
        "desc": "Tỷ lệ tăng trưởng doanh thu năm nay so với năm trước. Tăng trưởng quá nóng thường chịu áp lực duy trì bằng thủ thuật.",
        "default": 1.15, "min": 0.1, "max": 5.0, "step": 0.05
    },
    "DEPI": {
        "name": "Depreciation Index",
        "vi": "Chỉ số mức khấu hao",
        "desc": "Tỷ số tỷ lệ khấu hao năm trước so với năm nay. Nếu > 1, DN có thể đang kéo dài thời gian khấu hao để tăng lợi nhuận.",
        "default": 1.00, "min": 0.1, "max": 5.0, "step": 0.05
    },
    "SGAI": {
        "name": "Sales, General & Admin Index",
        "vi": "Chỉ số chi phí bán hàng & quản lý",
        "desc": "Đo lường tỷ lệ chi phí SG&A trên doanh thu. Nếu > 1, hiệu quả quản lý chi phí suy giảm.",
        "default": 1.03, "min": 0.1, "max": 5.0, "step": 0.05
    },
    "TATA": {
        "name": "Total Accruals to Total Assets",
        "vi": "Biến dồn tích trên tổng tài sản",
        "desc": "(Lợi nhuận thuần - Dòng tiền thuần từ HĐKD) / Tổng tài sản. Tỷ lệ dương cao cho thấy lợi nhuận kế toán thiếu dòng tiền hỗ trợ.",
        "default": 0.05, "min": -1.0, "max": 1.0, "step": 0.01
    },
    "LVGI": {
        "name": "Leverage Index",
        "vi": "Chỉ số đòn bẩy tài chính",
        "desc": "Tỷ lệ nợ trên tổng tài sản so với năm trước. Nếu > 1, đòn bẩy tăng làm gia tăng rủi ro tài chính và vi phạm cam kết nợ.",
        "default": 1.08, "min": 0.1, "max": 5.0, "step": 0.05
    }
}

# -----------------------------------------------------------------------------
# HÀM HUẤN LUYỆN VÀ XỬ LÝ DỮ LIỆU CÓ CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Tải dữ liệu mặc định MScore_data.csv nếu có trong thư mục"""
    try:
        df = pd.read_csv("MScore_data.csv")
        return df
    except Exception:
        return None

def clean_and_prepare_data(df):
    """Kiểm tra và chuẩn hóa dữ liệu đầu vào"""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {missing}")
    
    clean_df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.dropna().reset_index(drop=True)
    clean_df[TARGET] = clean_df[TARGET].astype(int)
    
    if not set(clean_df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Cột FRAUD_FLAG chỉ được chứa giá trị 0 hoặc 1.")
    
    return clean_df

@st.cache_resource
def train_logistic_model(data_bytes, test_size=0.20, random_state=42):
    """Huấn luyện mô hình Logistic Regression chuẩn hóa"""
    df = pd.read_csv(io.BytesIO(data_bytes))
    clean_df = clean_and_prepare_data(df)
    
    X = clean_df[FEATURES]
    y = clean_df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Pipeline gồm StandardScaler và LogisticRegression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    pipeline.fit(X_train, y_train)
    
    # Trích xuất hệ số
    logistic = pipeline.named_steps["logistic"]
    scaler = pipeline.named_steps["scaler"]
    intercept = logistic.intercept_[0]
    coef = logistic.coef_[0]
    
    coef_df = pd.DataFrame({
        "Chỉ số": FEATURES,
        "Hệ số (Beta)": coef,
        "Odds Ratio (e^Beta)": np.exp(coef),
        "Tác động rủi ro": ["Tăng nguy cơ gian lận" if c > 0 else "Giảm nguy cơ gian lận" for c in coef]
    })
    
    return pipeline, scaler, logistic, X_train, X_test, y_train, y_test, coef_df, intercept

def compute_beneish_mscore(row_dict):
    """Tính điểm số Beneish M-Score theo công thức gốc 1999"""
    m = (-4.84 +
         0.920 * float(row_dict.get("DSRI", 1.0)) +
         0.528 * float(row_dict.get("GMI", 1.0)) +
         0.404 * float(row_dict.get("AQI", 1.0)) +
         0.892 * float(row_dict.get("SGI", 1.0)) +
         0.115 * float(row_dict.get("DEPI", 1.0)) -
         0.172 * float(row_dict.get("SGAI", 1.0)) +
         4.037 * float(row_dict.get("TATA", 0.0)) +
         0.0327 * float(row_dict.get("LVGI", 1.0)))
    return m

# -----------------------------------------------------------------------------
# THANH ĐIỀU HƯỚNG (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=70)
    st.title("Hệ thống Dự báo BCTC")
    st.caption("Ứng dụng Machine Learning & Beneish M-Score")
    st.markdown("---")
    
    st.subheader("⚙️ Cấu hình Dữ liệu & Mô hình")
    data_source_option = st.radio(
        "Nguồn dữ liệu huấn luyện:",
        ["Dữ liệu mặc định (MScore_data.csv)", "Tải lên file CSV mới"],
        index=0
    )
    
    uploaded_file = None
    data_bytes = None
    if data_source_option == "Tải lên file CSV mới":
        uploaded_file = st.file_uploader("Chọn file CSV dữ liệu:", type=["csv"])
        if uploaded_file is not None:
            data_bytes = uploaded_file.getvalue()
    else:
        default_df = load_default_data()
        if default_df is not None:
            csv_buffer = io.StringIO()
            default_df.to_csv(csv_buffer, index=False)
            data_bytes = csv_buffer.getvalue().encode('utf-8')
        else:
            st.warning("Chưa tìm thấy file `MScore_data.csv` tại thư mục hiện hành. Vui lòng tải file lên.")
            uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
            if uploaded_file is not None:
                data_bytes = uploaded_file.getvalue()

    st.markdown("---")
    st.subheader("🎯 Tùy chỉnh Ngưỡng phân loại")
    threshold = st.slider(
        "Ngưỡng xác suất gian lận (Cut-off Threshold):",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Nếu P(Fraud) >= Ngưỡng, công ty sẽ bị gán nhãn Gian lận (1)."
    )
    
    st.markdown("---")
    st.markdown("""
    **Nhóm chỉ số Beneish (8 biến):**
    - `DSRI`: Khoản phải thu / Doanh thu
    - `GMI`: Biên lãi gộp
    - `AQI`: Chất lượng tài sản
    - `SGI`: Tăng trưởng doanh thu
    - `DEPI`: Mức khấu hao
    - `SGAI`: Chi phí bán hàng & QLDN
    - `TATA`: Biến dồn tích kế toán
    - `LVGI`: Đòn bẩy tài chính
    """)

# -----------------------------------------------------------------------------
# NỘI DUNG CHÍNH (MAIN TABS)
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Mô hình hồi quy Logistic kết hợp 8 chỉ số tài chính Beneish M-Score giúp kiểm toán viên và nhà đầu tư phát hiện sớm rủi ro gian lận.</div>', unsafe_allow_html=True)

if data_bytes is None:
    st.info("👋 Vui lòng tải lên file dữ liệu CSV để bắt đầu chạy mô hình.")
    st.stop()

# Huấn luyện mô hình
try:
    pipeline, scaler, logistic, X_train, X_test, y_train, y_test, coef_df, intercept = train_logistic_model(data_bytes)
except Exception as e:
    st.error(f"Lỗi khi nạp dữ liệu: {e}")
    st.stop()

# Dự báo tập Test với ngưỡng linh hoạt
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# Tính toán các chỉ số đánh giá
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc = roc_auc_score(y_test, y_prob)

# Tạo các Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Huấn luyện & Đánh giá Mô hình",
    "🔍 Dự báo Doanh nghiệp Đơn lẻ",
    "📁 Dự báo Hàng loạt (Batch Prediction)",
    "📖 Cẩm nang 8 Chỉ số Beneish"
])

# -----------------------------------------------------------------------------
# TAB 1: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. Tổng quan Bộ dữ liệu & Hiệu năng Mô hình")
    
    # Thống kê nhanh
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Tổng quan sát", f"{len(X_train) + len(X_test):,}")
    with col_m2:
        st.metric("Tập Train (80%)", f"{len(X_train):,}")
    with col_m3:
        st.metric("Tập Test (20%)", f"{len(X_test):,}")
    with col_m4:
        st.metric("Tỷ lệ Gian lận (Train)", f"{(y_train.sum() / len(y_train)):.1%}")
    with col_m5:
        st.metric("Ngưỡng phân loại", f"{threshold:.2f}")

    st.markdown("---")
    st.subheader("2. Các Chỉ tiêu Đánh giá trên Tập Test")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
    col_kpi1.metric("🎯 Accuracy (Độ chính xác)", f"{acc:.2%}")
    col_kpi2.metric("🔎 Precision (Độ chuẩn xác)", f"{prec:.2%}")
    col_kpi3.metric("🚨 Recall (Độ nhạy)", f"{rec:.2%}")
    col_kpi4.metric("⚖️ F1-Score", f"{f1:.2%}")
    col_kpi5.metric("📈 AUC-ROC", f"{auc:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 🔲 Ma trận Nhầm lẫn (Confusion Matrix)")
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
            xticklabels=["Không gian lận (0)", "Gian lận (1)"],
            yticklabels=["Không gian lận (0)", "Gian lận (1)"]
        )
        ax_cm.set_xlabel("Dự báo của mô hình", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("Thực tế", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.caption(f"**Chi tiết:** TN = {tn} (Đúng không GL) | FP = {fp} (Báo động giả) | FN = {fn} (Bỏ sót GL) | TP = {tp} (Phát hiện đúng GL)")

    with col_chart2:
        st.markdown("##### 📈 Đường cong ROC (Receiver Operating Characteristic)")
        fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 3.8))
        ax_roc.plot(fpr_vals, tpr_vals, color="#1E3A8A", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("Tỷ lệ Báo động giả (FPR)", fontsize=10)
        ax_roc.set_ylabel("Tỷ lệ Phát hiện đúng (TPR / Recall)", fontsize=10)
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_roc)

    st.markdown("---")
    st.subheader("3. Hệ số Hồi quy & Tác động của các Biến (Odds Ratio)")
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        styled_coef = coef_df.style.format({
            "Hệ số (Beta)": "{:.4f}",
            "Odds Ratio (e^Beta)": "{:.4f}"
        }).background_gradient(subset=["Hệ số (Beta)"], cmap="coolwarm")
        st.dataframe(styled_coef, use_container_width=True)
    
    with col_t2:
        st.markdown("**Phương trình Logit (Biến đã chuẩn hóa):**")
        eq_text = f"\\text{{logit}}(P) = {intercept:.4f}"
        for name, b in zip(FEATURES, logistic.coef_[0]):
            sign = "+" if b >= 0 else "-"
            eq_text += f" {sign} {abs(b):.4f} \\times \\text{{{name}}}^* "
        st.latex(eq_text)
        
        st.info("""
        💡 **Diễn giải Odds Ratio (OR):**
        - **OR > 1**: Khi chỉ số tăng lên, khả năng xảy ra gian lận **tăng** theo cấp số nhân.
        - **OR < 1**: Khi chỉ số tăng lên, xác suất gian lận có xu hướng **giảm**.
        """)

    # Nút tải file Excel kết quả
    st.markdown("---")
    metrics_summary_df = pd.DataFrame({
        "Chỉ tiêu": ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "AUC-ROC", "Threshold", "Train Obs", "Test Obs"],
        "Giá trị": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{spec:.4f}", f"{auc:.4f}", f"{threshold:.2f}", len(X_train), len(X_test)]
    })
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
        pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_excel(writer, sheet_name="Confusion_Matrix")
        metrics_summary_df.to_excel(writer, sheet_name="Metrics", index=False)
    
    st.download_button(
        label="📥 Tải Báo cáo Đánh giá Mô hình (Excel)",
        data=excel_buffer.getvalue(),
        file_name="Bao_Cao_Danh_Gia_Mo_Hinh_MScore.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -----------------------------------------------------------------------------
# TAB 2: DỰ BÁO DOANH NGHIỆP ĐƠN LẺ
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🔍 Dự báo Rủi ro Gian lận cho 1 Doanh nghiệp")
    st.write("Nhập 8 chỉ số Beneish tính từ BCTC hoặc chọn nhanh hồ sơ mẫu để kiểm tra:")
    
    # Nút chọn mẫu nhanh
    sample_col1, sample_col2, sample_col3 = st.columns([1, 1, 2])
    profile = sample_col1.selectbox(
        "Chọn mẫu tham khảo:",
        ["Tự nhập chỉ số", "Doanh nghiệp bình thường (An toàn)", "Doanh nghiệp rủi ro gian lận cao"]
    )
    
    sample_vals = {}
    if profile == "Doanh nghiệp bình thường (An toàn)":
        sample_vals = {"DSRI": 0.95, "GMI": 0.98, "AQI": 0.92, "SGI": 1.08, "DEPI": 0.99, "SGAI": 0.97, "TATA": -0.02, "LVGI": 0.95}
    elif profile == "Doanh nghiệp rủi ro gian lận cao":
        sample_vals = {"DSRI": 1.65, "GMI": 1.45, "AQI": 1.35, "SGI": 1.60, "DEPI": 1.25, "SGAI": 1.30, "TATA": 0.18, "LVGI": 1.40}

    with st.form("single_prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        inputs = {}
        
        with col1:
            val_dsri = sample_vals.get("DSRI", RATIO_INFO["DSRI"]["default"])
            inputs["DSRI"] = st.number_input("1. DSRI (Khoản phải thu)", min_value=0.0, max_value=10.0, value=float(val_dsri), step=0.05, help=RATIO_INFO["DSRI"]["desc"])
            val_depi = sample_vals.get("DEPI", RATIO_INFO["DEPI"]["default"])
            inputs["DEPI"] = st.number_input("5. DEPI (Mức khấu hao)", min_value=0.0, max_value=10.0, value=float(val_depi), step=0.05, help=RATIO_INFO["DEPI"]["desc"])

        with col2:
            val_gmi = sample_vals.get("GMI", RATIO_INFO["GMI"]["default"])
            inputs["GMI"] = st.number_input("2. GMI (Biên lãi gộp)", min_value=0.0, max_value=10.0, value=float(val_gmi), step=0.05, help=RATIO_INFO["GMI"]["desc"])
            val_sgai = sample_vals.get("SGAI", RATIO_INFO["SGAI"]["default"])
            inputs["SGAI"] = st.number_input("6. SGAI (Chi phí SG&A)", min_value=0.0, max_value=10.0, value=float(val_sgai), step=0.05, help=RATIO_INFO["SGAI"]["desc"])

        with col3:
            val_aqi = sample_vals.get("AQI", RATIO_INFO["AQI"]["default"])
            inputs["AQI"] = st.number_input("3. AQI (Chất lượng tài sản)", min_value=0.0, max_value=10.0, value=float(val_aqi), step=0.05, help=RATIO_INFO["AQI"]["desc"])
            val_tata = sample_vals.get("TATA", RATIO_INFO["TATA"]["default"])
            inputs["TATA"] = st.number_input("7. TATA (Biến dồn tích)", min_value=-2.0, max_value=2.0, value=float(val_tata), step=0.01, help=RATIO_INFO["TATA"]["desc"])

        with col4:
            val_sgi = sample_vals.get("SGI", RATIO_INFO["SGI"]["default"])
            inputs["SGI"] = st.number_input("4. SGI (Tăng trưởng DT)", min_value=0.0, max_value=10.0, value=float(val_sgi), step=0.05, help=RATIO_INFO["SGI"]["desc"])
            val_lvgi = sample_vals.get("LVGI", RATIO_INFO["LVGI"]["default"])
            inputs["LVGI"] = st.number_input("8. LVGI (Đòn bẩy tài chính)", min_value=0.0, max_value=10.0, value=float(val_lvgi), step=0.05, help=RATIO_INFO["LVGI"]["desc"])

        submit_btn = st.form_submit_button("🚀 Thực hiện Dự báo", use_container_width=True)

    if submit_btn or profile != "Tự nhập chỉ số":
        input_df = pd.DataFrame([inputs])[FEATURES]
        prob_fraud = pipeline.predict_proba(input_df)[0, 1]
        is_fraud = prob_fraud >= threshold
        beneish_score = compute_beneish_mscore(inputs)
        
        st.markdown("---")
        st.markdown("### 📋 Kết quả Phân tích & Đánh giá Rủi ro")
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])
        
        with res_col1:
            st.markdown("**1. Kết luận Mô hình Logistic:**")
            if is_fraud:
                st.markdown('<div class="badge-fraud">⚠️ NGUY CƠ GIAN LẬN CAO</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-safe">✅ AN TOÀN (Ít rủi ro)</div>', unsafe_allow_html=True)
            
            st.metric("Xác suất gian lận P(Fraud)", f"{prob_fraud:.1%}", delta=f"{prob_fraud - threshold:+.1%} so với ngưỡng {threshold:.0%}", delta_color="inverse")
            st.progress(float(prob_fraud))

        with res_col2:
            st.markdown("**2. Đối chiếu Beneish M-Score gốc:**")
            m_status = "Nguy cơ gian lận (M > -1.78)" if beneish_score > -1.78 else "Bình thường (M <= -1.78)"
            m_color = "inverse" if beneish_score > -1.78 else "normal"
            st.metric("Điểm M-Score", f"{beneish_score:.3f}", delta=m_status, delta_color=m_color)
            st.caption("Ngưỡng chuẩn Beneish 1999: M-score > -1.78 là dấu hiệu thao túng BCTC.")

        with res_col3:
            st.markdown("**3. Mức độ Cảnh báo Tổng hợp:**")
            if prob_fraud >= 0.70 or beneish_score > -1.49:
                st.error("🔴 **MỨC ĐỘ NGUY HIỂM CAO:** Cần thực hiện kiểm toán đặc biệt và rà soát kỹ các bút toán dồn tích, khoản phải thu.")
            elif prob_fraud >= threshold or beneish_score > -1.78:
                st.warning("🟡 **MỨC ĐỘ NGUY HIỂM TRUNG BÌNH:** Có nhiều dấu hiệu bất thường so với quy mô ngành, cần giải trình thêm.")
            else:
                st.success("🟢 **RỦI RO THẤP:** Các chỉ số tài chính nằm trong phạm vi dao động lành mạnh.")

        # Biểu đồ độ lệch chuẩn Z-score so sánh với tập mẫu
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📊 So sánh các chỉ số của DN với Trung bình Mẫu dữ liệu")
        train_means = X_train.mean()
        train_stds = X_train.std()
        std_deviations = (input_df.iloc[0] - train_means) / train_stds
        
        fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
        colors = ["#EF4444" if val > 1.0 else ("#10B981" if val < -0.5 else "#3B82F6") for val in std_deviations]
        bars = ax_bar.bar(FEATURES, std_deviations, color=colors)
        ax_bar.axhline(0, color="black", linestyle="--", lw=0.8)
        ax_bar.axhline(1.5, color="red", linestyle=":", lw=1, label="Ngưỡng lệch chuẩn cao (+1.5 SD)")
        ax_bar.set_ylabel("Độ lệch chuẩn (Z-score)")
        ax_bar.set_title("Mức độ lệch của từng chỉ số so với trung bình các DN trong mẫu")
        ax_bar.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_bar)

# -----------------------------------------------------------------------------
# TAB 3: DỰ BÁO HÀNG LOẠT (BATCH PREDICTION)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📁 Dự báo Hàng loạt Danh sách Doanh nghiệp")
    st.write("Tải lên file CSV hoặc Excel chứa danh sách doanh nghiệp (cần có tối thiểu 8 cột chỉ số Beneish).")
    
    # Tải file mẫu CSV
    sample_batch = pd.DataFrame([
        {"Ma_CK": "AAA", "DSRI": 1.25, "GMI": 1.10, "AQI": 1.05, "SGI": 1.30, "DEPI": 1.02, "SGAI": 1.15, "TATA": 0.08, "LVGI": 1.12},
        {"Ma_CK": "BBB", "DSRI": 0.90, "GMI": 0.85, "AQI": 0.95, "SGI": 1.05, "DEPI": 0.98, "SGAI": 0.92, "TATA": -0.05, "LVGI": 0.90},
        {"Ma_CK": "CCC", "DSRI": 1.85, "GMI": 1.60, "AQI": 1.45, "SGI": 1.75, "DEPI": 1.30, "SGAI": 1.40, "TATA": 0.22, "LVGI": 1.55}
    ])
    csv_template = sample_batch.to_csv(index=False).encode('utf-8')
    st.download_button("📄 Tải File CSV Mẫu", data=csv_template, file_name="mau_du_bao_gian_lan.csv", mime="text/csv")
    
    batch_file = st.file_uploader("Nạp file danh sách cần kiểm tra:", type=["csv", "xlsx"])
    if batch_file is not None:
        try:
            if batch_file.name.endswith(".csv"):
                batch_df = pd.read_csv(batch_file)
            else:
                batch_df = pd.read_excel(batch_file)
            
            missing_cols = [c for c in FEATURES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"File thiếu các cột chỉ số sau: {missing_cols}")
            else:
                st.success(f"Đã nạp thành công {len(batch_df)} doanh nghiệp.")
                
                # Dự báo xác suất
                X_batch = batch_df[FEATURES].copy()
                for c in FEATURES:
                    X_batch[c] = pd.to_numeric(X_batch[c], errors="coerce").fillna(X_train[c].median())
                
                probs = pipeline.predict_proba(X_batch)[:, 1]
                preds = (probs >= threshold).astype(int)
                m_scores = [compute_beneish_mscore(row) for _, row in X_batch.iterrows()]
                
                result_df = batch_df.copy()
                result_df["Xac_Suat_Gian_Lan"] = np.round(probs, 4)
                result_df["Phan_Loai"] = ["Gian lận (1)" if p == 1 else "An toàn (0)" for p in preds]
                result_df["Beneish_MScore"] = np.round(m_scores, 4)
                result_df["Muc_Do_Rui_Ro"] = [
                    "Rất cao" if p >= 0.7 else ("Cao" if p >= threshold else ("Trung bình" if p >= 0.3 else "Thấp"))
                    for p in probs
                ]
                
                # Hiển thị số lượng rủi ro
                c_total = len(result_df)
                c_fraud = int(preds.sum())
                
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Tổng DN kiểm tra", f"{c_total:,}")
                col_b2.metric("Số DN bị cảnh báo gian lận", f"{c_fraud:,}")
                col_b3.metric("Tỷ lệ bị cảnh báo", f"{(c_fraud / c_total):.1%}")
                
                st.markdown("##### 📋 Bảng kết quả dự báo chi tiết:")
                st.dataframe(result_df, use_container_width=True)
                
                # Tải kết quả về
                batch_buffer = io.BytesIO()
                with pd.ExcelWriter(batch_buffer, engine="openpyxl") as bwriter:
                    result_df.to_excel(bwriter, sheet_name="Ket_Qua_Du_Bao", index=False)
                
                st.download_button(
                    label="📥 Xuất Toàn Bộ Kết Quả Ra Excel",
                    data=batch_buffer.getvalue(),
                    file_name="Ket_Qua_Du_Bao_Gian_Lan_BCTC.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as err:
            st.error(f"Lỗi khi xử lý file: {err}")

# -----------------------------------------------------------------------------
# TAB 4: CẨM NANG 8 CHỈ SỐ BENEISH
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📖 Cẩm nang Tra cứu 8 Chỉ số Beneish M-Score")
    st.markdown("""
    Giáo sư Messod Beneish (Đại học Indiana, 1999) đã công bố mô hình định lượng phát hiện thao túng lợi nhuận.
    Mô hình dựa trên 8 chỉ số tài chính tính toán từ Báo cáo tài chính 2 năm liên tiếp (năm $t$ và năm $t-1$).
    """)
    
    for key, info in RATIO_INFO.items():
        with st.expander(f"📌 {key} - {info['vi']} ({info['name']})"):
            st.markdown(f"**Ý nghĩa chuyên môn:** {info['desc']}")
            if key == "DSRI":
                st.latex(r"DSRI = \frac{\text{Receivables}_t / \text{Sales}_t}{\text{Receivables}_{t-1} / \text{Sales}_{t-1}}")
            elif key == "GMI":
                st.latex(r"GMI = \frac{\text{Gross Margin}_{t-1}}{\text{Gross Margin}_t}")
            elif key == "AQI":
                st.latex(r"AQI = \frac{1 - (\text{Current Assets}_t + \text{PPE}_t + \text{Securities}_t)/\text{Total Assets}_t}{1 - (\text{Current Assets}_{t-1} + \text{PPE}_{t-1} + \text{Securities}_{t-1})/\text{Total Assets}_{t-1}}")
            elif key == "SGI":
                st.latex(r"SGI = \frac{\text{Sales}_t}{\text{Sales}_{t-1}}")
            elif key == "DEPI":
                st.latex(r"DEPI = \frac{\text{Depreciation Rate}_{t-1}}{\text{Depreciation Rate}_t}")
            elif key == "SGAI":
                st.latex(r"SGAI = \frac{\text{SG\&A}_t / \text{Sales}_t}{\text{SG\&A}_{t-1} / \text{Sales}_{t-1}}")
            elif key == "TATA":
                st.latex(r"TATA = \frac{\text{Net Income}_t - \text{Operating Cash Flow}_t}{\text{Total Assets}_t}")
            elif key == "LVGI":
                st.latex(r"LVGI = \frac{\text{Total Debt}_t / \text{Total Assets}_t}{\text{Total Debt}_{t-1} / \text{Total Assets}_{t-1}}")

    st.markdown("---")
    st.markdown("""
    ### 🔬 Mô hình Hồi quy Logistic kết hợp (Machine Learning)
    So với công thức M-score truyền thống cố định hệ số hồi quy theo dữ liệu thị trường Mỹ năm 1999,
    việc ứng dụng **Logistic Regression với chuẩn hóa StandardScaler** trên tập dữ liệu thực tế:
    1. Giúp mô hình tự học trọng số tối ưu phù hợp với thị trường và phân khúc doanh nghiệp.
    2. Cung cấp trực tiếp **Xác suất gian lận $P(\text{Fraud})$** chính xác thay vì chỉ đưa ra điểm số vô hướng.
    3. Cho phép kiểm toán viên chủ động điều chỉnh **Ngưỡng nhạy cảm (Threshold)** tùy theo mục tiêu quản trị rủi ro.
    """)
