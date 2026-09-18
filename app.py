import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dự báo Gian lận BCTC - Beneish M-Score",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #3B82F6;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Definitions
# ---------------------------------------------------------
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

FEATURE_NAMES_VI = {
    "DSRI": "DSRI - Days Sales in Receivables Index (Số ngày thu tiền khách hàng)",
    "GMI": "GMI - Gross Margin Index (Tỷ lệ lãi gộp)",
    "AQI": "AQI - Asset Quality Index (Chất lượng tài sản)",
    "SGI": "SGI - Sales Growth Index (Tăng trưởng doanh thu)",
    "DEPI": "DEPI - Depreciation Index (Khấu hao tài sản)",
    "SGAI": "SGAI - Sales, General & Admin Expenses Index (Chi phí QLDN & Bán hàng)",
    "TATA": "TATA - Total Accruals to Total Assets (Dồn tích trên tổng tài sản)",
    "LVGI": "LVGI - Leverage Index (Đòn bẩy tài chính)"
}

FEATURE_DESCRIPTIONS = {
    "DSRI": "Tỷ lệ số ngày thu tiền khách hàng năm t so với năm t-1. DSRI > 1 cho thấy khoản phải thu tăng nhanh hơn doanh thu (dấu hiệu ghi nhận doanh thu ảo).",
    "GMI": "Tỷ lệ lãi gộp năm t-1 so với năm t. GMI > 1 phản ánh tỷ lệ lãi gộp suy giảm (động cơ gian lận để duy trì tăng trưởng lợi nhuận).",
    "AQI": "Tỷ lệ tài sản phi tiền tệ & phi tài sản cố định hữu hình. AQI > 1 cho thấy công ty tăng vốn hóa chi phí vào tài sản phi hữu hình.",
    "SGI": "Tỷ lệ doanh thu năm t so với năm t-1. Tăng trưởng quá nóng (SGI > 1) làm tăng áp lực duy trì lợi nhuận.",
    "DEPI": "Tỷ lệ mức khấu hao năm t-1 so với năm t. DEPI > 1 cho thấy tỷ lệ khấu hao giảm (kéo dài thời gian khấu hao để tăng lợi nhuận).",
    "SGAI": "Tỷ lệ chi phí bán hàng và QLDN trên doanh thu năm t so với năm t-1. SGAI > 1 cho thấy hiệu quả quản lý chi phí giảm.",
    "TATA": "Biến dồn tích dồn trên tổng tài sản. TATA cao phản ánh lợi nhuận đến từ dồn tích kế toán thay vì dòng tiền thực tế.",
    "LVGI": "Tỷ lệ tổng nợ trên tổng tài sản năm t so với năm t-1. LVGI > 1 cho thấy đòn bẩy tài chính tăng, làm tăng rủi ro tài chính."
}

DEFAULT_BENCHMARKS = {
    "DSRI": 1.0,
    "GMI": 1.0,
    "AQI": 1.0,
    "SGI": 1.0,
    "DEPI": 1.0,
    "SGAI": 1.0,
    "TATA": 0.02,
    "LVGI": 1.0
}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    """Tải dữ liệu mặc định MScore_data.csv nếu có"""
    file_path = "MScore_data.csv"
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"Lỗi đọc file {file_path}: {e}")
            return None
    return None

def clean_data(df):
    """Làm sạch dữ liệu đầu vào"""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"File CSV thiếu các cột bắt buộc: {missing}")
        return None
    
    data = df[FEATURES + [TARGET]].copy()
    for c in FEATURES + [TARGET]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    
    data = data.dropna().reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)
    
    if not set(data[TARGET].unique()).issubset({0, 1}):
        st.error("Cột FRAUD_FLAG chỉ được chứa các giá trị 0 hoặc 1.")
        return None
        
    return data

def train_model(data, test_size=0.20, random_state=42, threshold=0.50):
    """Huấn luyện mô hình Logistic Regression với Pipeline StandardScaler"""
    X = data[FEATURES]
    y = data[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    
    model.fit(X_train, y_train)
    
    # Dự báo test set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Đánh giá metrics
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Coefficients
    logistic_step = model.named_steps["logistic"]
    intercept = logistic_step.intercept_[0]
    coef = logistic_step.coef_[0]
    
    coef_table = pd.DataFrame({
        "Biến": FEATURES,
        "Hệ số (Beta)": coef,
        "Odds Ratio (e^Beta)": np.exp(coef)
    })
    
    metrics_df = pd.DataFrame({
        "Chỉ tiêu": [
            "Accuracy (Độ chính xác)",
            "Precision (Độ chuẩn xác gian lận)",
            "Recall / Sensitivity (Tỷ lệ phát hiện gian lận)",
            "Specificity (Độ đặc hiệu)",
            "F1-Score",
            "False Positive Rate (FPR - Báo động giả)",
            "False Negative Rate (FNR - Bỏ sót gian lận)"
        ],
        "Giá trị": [accuracy, precision, recall, specificity, f1, fpr, fnr],
        "Giá trị (%)": [f"{accuracy*100:.2f}%", f"{precision*100:.2f}%", f"{recall*100:.2f}%", 
                        f"{specificity*100:.2f}%", f"{f1*100:.2f}%", f"{fpr*100:.2f}%", f"{fnr*100:.2f}%"]
    })
    
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cm": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "specificity": specificity, "fpr": fpr, "fnr": fnr,
        "intercept": intercept, "coef": coef,
        "coef_table": coef_table,
        "metrics_df": metrics_df,
        "threshold": threshold
    }


def train_xgboost_model(data, test_size=0.20, random_state=42, threshold=0.50):
    """Huấn luyện XGBoost trên cùng cách chia Train/Test."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost chưa được cài đặt. Chạy: pip install xgboost")
    X=data[FEATURES]; y=data[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    n0=(y_train==0).sum(); n1=(y_train==1).sum(); spw=n0/n1 if n1 else 1.0
    model=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,objective="binary:logistic",eval_metric="logloss",random_state=random_state,scale_pos_weight=spw,n_jobs=-1)
    model.fit(X_train,y_train)
    y_prob=model.predict_proba(X_test)[:,1]; y_pred=(y_prob>=threshold).astype(int)
    cm=confusion_matrix(y_test,y_pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    acc=accuracy_score(y_test,y_pred); pre=precision_score(y_test,y_pred,zero_division=0); rec=recall_score(y_test,y_pred,zero_division=0); f1=f1_score(y_test,y_pred,zero_division=0)
    spec=tn/(tn+fp) if tn+fp else 0.0; fpr=fp/(fp+tn) if fp+tn else 0.0; fnr=fn/(fn+tp) if fn+tp else 0.0
    metrics_df=pd.DataFrame({"Chỉ tiêu":["Accuracy","Precision","Recall / Sensitivity","Specificity","F1-Score","False Positive Rate (FPR)","False Negative Rate (FNR)"],"Giá trị":[acc,pre,rec,spec,f1,fpr,fnr],"Giá trị (%)":[f"{v*100:.2f}%" for v in [acc,pre,rec,spec,f1,fpr,fnr]]})
    importance=pd.DataFrame({"Biến":FEATURES,"Feature Importance":model.feature_importances_}).sort_values("Feature Importance",ascending=False).reset_index(drop=True)
    return {"model":model,"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"y_prob":y_prob,"y_pred":y_pred,"cm":cm,"tn":tn,"fp":fp,"fn":fn,"tp":tp,"accuracy":acc,"precision":pre,"recall":rec,"f1":f1,"specificity":spec,"fpr":fpr,"fnr":fnr,"metrics_df":metrics_df,"feature_importance":importance,"scale_pos_weight":spw,"threshold":threshold}

def calculate_beneish_mscore(dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi):
    """Tính điểm Beneish M-Score chuẩn theo phương trình gốc:
    M = -2.22 + 0.757*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    Ngưỡng phân loại: M > -1.78 -> Có nguy cơ gian lận cao
    """
    m_score = (-2.22 
               + 0.757 * dsri 
               + 0.528 * gmi 
               + 0.404 * aqi 
               + 0.892 * sgi 
               + 0.115 * depi 
               - 0.172 * sgai 
               + 4.679 * tata 
               - 0.327 * lvgi)
    return m_score

def export_results_to_excel(results):
    """Xuất toàn bộ kết quả mô hình ra file Excel in-memory"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results["coef_table"].to_excel(writer, sheet_name="Coefficients", index=False)
        
        cm_df = pd.DataFrame(
            results["cm"],
            index=["Thực tế: Không gian lận (0)", "Thực tế: Gian lận (1)"],
            columns=["Dự báo: Không gian lận (0)", "Dự báo: Gian lận (1)"]
        )
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        
        results["metrics_df"].to_excel(writer, sheet_name="Metrics", index=False)
        
        # Diễn giải
        interp = []
        for name, b in zip(FEATURES, results["coef"]):
            or_val = np.exp(b)
            meaning = f"{name} tăng 1 độ lệch chuẩn làm {'tăng' if b > 0 else 'giảm'} log-odds gian lận {abs(b):.4f} (Odds ratio: {or_val:.4f})"
            interp.append({"Biến": name, "Hệ số Beta": b, "Odds Ratio": or_val, "Diễn giải": meaning})
        pd.DataFrame(interp).to_excel(writer, sheet_name="Interpretation", index=False)
        
        pd.DataFrame({
            "Thông số": ["Intercept", "Threshold", "Train Size", "Test Size"],
            "Giá trị": [results["intercept"], results["threshold"], len(results["X_train"]), len(results["X_test"])]
        }).to_excel(writer, sheet_name="Model_Info", index=False)
        
    return output.getvalue()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/bar-chart--v1.png", width=80)
st.sidebar.title("⚙️ Cấu hình Ứng dụng")

st.sidebar.subheader("1. Dữ liệu huấn luyện")
data_source = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    options=["Dữ liệu mặc định (MScore_data.csv)", "Tải lên file CSV mới"],
    index=0
)

df_raw = None
if data_source == "Dữ liệu mặc định (MScore_data.csv)":
    df_raw = load_default_data()
    if df_raw is None:
        st.sidebar.warning("Không tìm thấy file `MScore_data.csv` trên máy chủ. Vui lòng tải lên file CSV!")
else:
    uploaded_file = st.sidebar.file_uploader("Tải lên file CSV (gồm 8 biến Beneish + FRAUD_FLAG)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Không thể đọc file: {e}")

st.sidebar.subheader("2. Tham số mô hình Logistic")
test_size = st.sidebar.slider("Tỷ lệ tập Test (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
threshold = st.sidebar.slider("Ngưỡng quyết định (Decision Threshold)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">🔍 Web App Dự Báo Gian Lận Báo Cáo Tài Chính</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Machine Learning dựa trên 8 chỉ số Beneish M-Score & Mô hình Logistic Regression</div>', unsafe_allow_html=True)

if df_raw is None:
    st.info("👋 **Chào mừng bạn!** Vui lòng kiểm tra lại file `MScore_data.csv` hoặc tải lên dữ liệu CSV ở thanh bên trái (Sidebar) để bắt đầu phân tích.")
    st.stop()

# Clean data
data = clean_data(df_raw)
if data is None:
    st.stop()

# Train model
results = train_model(data, test_size=test_size, random_state=random_seed, threshold=threshold)

xgb_results = None
if XGBOOST_AVAILABLE:
    try:
        xgb_results = train_xgboost_model(data, test_size=test_size, random_state=random_seed, threshold=threshold)
    except Exception as e:
        st.sidebar.warning(f"Không thể huấn luyện XGBoost: {e}")
else:
    st.sidebar.warning("XGBoost chưa được cài. Chạy: pip install xgboost")

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan Dữ liệu (EDA)",
    "🤖 Logistic Regression",
    "🌳 XGBoost",
    "🏢 Dự báo Công ty Đơn lẻ",
    "📁 Dự báo Hàng loạt (Batch)"
])

# =========================================================
# TAB 1: EDA & DATA OVERVIEW
# =========================================================
with tab1:
    st.subheader("📌 Tổng quan Tập Dữ Liệu")
    
    col1, col2, col3, col4 = st.columns(4)
    total_obs = len(data)
    fraud_count = data[TARGET].sum()
    non_fraud_count = total_obs - fraud_count
    fraud_rate = (fraud_count / total_obs) * 100
    
    with col1:
        st.markdown(f'''
        <div class="metric-box">
            <div class="metric-value">{total_obs:,}</div>
            <div class="metric-label">Tổng số quan sát</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-box" style="border-left-color: #EF4444;">
            <div class="metric-value" style="color: #DC2626;">{fraud_count:,}</div>
            <div class="metric-label">Số DN Gian lận (FLAG=1)</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-box" style="border-left-color: #10B981;">
            <div class="metric-value" style="color: #059669;">{non_fraud_count:,}</div>
            <div class="metric-label">Số DN Không gian lận (FLAG=0)</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="metric-box" style="border-left-color: #F59E0B;">
            <div class="metric-value" style="color: #D97706;">{fraud_rate:.2f}%</div>
            <div class="metric-label">Tỷ lệ gian lận trong mẫu</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.write("##### 📋 Xem trước Dữ liệu (Head)")
        st.dataframe(data.head(10), use_container_width=True)
        
    with col_right:
        st.write("##### 📈 Thống kê Mô tả các Biến Beneish")
        st.dataframe(data[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"}), use_container_width=True)

    st.markdown("---")
    st.write("##### 📉 Biểu đồ Ma trận Tương quan (Correlation Matrix)")
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
    corr = data[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr, cbar=True)
    plt.title("Ma trận Tương quan giữa 8 biến Beneish và FRAUD_FLAG")
    st.pyplot(fig_corr)
    
    st.markdown("---")
    st.write("##### 📦 So sánh Phân bố 8 Biến theo Trạng thái Gian lận")
    selected_feature = st.selectbox("Chọn biến Beneish để quan sát:", FEATURES, format_func=lambda x: FEATURE_NAMES_VI[x])
    
    fig_box, ax_box = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(x=TARGET, y=selected_feature, data=data, palette={0: "#10B981", 1: "#EF4444"}, ax=ax_box)
    ax_box.set_xticklabels(["Không gian lận (0)", "Gian lận (1)"])
    ax_box.set_title(f"Phân bố biến {selected_feature} theo Trạng thái Gian lận")
    st.pyplot(fig_box)

# =========================================================
# TAB 2: MODEL TRAINING & EVALUATION
# =========================================================
with tab2:
    st.subheader("🤖 Kết quả Huấn luyện & Đánh giá Mô hình Logistic Regression")
    
    st.write(f"**Tập dữ liệu:** Train size = **{len(results['X_train'])}** quan sát | Test size = **{len(results['X_test'])}** quan sát (Tỷ lệ split: {int((1-test_size)*100)}/{int(test_size*100)}) | Threshold = **{threshold:.2f}**")
    
    # Render metric summary cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Accuracy (Độ chính xác)", f"{results['accuracy']:.2%}")
        st.metric("Specificity (Độ đặc hiệu)", f"{results['specificity']:.2%}")
    with m_col2:
        st.metric("Precision (Độ chuẩn xác)", f"{results['precision']:.2%}")
        st.metric("False Positive Rate (FPR)", f"{results['fpr']:.2%}")
    with m_col3:
        st.metric("Recall (Độ nhạy)", f"{results['recall']:.2%}")
        st.metric("False Negative Rate (FNR)", f"{results['fnr']:.2%}")
    with m_col4:
        st.metric("F1-Score", f"{results['f1']:.2%}")
        
    st.markdown("---")
    
    col_coef, col_cm = st.columns([1.1, 0.9])
    
    with col_coef:
        st.write("##### ⚖️ Hệ số Mô hình & Odds Ratio")
        st.write(f"**Intercept ($\beta_0$)** = `{results['intercept']:.6f}`")
        st.dataframe(results["coef_table"], use_container_width=True)
        
        # Phương trình Logit
        eq = f"logit(P) = {results['intercept']:.4f}"
        for name, b in zip(FEATURES, results["coef"]):
            eq += f" {'+' if b >= 0 else '-'} {abs(b):.4f} \\times \\text{{{name}}}_{{std}}"
        st.latex(eq)
        
    with col_cm:
        st.write("##### 🧩 Ma trận Nhầm lẫn (Confusion Matrix)")
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        ConfusionMatrixDisplay(
            confusion_matrix=results["cm"],
            display_labels=["Không gian lận", "Gian lận"]
        ).plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
        ax_cm.set_xlabel("Dự báo")
        ax_cm.set_ylabel("Thực tế")
        st.pyplot(fig_cm)
        
        st.caption(f"**TN**: {results['tn']} | **FP**: {results['fp']} | **FN**: {results['fn']} | **TP**: {results['tp']}")

    st.markdown("---")
    
    st.write("##### 📊 Trực quan hóa Hệ số Beta của các Biến")
    fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5))
    coef_df = results["coef_table"].sort_values(by="Hệ số (Beta)", ascending=True)
    colors = ["#EF4444" if val > 0 else "#10B981" for val in coef_df["Hệ số (Beta)"]]
    ax_bar.barh(coef_df["Biến"], coef_df["Hệ số (Beta)"], color=colors)
    ax_bar.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_bar.set_title("Hệ số Beta (Tác động của biến đã chuẩn hóa tới log-odds gian lận)")
    ax_bar.set_xlabel("Giá trị Hệ số Beta")
    st.pyplot(fig_bar)
    
    st.markdown("---")
    st.write("##### 📝 Diễn giải Chi tiết Hệ số & Ý nghĩa Kế toán")
    
    interp_data = []
    for name, b in zip(FEATURES, results["coef"]):
        or_val = np.exp(b)
        if b > 0:
            meaning = f"Khi **{name}** tăng 1 độ lệch chuẩn, log-odds gian lận tăng **{b:.4f}**, Odds gian lận tăng gấp **{or_val:.4f} lần**."
        elif b < 0:
            meaning = f"Khi **{name}** tăng 1 độ lệch chuẩn, log-odds gian lận giảm **{abs(b):.4f}**, Odds gian lận được nhân **{or_val:.4f} lần**."
        else:
            meaning = f"Biến {name} không có tác động đáng kể."
        interp_data.append({"Biến": name, "Ý nghĩa & Tác động": meaning, "Mô tả chỉ số": FEATURE_DESCRIPTIONS[name]})
        
    st.dataframe(pd.DataFrame(interp_data), use_container_width=True)

    st.markdown("---")
    st.write("##### 📥 Xuất Báo cáo Kết quả Mô hình")
    excel_bytes = export_results_to_excel(results)
    st.download_button(
        label="📥 Tải Báo cáo Kết quả Chi tiết (Excel)",
        data=excel_bytes,
        file_name="Logistic_Regression_MScore_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================================================
# TAB 3: SINGLE COMPANY PREDICTION
# =========================================================
with tab4:
    st.subheader("🏢 Dự báo Nguy cơ Gian lận cho 1 Công ty Cụ thể")
    st.write("Nhập các chỉ số tài chính tính toán từ BCTC của doanh nghiệp để đưa ra dự báo:")

    col_in1, col_in2, col_in3, col_in4 = st.columns(4)
    
    input_vals = {}
    with col_in1:
        input_vals["DSRI"] = st.number_input("DSRI (Phải thu/DT)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["DSRI"])
        input_vals["DEPI"] = st.number_input("DEPI (Tỷ lệ khấu hao)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["DEPI"])
    with col_in2:
        input_vals["GMI"] = st.number_input("GMI (Tỷ lệ lãi gộp)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["GMI"])
        input_vals["SGAI"] = st.number_input("SGAI (Chi phí QLDN)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["SGAI"])
    with col_in3:
        input_vals["AQI"] = st.number_input("AQI (Chất lượng TS)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["AQI"])
        input_vals["TATA"] = st.number_input("TATA (Dồn tích/Tổng TS)", min_value=-2.0, max_value=2.0, value=0.02, step=0.01, help=FEATURE_DESCRIPTIONS["TATA"])
    with col_in4:
        input_vals["SGI"] = st.number_input("SGI (Tăng trưởng DT)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["SGI"])
        input_vals["LVGI"] = st.number_input("LVGI (Đòn bẩy tài chính)", min_value=0.0, max_value=10.0, value=1.00, step=0.05, help=FEATURE_DESCRIPTIONS["LVGI"])

    st.markdown("---")
    
    if st.button("🚀 Thực hiện Dự báo Gian lận", type="primary"):
        input_df = pd.DataFrame([input_vals])[FEATURES]
        
        # Predict using Logistic Regression Model
        model_pipeline = results["model"]
        fraud_prob = model_pipeline.predict_proba(input_df)[0, 1]
        is_fraud_pred = fraud_prob >= threshold

        xgb_fraud_prob = None
        xgb_is_fraud_pred = None
        if xgb_results is not None:
            xgb_fraud_prob = xgb_results["model"].predict_proba(input_df)[0, 1]
            xgb_is_fraud_pred = xgb_fraud_prob >= threshold
        
        # Calculate Traditional Beneish M-Score formula
        m_score_val = calculate_beneish_mscore(
            input_vals["DSRI"], input_vals["GMI"], input_vals["AQI"], input_vals["SGI"],
            input_vals["DEPI"], input_vals["SGAI"], input_vals["TATA"], input_vals["LVGI"]
        )
        is_beneish_fraud = m_score_val > -1.78

        st.write("### 📋 Kết quả Đánh giá Rủi ro")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown("##### 🤖 1. Kết quả từ Mô hình Logistic Regression")
            st.progress(float(fraud_prob))
            st.write(f"Xác suất Gian lận BCTC: **{fraud_prob:.2%}** (Ngưỡng phân loại: {threshold:.2%})")
            
            if is_fraud_pred:
                st.error("🚨 **CẢNH BÁO RỦI RO CAO:** Doanh nghiệp có dấu hiệu gian lận BCTC theo mô hình Logistic Regression!")
            else:
                st.success("✅ **AN TOÀN / RỦI RO THẤP:** Chưa phát hiện dấu hiệu gian lận bất thường theo mô hình Logistic Regression.")
                
        with res_col2:
            st.markdown("##### 🌳 2. Kết quả XGBoost")
            if xgb_fraud_prob is not None:
                st.progress(float(xgb_fraud_prob))
                st.write(f"Xác suất Gian lận BCTC: **{xgb_fraud_prob:.2%}** (Ngưỡng: {threshold:.2%})")
                if xgb_is_fraud_pred:
                    st.error("🚨 XGBoost phân loại doanh nghiệp vào nhóm gian lận!")
                else:
                    st.success("✅ XGBoost phân loại doanh nghiệp vào nhóm không gian lận.")
            else:
                st.info("XGBoost chưa khả dụng.")

        with res_col3:
            st.markdown("##### 🧮 3. Kết quả Beneish M-Score truyền thống")
            st.metric("Điểm M-Score", f"{m_score_val:.4f}", delta="Cảnh báo (> -1.78)" if is_beneish_fraud else "An toàn (<= -1.78)")
            
            if is_beneish_fraud:
                st.error("⚠️ **Beneish M-Score > -1.78**: Doanh nghiệp thuộc diện có nguy cơ thao túng BCTC cao!")
            else:
                st.success("👍 **Beneish M-Score ≤ -1.78**: Điểm số nằm trong ngưỡng bình thường.")
                
        st.markdown("---")
        st.write("##### 🔬 Phân tích Chi tiết từng Chỉ số so với Ngưỡng Cảnh báo:")
        
        warnings_list = []
        if input_vals["DSRI"] > 1.2:
            warnings_list.append("🔴 **DSRI cao (> 1.2)**: Các khoản phải thu tăng đột biến so với doanh thu. Cần kiểm tra xem có ghi nhận doanh thu ảo/sớm hay không.")
        if input_vals["GMI"] > 1.1:
            warnings_list.append("🔴 **GMI cao (> 1.1)**: Tỷ lệ lãi gộp suy giảm đáng kể so với năm trước. Áp lực gian lận để bù đắp sụt giảm lợi nhuận.")
        if input_vals["AQI"] > 1.2:
            warnings_list.append("🔴 **AQI cao (> 1.2)**: Tỷ lệ tài sản phi tài chính tăng cao. Cần kiểm tra vốn hóa chi phí hoặc tài sản vô hình ảo.")
        if input_vals["SGI"] > 1.3:
            warnings_list.append("🔴 **SGI cao (> 1.3)**: Tăng trưởng doanh thu quá nhanh. Cần xác minh chất lượng tăng trưởng.")
        if input_vals["DEPI"] > 1.1:
            warnings_list.append("🔴 **DEPI cao (> 1.1)**: Tỷ lệ khấu hao giảm. Cần kiểm tra xem DN có kéo dài thời gian khấu hao tài sản trái quy định để tăng lợi nhuận.")
        if input_vals["TATA"] > 0.08:
            warnings_list.append("🔴 **TATA cao (> 0.08)**: Biến dồn tích cao thể hiện lợi nhuận báo cáo không đi kèm dòng tiền từ hoạt động kinh doanh (CFO).")
            
        if warnings_list:
            for w in warnings_list:
                st.warning(w)
        else:
            st.info("ℹ️ Tất cả các chỉ số đầu vào đều nằm trong vùng dao động tương đối an toàn.")

# =========================================================
# TAB 4: BATCH PREDICTION
# =========================================================
with tab5:
    st.subheader("📁 Dự báo Gian lận Hàng loạt (Batch Prediction)")
    st.write("Tải lên file CSV chứa dữ liệu của nhiều doanh nghiệp để dự báo đồng thời.")
    
    st.markdown("""
    **Yêu cầu định dạng file CSV:**
    - Phải chứa đủ 8 cột: `DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `TATA`, `LVGI`.
    - Tùy chọn cột định danh doanh nghiệp: `Mã DN` hoặc `Company` (nếu không có ứng dụng sẽ tự đánh số).
    """)
    
    batch_file = st.file_uploader("Tải lên file CSV kiểm tra hàng loạt", type=["csv"], key="batch_uploader")
    
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_batch = [c for c in FEATURES if c not in batch_df.columns]
            
            if missing_batch:
                st.error(f"File CSV thiếu các cột bắt buộc: {missing_batch}")
            else:
                st.success(f"Đã đọc file thành công! Số lượng quan sát: **{len(batch_df)}** doanh nghiệp.")
                
                # Preprocess batch features
                X_batch = batch_df[FEATURES].copy()
                for c in FEATURES:
                    X_batch[c] = pd.to_numeric(X_batch[c], errors="coerce").fillna(0.0)
                
                # Predict
                model_pipeline = results["model"]
                batch_probs = model_pipeline.predict_proba(X_batch)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)
                
                # Add predictions to DataFrame
                result_batch_df = batch_df.copy()
                result_batch_df["Fraud_Probability (%)"] = np.round(batch_probs * 100, 2)
                result_batch_df["Predict_Risk"] = np.where(batch_preds == 1, "🔴 Rủi ro cao (Gian lận)", "🟢 An toàn")
                
                # Calculate Beneish M-Score
                m_scores = [
                    calculate_beneish_mscore(row["DSRI"], row["GMI"], row["AQI"], row["SGI"], row["DEPI"], row["SGAI"], row["TATA"], row["LVGI"])
                    for _, row in X_batch.iterrows()
                ]
                result_batch_df["Beneish_MScore"] = np.round(m_scores, 4)
                result_batch_df["Beneish_Warning"] = np.where(np.array(m_scores) > -1.78, "⚠️ Cảnh báo M > -1.78", "OK")
                
                # Display metrics
                b_col1, b_col2, b_col3 = st.columns(3)
                total_b = len(result_batch_df)
                high_risk_b = (batch_preds == 1).sum()
                low_risk_b = total_b - high_risk_b
                
                with b_col1:
                    st.metric("Tong so DN phân tích", f"{total_b}")
                with b_col2:
                    st.metric("DN Cảnh báo Rủi ro cao", f"{high_risk_b}", delta=f"{(high_risk_b/total_b)*100:.1f}%")
                with b_col3:
                    st.metric("DN An toàn", f"{low_risk_b}")
                    
                st.markdown("---")
                st.write("##### 📋 Kết quả Dự báo Hàng loạt")
                st.dataframe(result_batch_df, use_container_width=True)
                
                # Download batch results CSV
                csv_data = result_batch_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải xuống Kết quả Dự báo Batch (CSV)",
                    data=csv_data,
                    file_name="Batch_Fraud_Prediction_Results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi xử lý file: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("© 2026 Financial Fraud Detection App | Xây dựng dựa trên 8 biến Beneish M-Score, Logistic Regression & XGBoost.")
