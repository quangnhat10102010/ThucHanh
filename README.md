# ⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (BCTC)
### Ứng dụng Streamlit Web App: Beneish M-Score & Hồi quy Logistic

Ứng dụng web thông minh hỗ trợ Kiểm toán viên, Chuyên viên phân tích rủi ro tín dụng và Nhà đầu tư phát hiện sớm các dấu hiệu thao túng, bóp méo Báo cáo tài chính (BCTC) của doanh nghiệp thông qua việc kết hợp **8 chỉ số tài chính Beneish M-Score (1999)** và thuật toán học máy **Hồi quy Logistic (Logistic Regression)**.

---

## 🌟 Các Tính Năng Nổi Bật

1. **📊 Huấn luyện & Đánh giá Mô hình:**
   - Tự động chia tập dữ liệu `Train/Test` (80/20) với kỹ thuật phân tầng (`stratified`).
   - Trực quan hóa **Ma trận Nhầm lẫn (Confusion Matrix)** và **Đường cong ROC (AUC-ROC)**.
   - Đo lường đầy đủ các chỉ số độ chính xác: `Accuracy`, `Precision`, `Recall / Sensitivity`, `F1-Score`, `Specificity`.
   - Phân tích độ tác động của từng chỉ số qua hệ số hồi quy $\beta$ và tỷ số số chênh **Odds Ratio ($e^\beta$)**.
   - Trích xuất toàn bộ kết quả mô hình ra file Excel (`.xlsx`).

2. **🔍 Dự báo Doanh nghiệp Đơn lẻ (Interactive Prediction):**
   - Nhập liệu 8 chỉ số tài chính tính từ BCTC hoặc chọn nhanh các hồ sơ mẫu (Doanh nghiệp bình thường / Doanh nghiệp rủi ro cao).
   - Tính toán xác suất gian lận $P(\text{Fraud})$ kèm phân loại mức độ rủi ro trực quan (Thấp, Trung bình, Cao, Rất cao).
   - Tùy biến **Ngưỡng phân loại (Cut-off Threshold)** ngay trên thanh công cụ.
   - **Đối chiếu song song** kết quả dự báo học máy với điểm số **Beneish M-Score gốc (1999)**.
   - Biểu đồ Z-score so sánh mức độ lệch của từng chỉ số so với trung bình các doanh nghiệp trong mẫu nghiên cứu.

3. **📁 Dự báo Hàng loạt (Batch Prediction):**
   - Tải lên file Excel/CSV danh sách nhiều doanh nghiệp cần thẩm định cùng lúc.
   - Có sẵn file CSV mẫu để tải về xem định dạng chuẩn.
   - Tự động quét, tính xác suất, gắn nhãn rủi ro và xuất file Excel kết quả hoàn chỉnh chỉ với 1 cú click chuột.

4. **📖 Cẩm nang 8 Chỉ số Beneish:**
   - Tra cứu chi tiết ý nghĩa kiểm toán, công thức tính toán và các dấu hiệu cảnh báo đỏ (Red Flags) cho từng chỉ số: `DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `TATA`, `LVGI`.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
TaoAppThucHanh/
│
├── app.py                     # Mã nguồn ứng dụng web Streamlit
├── requirements.txt           # Danh sách các thư viện Python cần thiết
├── MScore_data.csv            # File dữ liệu huấn luyện mẫu (8 chỉ số + FRAUD_FLAG)
├── README.md                  # Hướng dẫn chi tiết cài đặt, sử dụng & deploy
└── logistic_regression_mscore_colab.py # File script gốc từ Colab (tham khảo)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Đã cài đặt **Python 3.9** trở lên (khuyến nghị 3.10 hoặc 3.11).
- Trình duyệt web (Chrome, Edge, Firefox...).

### 2. Các bước khởi chạy
Mở terminal (Command Prompt hoặc PowerShell trên Windows, Terminal trên macOS/Linux) tại thư mục dự án và thực hiện:

```bash
# 1. Tạo môi trường ảo (tùy chọn nhưng khuyến nghị)
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 3. Khởi chạy ứng dụng Streamlit
streamlit run app.py
```

Sau khi chạy lệnh trên, trình duyệt web sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Tải Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 1: Đưa mã nguồn lên GitHub

1. Truy cập [GitHub](https://github.com) và đăng nhập tài khoản của bạn.
2. Nhấn nút **New** (Tạo repository mới):
   - Đặt tên Repository: ví dụ `du-bao-gian-lan-bctc`.
   - Chọn chế độ: **Public**.
   - Không cần tích chọn tạo README (vì dự án đã có sẵn). Nhấn **Create repository**.
3. Mở terminal tại thư mục chứa 3 file trên máy tính của bạn và chạy các lệnh Git sau:

```bash
# Khởi tạo git
git init

# Thêm tất cả các file vào git
git add app.py requirements.txt README.md MScore_data.csv

# Lưu commit đầu tiên
git commit -m "Khoi tao du an Web App Du bao Gian lan BCTC"

# Đổi tên nhánh sang main
git branch -M main

# Liên kết với repo trên GitHub của bạn (thay username và repo name tương ứng)
git remote add origin https://github.com/<TEN_TAI_KHOAN_GITHUB>/du-bao-gian-lan-bctc.git

# Đẩy code lên GitHub
git push -u origin main
```

---

### Bước 2: Deploy lên Streamlit Community Cloud (Hoàn toàn miễn phí)

1. Truy cập trang: [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản **GitHub**.
2. Nhấn vào nút **Create app** (hoặc **New app**).
3. Điền các thông tin:
   - **Repository:** Chọn repository vừa đẩy lên (ví dụ: `<TEN_TAI_KHOAN_GITHUB>/du-bao-gian-lan-bctc`).
   - **Branch:** `main`.
   - **Main file path:** `app.py`.
   - **App URL (tùy chọn):** Bạn có thể tùy chỉnh tên miền con cho app của mình (ví dụ: `du-bao-gian-lan-bctc.streamlit.app`).
4. Nhấn **Deploy!**.
5. Đợi 1-2 phút để Streamlit Cloud tự động cài đặt các thư viện trong `requirements.txt` và khởi động app. Sau khi hoàn tất, bạn sẽ nhận được đường link web để chia sẻ cho bất kỳ ai truy cập trực tiếp!

---

## 📊 Tóm Tắt 8 Chỉ Số Beneish M-Score

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ngưỡng rủi ro |
| :--- | :--- | :--- | :--- |
| **DSRI** | Days Sales in Receivables Index | Chỉ số số ngày thu tiền khách hàng | $> 1$: Khoản phải thu tăng nhanh bất thường so với doanh thu |
| **GMI** | Gross Margin Index | Chỉ số tỷ suất lãi gộp | $> 1$: Biên lợi nhuận suy giảm, tạo động cơ làm đẹp số liệu |
| **AQI** | Asset Quality Index | Chỉ số chất lượng tài sản | $> 1$: Tăng tỷ trọng chi phí hoãn lại/vốn hóa chi phí vào tài sản |
| **SGI** | Sales Growth Index | Chỉ số tăng trưởng doanh thu | $> 1$: Tăng trưởng nóng tạo áp lực duy trì số liệu kỳ vọng |
| **DEPI** | Depreciation Index | Chỉ số mức khấu hao | $> 1$: Kéo dài thời gian khấu hao để giảm chi phí, tăng lãi |
| **SGAI** | Sales, General & Administrative Index | Chỉ số chi phí bán hàng & QLDN | $> 1$: Tỷ lệ chi phí hoạt động tăng, hiệu quả quản lý sụt giảm |
| **TATA** | Total Accruals to Total Assets | Biến dồn tích trên tổng tài sản | Giá trị dương cao: Lợi nhuận không đi kèm dòng tiền kinh doanh |
| **LVGI** | Leverage Index | Chỉ số đòn bẩy tài chính | $> 1$: Nợ vay tăng cao, áp lực vi phạm các cam kết tín dụng |

- **Công thức tính Beneish M-Score gốc (1999):**
  $$M = -4.84 + 0.920 \cdot DSRI + 0.528 \cdot GMI + 0.404 \cdot AQI + 0.892 \cdot SGI + 0.115 \cdot DEPI - 0.172 \cdot SGAI + 4.037 \cdot TATA + 0.0327 \cdot LVGI$$
  *Quy tắc chuẩn:* Nếu $M > -1.78$, doanh nghiệp có khả năng cao đang thao túng Báo cáo tài chính.

---

## 🛡️ Bản quyền & Đóng góp
Dự án được xây dựng phục vụ mục đích nghiên cứu, học thuật, thực hành phân tích tài chính và hỗ trợ kiểm toán. Mọi đóng góp hoặc ý kiến xây dựng đều được hoan nghênh qua GitHub Issues/Pull Requests.
