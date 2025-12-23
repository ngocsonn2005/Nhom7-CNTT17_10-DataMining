# Shopping Cart Analysis

Phân tích giỏ hàng bán lẻ bằng **Apriori** và **FP-Growth**. Pipeline tự động hóa từ làm sạch dữ liệu → khai phá luật → so sánh thuật toán.

---

## 📁 Project Structure

```text
shopping_cart_analysis/
├── data/
│   ├── raw/online_retail.csv
│   └── processed/ [cleaned_uk_data.csv, basket_bool.parquet, rules_*.csv]
├── notebooks/
│   ├── preprocessing_and_eda.ipynb        # Bước 1
│   ├── basket_preparation.ipynb           # Bước 2  
│   ├── apriori_modelling.ipynb            # Bước 3a
│   ├── fp_growth_modeling.ipynb           # Bước 3b (NEW)
│   ├── compare_apriori_fpgrowth.ipynb     # Bước 4 (NEW)
│   └── runs/ [*_run.ipynb]
├── src/apriori_library.py                 # Thư viện chính
├── run_papermill.py                       # Pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone <your-repo>
cd shopping_cart_analysis
pip install -r requirements.txt

# Đặt dữ liệu vào: data/raw/online_retail.csv

# Chạy toàn bộ pipeline
python run_papermill.py
```

Pipeline chạy tự động 5 bước:
1. Preprocessing & EDA
2. Basket Preparation  
3. Apriori Modelling
4. FP-Growth Modelling
5. Compare Algorithms

---

## 📊 Output Files

```bash
data/processed/
├── cleaned_uk_data.csv
├── basket_bool.parquet
├── rules_apriori_filtered.csv
└── rules_fpgrowth_filtered.csv    # NEW

notebooks/runs/
├── preprocessing_and_eda_run.ipynb
├── basket_preparation_run.ipynb
├── apriori_modelling_run.ipynb
├── fp_growth_modeling_run.ipynb     # NEW
└── compare_apriori_fpgrowth_run.ipynb  # NEW
```

---

## ⚙️ Customization

Sửa trong `run_papermill.py`:

```python
MIN_SUPPORT = 0.05        # Ngưỡng support tối thiểu
MAX_LEN = 3               # Độ dài itemset tối đa
FILTER_MIN_CONF = 0.3     # Ngưỡng confidence
FILTER_MIN_LIFT = 1.2     # Ngưỡng lift
```

---

## 🛠️ Tech Stack

| Công nghệ | Mục đích |
|-----------|----------|
| Python, Pandas | Xử lý dữ liệu |
| MLxtend | Apriori / FP-Growth |
| Papermill | Tự động hóa pipeline |
| Matplotlib/Seaborn | Visualization |
| Plotly | Interactive charts |
| Jupyter | Notebook environment |

---

## 📄 License

Educational use only. Contact for commercial licensing.

**Author**: Ngoc Son