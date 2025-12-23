import papermill as pm
import os

os.makedirs("notebooks/runs", exist_ok=True)

# ============================================================================
# BƯỚC 1: PREPROCESSING AND EDA
# ============================================================================
print("\n" + "="*60)
print("BƯỚC 1: PREPROCESSING AND EDA")
print("="*60)

pm.execute_notebook(
    "notebooks/preprocessing_and_eda.ipynb",
    "notebooks/runs/preprocessing_and_eda_run.ipynb",
    parameters=dict(
        DATA_PATH="data/raw/online_retail.csv",
        COUNTRY="United Kingdom",
        OUTPUT_DIR="data/processed",
        PLOT_REVENUE=False,
        PLOT_TIME_PATTERNS=False,
        PLOT_PRODUCTS=False,
        PLOT_CUSTOMERS=False,
        PLOT_RFM=False,
    ),
    kernel_name="python3",
)

print("✅ Bước 1 hoàn thành: Làm sạch dữ liệu")

# ============================================================================
# BƯỚC 2: BASKET PREPARATION
# ============================================================================
print("\n" + "="*60)
print("BƯỚC 2: BASKET PREPARATION")
print("="*60)

pm.execute_notebook(
    "notebooks/basket_preparation.ipynb",
    "notebooks/runs/basket_preparation_run.ipynb",
    parameters=dict(
        CLEANED_DATA_PATH="data/processed/cleaned_uk_data.csv",
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        INVOICE_COL="InvoiceNo",
        ITEM_COL="Description",
        QUANTITY_COL="Quantity",
        THRESHOLD=1,
    ),
    kernel_name="python3",
)

print("✅ Bước 2 hoàn thành: Chuẩn bị basket")

# ============================================================================
# BƯỚC 3a: APRIORI MODELLING
# ============================================================================
print("\n" + "="*60)
print("BƯỚC 3a: APRIORI MODELLING")
print("="*60)

pm.execute_notebook(
    "notebooks/apriori_modelling.ipynb",
    "notebooks/runs/apriori_modelling_run.ipynb",
    parameters=dict(
        BASKET_BOOL_PATH="data/processed/basket_bool.parquet",
        RULES_OUTPUT_PATH="data/processed/rules_apriori_filtered.csv",

        # Tham số Apriori
        MIN_SUPPORT=0.05,
        MAX_LEN=3,

        # Generate rules
        METRIC="lift",
        MIN_THRESHOLD=1.0,

        # Lọc luật
        FILTER_MIN_SUPPORT=0.05,
        FILTER_MIN_CONF=0.3,
        FILTER_MIN_LIFT=1.2,
        FILTER_MAX_ANTECEDENTS=2,
        FILTER_MAX_CONSEQUENTS=1,

        # Số luật để vẽ
        TOP_N_RULES=20,

        # Tắt plot khi chạy batch
        PLOT_TOP_LIFT=False,
        PLOT_TOP_CONF=False,
        PLOT_SCATTER=False,
        PLOT_NETWORK=False,
        PLOT_PLOTLY_NETWORK=False,
        PLOT_PLOTLY_SCATTER=False,  
    ),
    kernel_name="python3",
)

print("✅ Bước 3a hoàn thành: Apriori modelling")

# ============================================================================
# BƯỚC 3b: FP-GROWTH MODELLING (FIXED VERSION)
# ============================================================================
print("\n" + "="*60)
print("BƯỚC 3b: FP-GROWTH MODELLING")
print("="*60)

# Tạo parameters cell trực tiếp trong execute_notebook
fp_params = {
    "BASKET_BOOL_PATH": "data/processed/basket_bool.parquet",
    "RULES_OUTPUT_PATH": "data/processed/rules_fpgrowth_filtered.csv",
    "MIN_SUPPORT": 0.05,
    "MAX_LEN": 3,
    "METRIC": "lift",
    "MIN_THRESHOLD": 1.0,
    "FILTER_MIN_SUPPORT": 0.01,
    "FILTER_MIN_CONF": 0.3,
    "FILTER_MIN_LIFT": 1.2,
    "FILTER_MAX_ANTECEDENTS": 2,
    "FILTER_MAX_CONSEQUENTS": 1,
    "TOP_N_RULES": 20,
    "PLOT_TOP_LIFT": False,
    "PLOT_TOP_CONF": False,
    "PLOT_SCATTER": False,
    "PLOT_NETWORK": False,
    "PLOT_PLOTLY_SCATTER": False,
}

# Trước khi chạy, đảm bảo notebook có cell parameters
import json

# Đọc notebook
with open('notebooks/fp_growth_modeling.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Kiểm tra và thêm tag parameters nếu cần
parameters_added = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'PARAMETERS' in source or 'BASKET_BOOL_PATH' in source:
            # Thêm tag parameters
            if 'metadata' not in cell:
                cell['metadata'] = {}
            if 'tags' not in cell['metadata']:
                cell['metadata']['tags'] = []
            if 'parameters' not in cell['metadata']['tags']:
                cell['metadata']['tags'].append('parameters')
                parameters_added = True
                print("✅ Đã thêm tag 'parameters' vào notebook")
            break

if parameters_added:
    # Lưu notebook đã sửa
    with open('notebooks/fp_growth_modeling_temp.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f)
    
    notebook_path = "notebooks/fp_growth_modeling_temp.ipynb"
else:
    notebook_path = "notebooks/fp_growth_modeling.ipynb"

# Chạy notebook
pm.execute_notebook(
    notebook_path,
    "notebooks/runs/fp_growth_modeling_run.ipynb",
    parameters=fp_params,
    kernel_name="python3",
)

print("✅ Bước 3b hoàn thành: FP-Growth modelling")

# ============================================================================
# BƯỚC 4: SO SÁNH APRIORI VÀ FP-GROWTH
# ============================================================================
print("\n" + "="*60)
print("BƯỚC 4: SO SÁNH APRIORI VÀ FP-GROWTH")
print("="*60)

pm.execute_notebook(
    "notebooks/compare_apriori_fpgrowth.ipynb",
    "notebooks/runs/compare_apriori_fpgrowth_run.ipynb",
    parameters=dict(
        # Không cần tham số cụ thể cho comparison
    ),
    kernel_name="python3",
)

print("✅ Bước 4 hoàn thành: So sánh thuật toán")

# ============================================================================
# KẾT THÚC PIPELINE
# ============================================================================
print("\n" + "="*60)
print("🎉 PIPELINE HOÀN THÀNH!")
print("="*60)
print("\n📁 Các file đã được tạo:")
print("  1. data/processed/cleaned_uk_data.csv")
print("  2. data/processed/basket_bool.parquet")
print("  3. data/processed/rules_apriori_filtered.csv")
print("  4. data/processed/rules_fpgrowth_filtered.csv")
print("\n📓 Các notebook đã chạy:")
print("  1. notebooks/runs/preprocessing_and_eda_run.ipynb")
print("  2. notebooks/runs/basket_preparation_run.ipynb")
print("  3. notebooks/runs/apriori_modelling_run.ipynb")
print("  4. notebooks/runs/fp_growth_modeling_run.ipynb")
print("  5. notebooks/runs/compare_apriori_fpgrowth_run.ipynb")
print("\n✅ Lab 1 & Lab 2 đã hoàn thành!")
print("   - Apriori algorithm (Lab 1)")
print("   - FP-Growth algorithm (Lab 2)")
print("   - So sánh cả hai thuật toán")
print("\n📊 Để xem kết quả, mở các file CSV trong thư mục data/processed/")
print("📊 Để xem chi tiết, mở các notebook trong thư mục notebooks/runs/")