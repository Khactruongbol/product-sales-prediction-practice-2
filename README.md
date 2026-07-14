# Product Sales Prediction - Practice 2

Project hồi quy dự đoán số lượng sản phẩm bán được theo cửa hàng, sản phẩm và ngày. Pipeline tải dữ liệu công khai, làm sạch, tạo đặc trưng lịch sử chống rò rỉ, so sánh nhiều mô hình, kiểm toán leakage, lưu model tốt nhất và cung cấp giao diện Streamlit kèm khoảng dự đoán.

## Kết quả sau vòng cải tiến

- Dữ liệu raw: 73.100 dòng, 15 cột, từ 2022-01-01 đến 2024-01-01.
- Dữ liệu modeling: 70.300 dòng sau khi bỏ 2.800 dòng đầu chuỗi chưa đủ lịch sử 28 ngày.
- Split theo thời gian: train 49.200 dòng, validation 10.500 dòng, test 10.600 dòng.
- Model được chọn bằng validation RMSE: `inventory_only_linear`.
- Test RMSE: 87,859; MAE: 68,833; MSE: 7.719,167; R²: 0,34032.
- R² cũ: 0,33965. Mức tăng an toàn: +0,00067.
- Khoảng dự đoán tham khảo 90%: point prediction ±152,1 đơn vị, chặn dưới tại 0.
- Baseline lag-7 có RMSE 151,984; model được chọn giảm RMSE khoảng 42,2%.

| Model | MAE test | MSE test | RMSE test | R² test |
|---|---:|---:|---:|---:|
| Inventory-only Linear | 68,833 | 7.719,167 | 87,859 | 0,34032 |
| Ridge Regression | 68,887 | 7.726,326 | 87,900 | 0,33971 |
| Linear Regression | 68,890 | 7.726,998 | 87,903 | 0,33965 |
| HistGradientBoosting | 68,928 | 7.761,023 | 88,097 | 0,33674 |
| Extra Trees | 69,073 | 7.794,562 | 88,287 | 0,33388 |
| Random Forest | 70,364 | 7.937,884 | 89,095 | 0,32163 |
| Decision Tree | 70,830 | 8.380,871 | 91,547 | 0,28377 |
| Seasonal lag-7 baseline | 118,576 | 23.099,081 | 151,984 | -0,97405 |

Hyperparameter và họ model cuối được chọn bằng validation RMSE. Test chỉ được dùng để báo cáo cuối.

## Vì sao R² chỉ khoảng 0,34?

Kiểm toán dữ liệu cho thấy:

- `units_sold <= inventory_level` đúng với 100% bản ghi.
- `inventory_level` là feature triển khai duy nhất có tương quan đáng kể với target: khoảng 0,5899.
- Tỷ lệ `units_sold / inventory_level` có trung bình 0,4979 và độ lệch chuẩn 0,2899, gần với cơ chế lấy ngẫu nhiên trong khoảng tồn kho.
- Giá, discount, lượng đặt và các lag/rolling gần như không có tương quan với target.

Do đó quy tắc đơn giản `inventory_level / 2` đã gần chạm trần tín hiệu của bộ dữ liệu synthetic. Thêm model phức tạp không tạo thêm thông tin mà dữ liệu không chứa.

`Demand Forecast` có tương quan 0,9969 với target và benchmark cho R² khoảng 0,9918. Đây là near-direct target proxy có nguy cơ leakage, nên chỉ được lưu trong `leakage_benchmark.json` để giải thích và không được dùng trong model triển khai hoặc UI.

## Nguồn dữ liệu

- Kaggle: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
- Dataset ref: `anirudhchauhan/retail-store-inventory-forecasting-dataset`
- Giấy phép: CC0 - Public Domain.
- Target: `Units Sold`.

Dữ liệu là dữ liệu tổng hợp ở cấp cửa hàng - sản phẩm - ngày, không có khách hàng riêng lẻ; project không tạo giả nhân khẩu học hoặc Customer Lifetime Value.

## Cài đặt

Yêu cầu Python 3.11 trở lên.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy toàn bộ pipeline

```powershell
python -m src.run_pipeline
```

Pipeline thực hiện:

1. Tải ZIP bằng Kaggle public API và lưu raw CSV cùng metadata/SHA-256.
2. Làm sạch, tạo feature thời gian, lag 1/7/14/28 và rolling 7/28 ngày bằng `shift(1)`.
3. Chia train/validation/test theo ngày.
4. Tune và so sánh bảy model ML cùng baseline lag-7.
5. Chọn model bằng validation, đánh giá test và tính prediction interval.
6. Xuất leakage audit, feature-signal audit, stability report và 11 ảnh.

## Chạy giao diện

```powershell
streamlit run app.py
```

App chỉ nạp `models/best_model.joblib`, không train lại khi mở. Kết quả gồm point prediction và khoảng tham khảo 90%.

## Kiểm thử

```powershell
python -m compileall -q app.py src tests
python -m pytest -q
```

Validation bao gồm schema dữ liệu, chống leakage của lag, temporal split, model/artifact, R² không regression, leakage benchmark không deploy, notebook Markdown-only và toàn bộ đường dẫn ảnh.

## Artifact chính

- `MODEL_IMPROVEMENT_PLAN.md`: kế hoạch và tiêu chí cải tiến.
- `notebooks/99_product_sales_workflow.ipynb`: notebook báo cáo cuối Markdown-only với 11 ảnh.
- `data/raw/source_metadata.json`: nguồn, giấy phép, thời điểm tải và SHA-256.
- `data/processed/cleaning_report.json`: kết quả làm sạch và temporal split.
- `reports/metrics/model_comparison.csv`: bảng so sánh tám dòng model/baseline.
- `reports/metrics/feature_signal_audit.json`: correlation và chẩn đoán giới hạn dữ liệu.
- `reports/metrics/leakage_benchmark.json`: benchmark dùng `demand_forecast`, không deploy.
- `reports/metrics/test_period_stability.csv`: MAE/RMSE/R² theo tháng test.
- `reports/figures/`: 11 ảnh EDA, model, leakage và stability.
- `models/best_model.joblib`: pipeline model triển khai.
- `models/feature_schema.json`: schema và feature thực sự được model sử dụng.

## Giới hạn

- Dataset synthetic nên metrics không đại diện trực tiếp cho một doanh nghiệp thật.
- Model triển khai hiện dùng `inventory_level`; các biến còn lại không mang tín hiệu đáng kể trong dữ liệu nguồn.
- R² khoảng 0,34 phản ánh giới hạn tín hiệu của dữ liệu, không phải lý do để đưa target proxy vào model.
- Khoảng dự đoán khá rộng, cho thấy độ bất định cao ở cấp store-product-day.

Chi tiết phạm vi gốc nằm trong `implementation.md`, `rule.md`; vòng cải tiến nằm trong `MODEL_IMPROVEMENT_PLAN.md`.
