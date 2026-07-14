# Product Sales Prediction - Practice 2

Project hồi quy dự đoán số lượng sản phẩm bán được theo cửa hàng, sản phẩm và ngày. Pipeline tải dữ liệu công khai, làm sạch, tạo đặc trưng lịch sử chống rò rỉ, so sánh bốn mô hình ML với baseline mùa vụ, lưu model tốt nhất và cung cấp giao diện Streamlit.

## Kết quả đã chạy

- Dữ liệu raw: 73.100 dòng, 15 cột, từ 2022-01-01 đến 2024-01-01.
- Dữ liệu modeling: 70.300 dòng sau khi bỏ 2.800 dòng đầu chuỗi chưa đủ lịch sử 28 ngày.
- Split theo thời gian: train 49.200 dòng, validation 10.500 dòng, test 10.600 dòng.
- Model được chọn: Linear Regression.
- Test RMSE: 87,903; MAE: 68,890; MSE: 7.726,998; R2: 0,340.
- Baseline lag-7 có RMSE 151,984; model được chọn giảm RMSE khoảng 42,2% so với baseline.

| Model | MAE test | MSE test | RMSE test | R2 test |
|---|---:|---:|---:|---:|
| Linear Regression | 68,890 | 7.726,998 | 87,903 | 0,340 |
| HistGradientBoosting | 68,928 | 7.761,023 | 88,097 | 0,337 |
| Random Forest | 70,364 | 7.937,884 | 89,095 | 0,322 |
| Decision Tree | 70,830 | 8.380,871 | 91,547 | 0,284 |
| Seasonal lag-7 baseline | 118,576 | 23.099,081 | 151,984 | -0,974 |

Hyperparameter được chọn trên validation. Mỗi cấu hình thắng của từng họ model được fit lại trên train + validation và đánh giá một lần trên test.

## Nguồn dữ liệu

- Kaggle: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
- Dataset ref: `anirudhchauhan/retail-store-inventory-forecasting-dataset`
- Giấy phép: CC0 - Public Domain.
- Target: `Units Sold`.

`Demand Forecast` là dự báo có sẵn trong nguồn nên bị loại khỏi feature để tránh rò rỉ mục tiêu. Dữ liệu là dữ liệu tổng hợp ở cấp cửa hàng - sản phẩm - ngày, không có khách hàng riêng lẻ; vì vậy project không tạo giả nhân khẩu học hoặc Customer Lifetime Value.

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

Pipeline thực hiện tuần tự:

1. Tải ZIP qua Kaggle public dataset API và lưu raw CSV cùng metadata/SHA-256.
2. Làm sạch, tạo feature thời gian, lag 1/7/14/28 và rolling 7/28 ngày bằng `shift(1)`.
3. Chia train/validation/test theo ngày.
4. Tune và so sánh Linear Regression, Decision Tree, Random Forest, HistGradientBoosting cùng baseline lag-7.
5. Lưu model, schema, metrics và chín ảnh báo cáo.

## Chạy giao diện

```powershell
streamlit run app.py
```

Mở URL Streamlit hiển thị trên terminal, nhập kịch bản và các thống kê bán lịch sử. App chỉ nạp `models/best_model.joblib`, không train lại khi mở.

## Kiểm thử

```powershell
python -m compileall -q app.py src tests
python -m pytest -q
```

Lần kiểm thử bàn giao: 5 test pass; Streamlit trả HTTP 200 trên localhost.

## Artifact chính

- `data/raw/source_metadata.json`: nguồn, giấy phép, thời điểm tải và SHA-256.
- `data/processed/cleaning_report.json`: số dòng và khoảng thời gian của từng split.
- `reports/metrics/model_comparison.csv`: bảng so sánh đầy đủ.
- `reports/metrics/training_summary.json`: cấu hình và thông tin training.
- `reports/figures/`: 9 ảnh EDA, model comparison, prediction/residual và permutation importance.
- `models/best_model.joblib`: pipeline preprocessing + model tốt nhất.
- `models/feature_schema.json`: schema cho giao diện/dự đoán.

## Giới hạn

- Dữ liệu là synthetic nên metrics không đại diện trực tiếp cho một doanh nghiệp thật.
- Đây là dự đoán theo ngày trên panel store-product, không phải mô hình khách hàng cá nhân.
- App yêu cầu các lag/rolling được tính từ lịch sử bán trước ngày dự đoán.
- R2 test khoảng 0,34 cho thấy vẫn còn biến động chưa giải thích; kết quả chỉ phù hợp mục đích thực hành và hỗ trợ tham khảo.

Chi tiết phạm vi và quy tắc tái tạo nằm trong `implementation.md` và `rule.md`.
