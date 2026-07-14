# Kế hoạch triển khai Practice 2 - Predicting Product Sales

## 1. Mục tiêu và phạm vi

- Bài toán: hồi quy có giám sát trên dữ liệu bảng theo thời gian, dự đoán `Units Sold` cho một cặp cửa hàng - sản phẩm trong một ngày.
- Mục tiêu nghiệp vụ: hỗ trợ chuẩn bị tồn kho và đánh giá các kịch bản giá, giảm giá, khuyến mãi và mùa vụ.
- Đơn vị dự đoán: số sản phẩm bán được/ngày, là số không âm.
- Phạm vi dữ liệu: dữ liệu tổng hợp ở cấp cửa hàng - sản phẩm - ngày. Không suy diễn nhân khẩu học hoặc Customer Lifetime Value vì nguồn được chọn không có khóa khách hàng.
- Chỉ tiêu chọn mô hình: RMSE nhỏ nhất trên tập kiểm thử theo thời gian; MAE, MSE và R2 là chỉ tiêu bổ sung. Không dùng tập test để tinh chỉnh.

## 2. Nguồn dữ liệu

### Nguồn chính

- Kaggle: `anirudhchauhan/retail-store-inventory-forecasting-dataset`.
- File: `retail_store_inventory.csv`.
- Giấy phép: CC0 - Public Domain.
- Quy mô đã xác minh: 73.100 dòng, 15 cột, 731 ngày từ 2022-01-01 đến 2024-01-01.
- Trường sử dụng: ngày, cửa hàng, sản phẩm, danh mục, vùng, tồn kho, số lượng bán, số lượng đặt, giá, giảm giá, thời tiết, ngày lễ/khuyến mãi, giá đối thủ và mùa vụ.
- `Demand Forecast` chỉ dùng làm đường cơ sở tham khảo, không đưa vào đặc trưng mô hình vì đây là dự báo có sẵn và có nguy cơ rò rỉ mục tiêu.

### Nguồn dự phòng đã đánh giá nhưng không sử dụng

- `abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing`: mô tả trên trang phù hợp nhưng file tải thực tế không có đầy đủ các trường marketing/mùa vụ như mô tả, nên không dùng.
- UCI Online Retail: dữ liệu giao dịch thực nhưng thiếu tồn kho, khuyến mãi, thời tiết và mùa vụ; chỉ là phương án mở rộng khi có yêu cầu mới.

## 3. Cấu trúc thư mục phải tạo

```text
Product-Sales-Prediction/
|-- implementation.md
|-- rule.md
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- app.py
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- reports/
|   |-- figures/
|   |-- metrics/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- collect_data.py
|   |-- prepare_data.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- run_pipeline.py
|-- tests/
|   |-- test_pipeline.py
```

## 4. Thu thập dữ liệu thô

1. `src/collect_data.py` tải ZIP bằng Kaggle public dataset API, không cần nhúng tài khoản hoặc token.
2. Giải nén đúng file CSV vào `data/raw/retail_store_inventory.csv`.
3. Lưu `data/raw/source_metadata.json` gồm URL nguồn, thời điểm tải, kích thước, SHA-256, giấy phép và tên file.
4. Kiểm tra file tải được là CSV, có 15 cột bắt buộc và có dữ liệu; lỗi mạng hoặc sai schema phải dừng pipeline với thông báo rõ ràng.

## 5. Làm sạch và feature engineering

1. Chuẩn hóa tên cột sang `snake_case`, parse `date`, ép kiểu số và xóa dòng trùng hoàn toàn.
2. Kiểm tra missing; nếu phát sinh, điền median cho số và mode cho phân loại bằng pipeline chỉ fit trên train.
3. Loại bản ghi vô lý: target âm, giá không dương, giảm giá ngoài 0-100, tồn kho hoặc số lượng đặt âm.
4. Không xóa các điểm bán cao chỉ vì là outlier; capping đặc trưng số bằng ngưỡng học từ train nếu cần, tránh xóa peak mùa vụ hợp lệ.
5. Tạo đặc trưng thời gian: năm, tháng, quý, tuần ISO, thứ, cuối tuần, đầu/cuối tháng và biểu diễn sin/cos cho tháng/thứ.
6. Tạo đặc trưng lịch sử theo từng `store_id + product_id`, đều dùng `shift(1)` để chống rò rỉ: lag 1/7/14/28 ngày, rolling mean/std 7 và 28 ngày.
7. Loại các dòng đầu chuỗi chưa đủ lịch sử 28 ngày sau khi tạo lag.
8. Chia theo thời gian: train 70% ngày đầu, validation 15% ngày tiếp theo, test 15% ngày cuối; bảo đảm `max(train_date) < min(validation_date) < min(test_date)`.
9. Lưu `data/processed/modeling_data.csv`, `train.csv`, `validation.csv`, `test.csv`, `cleaning_report.json` và `data_dictionary.json`.

## 6. Huấn luyện và so sánh mô hình

Huấn luyện cùng một tập đặc trưng và cùng cách chia dữ liệu:

1. Baseline theo mùa: dự đoán bằng `units_sold_lag_7`.
2. Linear Regression với one-hot categorical và StandardScaler cho numeric.
3. Decision Tree Regressor.
4. Random Forest Regressor.
5. HistGradientBoosting Regressor với preprocessing tương thích dữ liệu phân loại.

Quy trình:

- Chọn một lưới tham số nhỏ, cố định và có thể chạy lại; tìm tham số trên validation, không dùng test.
- Fit lại cấu hình thắng trên train + validation, đánh giá đúng một lần trên test.
- Tính MAE, MSE, RMSE, R2 và thời gian fit/predict; lưu toàn bộ vào `reports/metrics/model_comparison.csv`.
- Lưu `reports/metrics/training_summary.json`, `reports/metrics/best_model.json`, mô hình thắng tại `models/best_model.joblib` và schema đầu vào tại `models/feature_schema.json`.
- Cố định random seed 42 và số luồng hợp lý để kết quả có thể tái lập.

## 7. Ảnh và báo cáo sau huấn luyện

Xuất tối thiểu các ảnh PNG sau vào `reports/figures/`:

1. `target_distribution.png`: phân phối `Units Sold`.
2. `sales_over_time.png`: doanh số trung bình/tổng theo ngày hoặc tuần.
3. `sales_by_category.png`: doanh số theo danh mục.
4. `promotion_effect.png`: so sánh doanh số khi có/không khuyến mãi hoặc ngày lễ.
5. `model_comparison_rmse.png`: RMSE của tất cả mô hình và baseline.
6. `actual_vs_predicted.png`: thực tế so với dự đoán của mô hình tốt nhất.
7. `residual_distribution.png`: phân phối sai số.
8. `residuals_over_time.png`: sai số theo thời gian test.
9. `feature_importance.png`: permutation importance trên mẫu test cho mô hình tốt nhất.

Ảnh phải có tiêu đề, tên trục, đơn vị, chú giải khi cần, độ phân giải tối thiểu 150 DPI và không bị cắt chữ.

## 8. Giao diện Python

- Dùng Streamlit trong `app.py`.
- Nạp duy nhất `models/best_model.joblib` và `models/feature_schema.json`.
- Form nhập một kịch bản dự đoán gồm ngày, store/product, category/region, tồn kho, số lượng đặt, giá, giảm giá, thời tiết, holiday/promotion, giá đối thủ, mùa vụ và các thống kê bán gần đây cần cho lag/rolling.
- Kiểm tra miền giá trị, không chấp nhận đầu vào âm hoặc thiếu; khóa target và `Demand Forecast` khỏi input mô hình.
- Hiển thị dự đoán không âm, làm tròn theo số đơn vị, kèm thông tin model và metrics test từ `best_model.json`.
- Thực hiện smoke test bằng cách import app/predictor và gọi dự đoán với một bản ghi hợp lệ; kiểm tra lệnh `streamlit run app.py` khởi động được.

## 9. Kiểm thử và tiêu chí hoàn thành

- Kiểm tra downloader, schema raw, ràng buộc dữ liệu sạch, lag dùng dữ liệu quá khứ, chia thời gian không chồng lấn, artifact mô hình và dự đoán không âm.
- Chạy `python -m compileall -q app.py src tests`.
- Chạy `python -m pytest -q`.
- Chạy toàn bộ `python -m src.run_pipeline` từ dữ liệu thô đến model/ảnh/metrics.
- Kiểm tra CSV/JSON đọc được, model load được, mọi ảnh tồn tại và có kích thước hợp lệ.
- Chạy smoke test Streamlit trên localhost và xác nhận HTTP 200 trước khi Git.
- README ghi cách cài đặt, chạy pipeline, chạy app, nguồn/giấy phép dữ liệu, kết quả model và giới hạn của dữ liệu tổng hợp.

## 10. GitHub

1. Khởi tạo Git riêng trong thư mục project; không commit dữ liệu tạm, cache, môi trường ảo hoặc secrets.
2. Tạo GitHub repository công khai tên `product-sales-prediction-practice-2` trong tài khoản GitHub hiện đang đăng nhập.
3. Tạo branch `codex/implement-practice-2`, commit toàn bộ artifact cần tái lập và push branch.
4. Tạo/push `main`, sau đó merge branch bằng merge commit (hoặc Pull Request nếu giao diện/API hỗ trợ), bảo đảm `main` chứa commit triển khai.
5. Xác minh remote, branch, commit cuối, trạng thái sạch và URL repo sau merge.

## 11. Thứ tự thực hiện khóa phạm vi

Thực hiện đúng thứ tự: tạo cấu trúc -> viết downloader -> tải raw -> làm sạch/feature -> train/compare -> xuất metrics/ảnh -> chọn/lưu model -> viết UI -> viết test/README -> chạy validation -> GitHub branch/push/merge. Không thêm notebook, deep learning, API service, Docker, database, recommendation system hoặc ARIMA/Prophet trong lượt này.
