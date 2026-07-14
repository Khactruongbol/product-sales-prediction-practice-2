# Kế hoạch cải thiện mô hình và chất lượng Practice 2

## 1. Baseline hiện tại

- Model triển khai: `linear_regression` với 32 feature.
- Test RMSE: `87.90335`.
- Test MAE: `68.89005`.
- Test R2: `0.33965`.
- Baseline lag-7 test R2: `-0.97405`.
- Quy trình chia dữ liệu: train/validation/test theo thời gian, không random split.

## 2. Chẩn đoán giới hạn dữ liệu

Kết quả kiểm tra trên 70.300 dòng modeling:

- `units_sold <= inventory_level` đúng với 100% bản ghi.
- Tương quan `inventory_level` với target khoảng `0.5899`.
- Tỷ lệ `units_sold / inventory_level` có trung bình khoảng `0.4979` và độ lệch chuẩn `0.2899`, gần với cơ chế lấy ngẫu nhiên trong khoảng tồn kho.
- Giá, discount, lượng đặt, lịch sử bán và rolling features có tương quan gần 0 với target.
- Quy tắc `inventory_level / 2` đã đạt test R2 khoảng `0.34013`, gần bằng model ML hiện tại. Đây là dấu hiệu R2 an toàn bị giới hạn bởi cách sinh dữ liệu synthetic.
- `demand_forecast` tương quan khoảng `0.9969` với target và sai số so với target chỉ khoảng 8-10 đơn vị. Dùng biến này cho test R2 khoảng `0.992`, nhưng có nguy cơ target leakage rất cao.

Kết luận: không đặt mục tiêu R2 cao giả tạo. Mọi cải thiện phải giữ `demand_forecast` ngoài model triển khai.

## 3. Cải thiện mô hình

1. Thêm `inventory_only_linear` để giảm nhiễu từ các feature không mang tín hiệu.
2. Thêm `ridge_regression` để regularize các coefficient của mô hình đầy đủ.
3. Thêm `extra_trees` như một ensemble phi tuyến bổ sung để kiểm tra khả năng cải thiện.
4. Giữ Linear Regression đầy đủ, Decision Tree, Random Forest và HistGradientBoosting để so sánh nhất quán.
5. Chọn họ model cuối bằng RMSE validation thấp nhất; test chỉ dùng để báo cáo khả năng tổng quát hóa của cấu hình đã chọn.
6. Chỉ thay model triển khai nếu test R2 không thấp hơn `0.33965` và validation RMSE tốt hơn model cũ.

## 4. Kiểm toán leakage và chất lượng dữ liệu

1. Tạo `reports/metrics/feature_signal_audit.json` gồm correlation, constraint, tỷ lệ sell-through và thống kê `demand_forecast - units_sold`.
2. Tạo `reports/metrics/leakage_benchmark.json` để so sánh model an toàn với benchmark dùng `demand_forecast`.
3. Benchmark leakage chỉ phục vụ giải thích; không lưu làm `best_model.joblib`, không đưa vào UI và không được chọn làm model cuối.
4. Tạo hình `reports/figures/safe_vs_leakage_r2.png` với nhãn cảnh báo rõ ràng.

## 5. Cải thiện đánh giá và tính hữu dụng

1. Tính khoảng dự đoán 90% bằng quantile tuyệt đối của residual validation.
2. Lưu interval vào `best_model.json`; UI hiển thị point prediction và khoảng tham khảo, chặn dưới tại 0.
3. Tạo `reports/metrics/test_period_stability.csv` theo tháng test với rows, MAE, RMSE và R2.
4. Tạo `reports/figures/monthly_model_stability.png` để kiểm tra drift theo thời gian.
5. Cập nhật feature schema để ghi rõ feature thực sự được model tốt nhất sử dụng.

## 6. Cập nhật báo cáo, notebook và UI

1. README bổ sung nguyên nhân R2 thấp, kết quả cải thiện an toàn và cảnh báo leakage.
2. Notebook báo cáo cuối bổ sung phần cải thiện model, kiểm toán `demand_forecast`, khoảng dự đoán và hai hình mới.
3. Streamlit hiển thị khoảng dự đoán 90% và ghi rõ model dùng feature nào.
4. Model comparison và các hình cũ phải được tái tạo từ lần train mới.

## 7. Kiểm thử và tiêu chí chấp nhận

- `demand_forecast` không có trong feature list của model triển khai.
- Model cuối có test R2 >= `0.33965`.
- Có đầy đủ metrics của ít nhất 7 dòng so sánh gồm baseline và các model ML.
- `feature_signal_audit.json`, `leakage_benchmark.json`, `test_period_stability.csv` đọc được.
- Có 11 hình hợp lệ sau cải tiến.
- Notebook vẫn là Markdown-only, 0 code cells, mọi liên kết ảnh tồn tại.
- `python -m compileall -q app.py src tests` pass.
- `python -m pytest -q` pass.
- Streamlit smoke test trả HTTP 200.

## 8. Git

1. Tạo branch `codex/improve-model-r2` từ `main` sạch.
2. Commit kế hoạch và toàn bộ cải tiến sau validation.
3. Push branch lên `origin`.
4. Merge branch vào `main` bằng merge commit và push `main`.
5. Xác minh `HEAD == origin/main` và working tree sạch.
