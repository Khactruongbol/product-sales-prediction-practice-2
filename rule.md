# Quy tắc thực thi Practice 2

## 1. Quy tắc phạm vi

1. `implementation.md` là nguồn sự thật duy nhất cho lượt triển khai này.
2. Chỉ tạo, sửa và chạy các thành phần được liệt kê trong `implementation.md`.
3. Không tự bổ sung notebook, deep learning, ARIMA/Prophet, hệ gợi ý, API, Docker, cloud deployment, cơ sở dữ liệu hoặc nguồn dữ liệu khác.
4. Nếu một bước không thể thực hiện đúng kế hoạch, dừng bước đó, ghi rõ bằng chứng và không thay bằng một phạm vi mới.
5. Không sửa các project khác trong thư mục cha `Machine Learning - Trường`.

## 2. Quy tắc dữ liệu

1. Chỉ dùng nguồn Kaggle `anirudhchauhan/retail-store-inventory-forecasting-dataset` trong pipeline chính.
2. Dữ liệu tải bằng public API; tuyệt đối không ghi token, cookie, mật khẩu hoặc thông tin đăng nhập vào repo/log.
3. Giữ nguyên file raw; mọi biến đổi ghi sang `data/processed/`.
4. Không bịa thêm dòng, không ghép nguồn không có khóa chung và không tạo giả nhân khẩu học khách hàng.
5. `units_sold` là target. `demand_forecast` không được đưa vào feature model.
6. Mọi lag/rolling phải dùng `shift(1)` theo `store_id + product_id`; không được nhìn dữ liệu hiện tại/tương lai.
7. Chia train/validation/test theo ngày, không random split và không fit preprocessing trên validation/test.

## 3. Quy tắc mô hình

1. Cùng tập split và metrics cho mọi model.
2. Hyperparameter chỉ được chọn theo validation; test chỉ đánh giá cuối.
3. Mô hình tốt nhất là mô hình ML có RMSE test nhỏ nhất; baseline phải được báo cáo nhưng không đóng gói làm model chính.
4. Luôn báo cáo MAE, MSE, RMSE và R2; không chỉ báo cáo một metric thuận lợi.
5. Seed mặc định là 42. Mọi ngoại lệ phải được ghi trong `training_summary.json`.
6. Dự đoán xuất cho người dùng phải được chặn dưới tại 0 vì doanh số đơn vị không thể âm.

## 4. Quy tắc mã nguồn và artifact

1. Python 3.11+, UTF-8, tên hàm/biến tiếng Anh, tài liệu hướng dẫn có thể dùng tiếng Việt.
2. Dùng `pathlib`; không hard-code đường dẫn máy cá nhân trong mã nguồn.
3. Mỗi script phải có hàm chính có thể import để test; chỉ chạy CLI dưới `if __name__ == "__main__"`.
4. Không bỏ qua exception quan trọng; lỗi tải data/schema/model phải có thông báo hành động được.
5. CSV và JSON phải có schema ổn định; JSON ghi UTF-8 và indent dễ đọc.
6. Không commit `.venv`, `__pycache__`, `.pytest_cache`, file ZIP tạm, log runtime hoặc secrets.
7. Chỉ lưu model tốt nhất và các artifact được yêu cầu; không tạo file thừa ngoài kế hoạch.

## 5. Quy tắc ảnh và giao diện

1. Ảnh sử dụng backend không cần GUI, tối thiểu 150 DPI, không cắt tiêu đề/nhãn/chú giải.
2. Mọi ảnh phải được tạo từ dữ liệu sạch hoặc kết quả test; không dùng số minh họa thủ công.
3. Streamlit chỉ nạp model đã chọn, không train lại khi mở app.
4. Form phải validate kiểu/miền dữ liệu và giữ đúng tên/thuộc tính theo `feature_schema.json`.
5. UI phải hiển thị rõ đây là dự đoán tham khảo trên dữ liệu tổng hợp, không phải cam kết kinh doanh.

## 6. Quy tắc kiểm thử

1. Không coi task hoàn thành nếu pipeline, compileall, pytest, artifact checks hoặc UI smoke test chưa pass.
2. Không sửa test để che lỗi nghiệp vụ hoặc giảm tiêu chí đã nêu trong `implementation.md`.
3. Khi có lỗi, sửa nguyên nhân nhỏ nhất trong phạm vi kế hoạch rồi chạy lại toàn bộ validation liên quan.
4. Báo cáo cuối phải nêu số dòng raw/processed, khoảng ngày split, model thắng, metrics test, số test pass và kết quả UI smoke test.

## 7. Quy tắc Git/GitHub

1. Git root phải là thư mục `Product-Sales-Prediction`, không dùng Git root của thư mục cha.
2. Chỉ commit sau khi validation pass.
3. Branch triển khai bắt buộc là `codex/implement-practice-2`; nhánh đích là `main`.
4. Không force-push, không reset hard và không ghi đè repo đã tồn tại ngoài project này.
5. Chỉ merge khi branch trên remote đúng commit đã kiểm thử.
6. Sau merge phải xác minh `origin/main`, repo URL và working tree sạch.
