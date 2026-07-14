"""Streamlit interface for the selected product-sales model."""

from __future__ import annotations

import json
from datetime import date

import streamlit as st

from src.config import BEST_MODEL_INFO_PATH
from src.predict import load_artifacts, predict_units


def main() -> None:
    st.set_page_config(page_title="Product Sales Prediction", page_icon="📦", layout="wide")
    st.title("Product Sales Prediction")
    st.caption("Practice 2 - dự đoán số lượng bán theo cửa hàng, sản phẩm và ngày")
    try:
        model, schema = load_artifacts()
        model_info = json.loads(BEST_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    metrics = model_info["test_metrics"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Best model", model_info["model"])
    metric_columns[1].metric("Test RMSE", f"{metrics['rmse']:.2f}")
    metric_columns[2].metric("Test MAE", f"{metrics['mae']:.2f}")
    metric_columns[3].metric("Test R²", f"{metrics['r2']:.3f}")

    selected_features = model_info.get("selected_features", schema.get("selected_model_features", []))
    if selected_features:
        st.info(
            "Feature được model triển khai sử dụng: "
            + ", ".join(f"`{feature}`" for feature in selected_features)
            + ". Các trường còn lại được giữ để so sánh model và tương thích schema."
        )

    categorical = schema["categorical"]
    numeric = schema["numeric"]
    with st.form("prediction_form"):
        st.subheader("Kịch bản cần dự đoán")
        left, middle, right = st.columns(3)
        with left:
            prediction_date = st.date_input("Ngày dự đoán", value=date(2024, 1, 2))
            store_id = st.selectbox("Cửa hàng", categorical["store_id"])
            product_id = st.selectbox("Sản phẩm", categorical["product_id"])
            category = st.selectbox("Danh mục", categorical["category"])
            region = st.selectbox("Khu vực", categorical["region"])
        with middle:
            inventory_level = _number("Tồn kho đầu ngày", numeric["inventory_level"], step=1.0)
            units_ordered = _number("Số lượng đã đặt", numeric["units_ordered"], step=1.0)
            price = _number("Giá bán", numeric["price"], step=0.01)
            discount = _number("Giảm giá (%)", numeric["discount"], step=1.0)
            competitor_pricing = _number("Giá đối thủ", numeric["competitor_pricing"], step=0.01)
        with right:
            weather_condition = st.selectbox("Thời tiết", categorical["weather_condition"])
            holiday_promotion = st.selectbox(
                "Ngày lễ/khuyến mãi", categorical["holiday_promotion"], format_func=lambda x: "Có" if x == "1" else "Không"
            )
            seasonality = st.selectbox("Mùa", categorical["seasonality"])
            st.info("Các đặc trưng lịch sử bên dưới phải được tính từ dữ liệu bán trước ngày dự đoán.")

        st.subheader("Lịch sử bán gần đây")
        history_columns = st.columns(4)
        history_fields = [
            ("units_sold_lag_1", "Bán 1 ngày trước"),
            ("units_sold_lag_7", "Bán 7 ngày trước"),
            ("units_sold_lag_14", "Bán 14 ngày trước"),
            ("units_sold_lag_28", "Bán 28 ngày trước"),
            ("units_sold_rolling_mean_7", "Trung bình 7 ngày"),
            ("units_sold_rolling_std_7", "Độ lệch chuẩn 7 ngày"),
            ("units_sold_rolling_mean_28", "Trung bình 28 ngày"),
            ("units_sold_rolling_std_28", "Độ lệch chuẩn 28 ngày"),
        ]
        history_values: dict[str, float] = {}
        for index, (field, label) in enumerate(history_fields):
            with history_columns[index % 4]:
                history_values[field] = _number(label, numeric[field], step=1.0)

        submitted = st.form_submit_button("Dự đoán doanh số", type="primary", use_container_width=True)

    if submitted:
        values = {
            "date": prediction_date,
            "store_id": store_id,
            "product_id": product_id,
            "category": category,
            "region": region,
            "inventory_level": inventory_level,
            "units_ordered": units_ordered,
            "price": price,
            "discount": discount,
            "weather_condition": weather_condition,
            "holiday_promotion": holiday_promotion,
            "competitor_pricing": competitor_pricing,
            "seasonality": seasonality,
            **history_values,
        }
        prediction = predict_units(model, values)
        st.success(f"Doanh số dự đoán: **{round(prediction):,} đơn vị**")
        st.caption(f"Giá trị liên tục trước khi làm tròn: {prediction:.2f}")
        interval_error = float(model_info.get("prediction_interval_90_abs_error", 0.0))
        if interval_error > 0:
            lower = max(0.0, prediction - interval_error)
            upper = prediction + interval_error
            st.info(
                f"Khoảng dự đoán tham khảo 90%: **{round(lower):,} - {round(upper):,} đơn vị** "
                f"(hiệu chỉnh từ residual validation ±{interval_error:.1f})."
            )

    st.warning(
        "Kết quả chỉ mang tính tham khảo học thuật. Mô hình được huấn luyện trên dữ liệu tổng hợp và không phải cam kết kinh doanh."
    )


def _number(label: str, stats: dict[str, float], step: float) -> float:
    lower = max(0.0, float(stats["min"]))
    upper = max(lower, float(stats["max"]))
    default = min(max(float(stats["median"]), lower), upper)
    return float(st.number_input(label, min_value=lower, max_value=upper, value=default, step=step))


if __name__ == "__main__":
    main()

