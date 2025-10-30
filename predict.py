import pandas as pd
from xgboost import Booster, DMatrix

print("🚀 Bắt đầu kiểm thử mô hình DỰ ĐOÁN DELAY...")

# 1️⃣ Nạp mô hình đã huấn luyện
model_path = "flight_delay_model.json"
booster = Booster()
booster.load_model(model_path)
print(f"✅ Đã tải mô hình từ: {model_path}")

# 2️⃣ Chọn file dữ liệu thật để kiểm thử
test_file = "data/2008.csv"   # bạn có thể đổi sang file khác trong thư mục data/
print(f"📂 Đọc dữ liệu từ {test_file} ...")

# Đọc một phần dữ liệu để kiểm thử nhanh
try:
    df = pd.read_csv(test_file, nrows=50_000, low_memory=False, encoding="utf-8")
except UnicodeDecodeError:
    print("⚠️ Lỗi mã hóa, thử lại với latin1 ...")
    df = pd.read_csv(test_file, nrows=50_000, low_memory=False, encoding="latin1")

print("✅ Dữ liệu đọc thành công:", df.shape)

# 3️⃣ Lọc cột cần thiết
cols = ["Month", "DayofMonth", "DayOfWeek", "TaxiOut", "Distance", "DepDelay"]
df = df[cols].dropna()
print("📊 Sau khi lọc:", df.shape)

# 4️⃣ Tạo nhãn thật để so sánh (nếu có)
df["Delayed"] = (df["DepDelay"] > 15).astype(int)

# 5️⃣ Tạo DMatrix để dự đoán
features = ["Month", "DayofMonth", "DayOfWeek", "TaxiOut", "Distance"]
dmatrix = DMatrix(df[features])

# 6️⃣ Dự đoán
print("🤖 Đang dự đoán...")
y_pred_prob = booster.predict(dmatrix)
y_pred = (y_pred_prob > 0.5).astype(int)

# 7️⃣ Gộp kết quả và hiển thị
df["Predicted"] = y_pred

# 8️⃣ Tính độ chính xác
accuracy = (df["Predicted"] == df["Delayed"]).mean()
print(f"🎯 Độ chính xác trên dữ liệu mẫu: {accuracy:.2%}")

# 9️⃣ Hiển thị một vài kết quả dự đoán đầu tiên
print("\n📋 Kết quả mẫu:")
print(df[["Month", "DayofMonth", "DayOfWeek", "TaxiOut", "Distance", "DepDelay", "Delayed", "Predicted"]].head(10))

# 🔟 Lưu kết quả ra file CSV (tùy chọn)
output_file = "prediction_results.csv"
df.to_csv(output_file, index=False)
print(f"\n💾 Đã lưu kết quả dự đoán vào: {output_file}")
