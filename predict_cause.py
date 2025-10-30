import pandas as pd
import glob
import random
import joblib
import os
import xgboost as xgb
import numpy as np

print("🚀 Bắt đầu kiểm thử mô hình DỰ ĐOÁN DELAY & NGUYÊN NHÂN...")

# ====== 1️⃣ Nạp mô hình ======
print("🧠 Đang nạp mô hình...")

delay_model = xgb.Booster()
delay_model.load_model("flight_delay_model.json")

cause_model = xgb.Booster()
cause_model.load_model("flight_cause_model.json")

label_encoder = joblib.load("label_encoder.pkl")

# ====== 2️⃣ Lấy file CSV ngẫu nhiên (2002–2008) ======
files = sorted(glob.glob("data/*.csv"))
selected_files = [f for f in files if any(str(y) in f for y in range(2005, 2009))]

if not selected_files:
    raise FileNotFoundError("❌ Không tìm thấy file CSV nào trong thư mục data/.")

df = None
for attempt in range(5):
    random_file = random.choice(selected_files)
    print(f"📂 Thử đọc file: {random_file}")
    try:
        df = pd.read_csv(random_file, nrows=100_000, low_memory=False, encoding='utf-8')
        print(f"✅ Đọc thành công {len(df)} dòng từ {os.path.basename(random_file)}")
        break
    except UnicodeDecodeError:
        print("⚠️ File không phải UTF-8, thử lại với latin1 ...")
        try:
            df = pd.read_csv(random_file, nrows=100_000, low_memory=False, encoding='latin1')
            print(f"✅ Đọc thành công (latin1): {len(df)} dòng.")
            break
        except Exception as e:
            print(f"❌ Lỗi khi đọc file (latin1): {e}")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {random_file}: {e}")

if df is None or df.empty:
    raise RuntimeError("❌ Không thể đọc được file CSV hợp lệ nào để kiểm thử.")

# ====== 3️⃣ Lấy mẫu ngẫu nhiên trong file ======
needed_cols = [
    "Month","DayofMonth","DayOfWeek","DepDelay","TaxiOut","Distance",
    "Cancelled","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"
]

df = df[[c for c in needed_cols if c in df.columns]].dropna()

if df.empty:
    raise ValueError("❌ File không có đủ cột cần thiết hoặc toàn NA.")

sample = df.sample(1, random_state=random.randint(1, 9999)).iloc[0]
print("\n🎯 Mẫu dữ liệu thực tế:")
print(sample)

# ====== 4️⃣ Chuẩn bị input cho mô hình ======
X_input = pd.DataFrame([{
    "Month": sample["Month"],
    "DayofMonth": sample["DayofMonth"],
    "DayOfWeek": sample["DayOfWeek"],
    "TaxiOut": sample["TaxiOut"],
    "Distance": sample["Distance"]
}])

dtest = xgb.DMatrix(X_input)

# ====== 5️⃣ Dự đoán Delay ======
delay_pred = delay_model.predict(dtest)
delay_pred_label = int(round(delay_pred[0]))

real_delay = int(sample["DepDelay"] > 15)

print("\n📊 Kết quả DỰ ĐOÁN DELAY:")
print(f"  ➤ Thực tế: {'Delay' if real_delay else 'Đúng giờ'}")
print(f"  ➤ Dự đoán: {'Delay' if delay_pred_label else 'Đúng giờ'}")

# ====== 6️⃣ Nếu delay → dự đoán nguyên nhân ======
if delay_pred_label == 1:
    if all(c in sample for c in ["CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]):
        real_cause = sample[["CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]].idxmax()
        y_pred_cause = cause_model.predict(dtest)

        # Nếu multi-class → lấy chỉ số xác suất lớn nhất
        if y_pred_cause.ndim > 1:
            cause_pred_idx = int(np.argmax(y_pred_cause, axis=1)[0])
        else:
            cause_pred_idx = int(round(y_pred_cause[0]))

        cause_pred = label_encoder.inverse_transform([cause_pred_idx])[0]

        print("\n🧠 Kết quả DỰ ĐOÁN NGUYÊN NHÂN:")
        print(f"  ➤ Nguyên nhân thực tế: {real_cause}")
        print(f"  ➤ Nguyên nhân dự đoán: {cause_pred}")
    else:
        print("\n⚠️ File không có đủ cột nguyên nhân delay, bỏ qua phần dự đoán nguyên nhân.")
else:
    print("\n🟢 Chuyến bay này đúng giờ, không cần dự đoán nguyên nhân.")
