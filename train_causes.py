import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import glob
import time

start = time.time()
print("🚀 Huấn luyện mô hình DỰ ĐOÁN NGUYÊN NHÂN...")

# 1️⃣ Đọc dữ liệu (2002–2008)
files = sorted(glob.glob("data/*.csv"))
selected_files = [f for f in files if any(str(y) in f for y in range(2002, 2009))]

dfs = []
for f in selected_files:
    print(f"📥 Đọc {f} ...")
    try:
        df_temp = pd.read_csv(f, nrows=1_000_000, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df_temp = pd.read_csv(f, nrows=1_000_000, low_memory=False, encoding='latin1')
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
print("✅ Dữ liệu tổng hợp:", df.shape)

# 2️⃣ Giữ các chuyến bay bị delay
cols = [
    "Month","DayofMonth","DayOfWeek","TaxiOut","Distance",
    "CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"
]
df = df[cols].dropna()

# 3️⃣ Tạo nhãn nguyên nhân chính (delay lớn nhất)
df["MainCause"] = df[
    ["CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]
].idxmax(axis=1)

# 4️⃣ Mã hóa nhãn
label_encoder = LabelEncoder()
df["MainCauseCode"] = label_encoder.fit_transform(df["MainCause"])
joblib.dump(label_encoder, "label_encoder.pkl")

# 5️⃣ Chuẩn bị dữ liệu train/test
X = df[["Month","DayofMonth","DayOfWeek","TaxiOut","Distance"]]
y = df["MainCauseCode"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Huấn luyện mô hình — tự động chọn GPU nếu có
try:
    model = XGBClassifier(
        tree_method="gpu_hist",  # GPU tăng tốc
        device="cuda",
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
    print("⚡ GPU khả dụng — đang huấn luyện bằng GPU.")
    model.fit(X_train, y_train)
except Exception as e:
    print(f"💻 GPU không khả dụng ({e}). Chuyển sang CPU ...")
    model = XGBClassifier(
        tree_method="hist",  # CPU fallback
        device="cpu",
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

# 7️⃣ Đánh giá
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8️⃣ Lưu mô hình
model.save_model("flight_cause_model.json")

end = time.time()
print(f"💾 Huấn luyện xong! ⏱️ {(end - start)/60:.2f} phút")
