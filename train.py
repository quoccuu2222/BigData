import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
import glob
import time

start = time.time()
print("🚀 Bắt đầu huấn luyện mô hình DỰ ĐOÁN DELAY...")

# 1️⃣ Đọc dữ liệu
files = sorted(glob.glob("data/*.csv"))
selected_files = [f for f in files if any(str(y) in f for y in range(2002, 2009))]
print(f"📂 Nạp {len(selected_files)} file:", selected_files)

dfs = []
for f in selected_files:
    print(f"📥 Đọc {f} ...")
    try:
        df_temp = pd.read_csv(f, nrows=1_000_000, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️ Lỗi mã hóa, thử latin1 ...")
        df_temp = pd.read_csv(f, nrows=1_000_000, low_memory=False, encoding='latin1')
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
print("✅ Dữ liệu tổng hợp:", df.shape)

# 2️⃣ Lọc cột
cols = ["Month", "DayofMonth", "DayOfWeek", "DepDelay", "TaxiOut", "Distance", "Cancelled"]
df = df[cols].dropna()

# 3️⃣ Nhãn delay
df["Delayed"] = (df["DepDelay"] > 15).astype(int)
print("📊 Tỷ lệ delay:", df["Delayed"].mean())

# 4️⃣ Cân bằng dữ liệu
df_major = df[df.Delayed == 0]
df_minor = df[df.Delayed == 1]
df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major)//2, random_state=42)
df_balanced = pd.concat([df_major, df_minor_up])
print("⚖️ Sau cân bằng:", df_balanced["Delayed"].value_counts())

# 5️⃣ Train/test split
X = df_balanced[["Month", "DayofMonth", "DayOfWeek", "TaxiOut", "Distance"]]
y = df_balanced["Delayed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Tạo model — tự động chọn GPU nếu có, fallback CPU nếu không
try:
    model = XGBClassifier(
        tree_method="gpu_hist",  # GPU
        device="cuda",
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2,
        eval_metric="logloss"
    )
    print("⚡ GPU khả dụng — đang huấn luyện bằng GPU.")
    model.fit(X_train, y_train)
except Exception as e:
    print(f"💻 GPU không khả dụng ({e}). Chuyển sang CPU ...")
    model = XGBClassifier(
        tree_method="hist",  # CPU
        device="cpu",
        n_estimators=250,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

# 7️⃣ Đánh giá
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8️⃣ Lưu model
model.save_model("flight_delay_model.json")
end = time.time()
print(f"💾 Huấn luyện xong! ⏱️ Thời gian: {(end - start)/60:.2f} phút")
