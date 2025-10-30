from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from hdfs import InsecureClient, HdfsError
import os
from io import BytesIO
import traceback
import logging
import base64 
from collections import Counter
import zipfile 
import tempfile 
import findspark

# THIẾT LẬP MÔI TRƯỜNG BẮT BUỘC TRÊN WINDOWS
# 💡 Thiết lập đường dẫn SPARK_HOME (Cần thiết cho PySpark)
os.environ['SPARK_HOME'] = "G:\\Spark\\spark-3.5.7-bin-hadoop3"
print(f"SPARK_HOME được thiết lập: {os.environ['SPARK_HOME']}")

# 💡 Đã cập nhật HADOOP_HOME theo đường dẫn của người dùng
os.environ['HADOOP_HOME'] = "G:\\hadoop\\hadoop-3.3.6"
if not os.path.exists(os.path.join(os.environ['HADOOP_HOME'], 'bin', 'winutils.exe')):
    print("!!! LƯU Ý QUAN TRỌNG: Không tìm thấy winutils.exe tại đường dẫn HADOOP_HOME đã thiết lập. Vui lòng kiểm tra lại để PySpark hoạt động ổn định.")
else:
    print(f"HADOOP_HOME được thiết lập thành công: {os.environ['HADOOP_HOME']}")

# 💡 THÊM: Thiết lập thư mục cấu hình Hadoop
os.environ['HADOOP_CONF_DIR'] = os.path.join(os.environ['HADOOP_HOME'], 'etc', 'hadoop')

# 💡 THÊM: Thiết lập thư mục Temp cục bộ 
TEMP_DIR = os.path.join(os.getcwd(), 'spark_temp_dir')
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR

# THƯ VIỆN PYSPARK
try:
    findspark.init() 
except:
    print("Cảnh báo: Không thể khởi tạo findspark. Đảm bảo SPARK_HOME đã được đặt.")
    
os.environ['PYSPARK_PYTHON'] = os.path.join(os.path.dirname(os.sys.executable), 'python.exe')

from pyspark.sql import SparkSession
# 💡 ĐÃ CẬP NHẬT: Thêm DoubleType để ép kiểu an toàn
from pyspark.sql.functions import explode, split, lower, col, avg, round as spark_round, count as spark_count, lit, when, regexp_replace, sum
from pyspark.sql.types import DoubleType, StringType

# THƯ VIỆN TRỰC QUAN HÓA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import builtins

try:
    import joblib
    import xgboost as xgb
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("CẢNH BÁO: Không thể import joblib, xgboost, hoặc pandas. Chức năng dự đoán sẽ không hoạt động.")
    joblib, xgb, pd, LabelEncoder = None, None, None, None

# Cấu hình log PySpark để tránh spam console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# --- CẤU HÌNH HDFS ---
HDFS_WEB_URL = 'http://localhost:9870'
HDFS_USER = 'hadoop' # Thay bằng username của bạn
client = InsecureClient(HDFS_WEB_URL, user=HDFS_USER)
HDFS_RPC_URL = 'hdfs://localhost:9000'
global_spark_session = None

# --- KHỞI TẠO VÀ TẢI MÔ HÌNH DỰ ĐOÁN ---
# Khởi tạo LABEL_ENCODER mặc định nếu không tải được
LABEL_ENCODER = LabelEncoder() if LabelEncoder else None 
if LABEL_ENCODER:
    # Đây là các nhãn được huấn luyện từ mô hình cũ
    LABEL_ENCODER.classes_ = ['CarrierDelay', 'LateAircraftDelay', 'NASDelay', 'SecurityDelay', 'WeatherDelay'] 
    
DELAY_MODEL = None
CAUSE_MODEL = None
MODEL_LOAD_SUCCESS = False

if joblib and xgb and tempfile and client:
    try:
        print("--- Đang thử tải Mô hình Dự đoán từ HDFS (Thư mục gốc) ---")
        
        # SỬA LỖI: Chỉ tạo tên file tạm thời, không mở file trước
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp_dir:
            
            # 1. Tải Label Encoder (.pkl)
            tmp_encoder_path = os.path.join(temp_dir, 'label_encoder.pkl')
            client.download('/label_encoder.pkl', tmp_encoder_path)
            LABEL_ENCODER = joblib.load(tmp_encoder_path)
            
            # 2. Tải Mô hình Binary Classification (Delay/No Delay)
            tmp_delay_model_path = os.path.join(temp_dir, 'flight_delay_model.json')
            client.download('/flight_delay_model.json', tmp_delay_model_path)
            DELAY_MODEL = xgb.Booster()
            DELAY_MODEL.load_model(tmp_delay_model_path)

            # 3. Tải Mô hình Multi-class Classification (Delay Cause)
            tmp_cause_model_path = os.path.join(temp_dir, 'flight_cause_model.json')
            client.download('/flight_cause_model.json', tmp_cause_model_path)
            CAUSE_MODEL = xgb.Booster()
            CAUSE_MODEL.load_model(tmp_cause_model_path)
        
        MODEL_LOAD_SUCCESS = True
        print("--- Tải Mô hình Dự đoán thành công ---")

    except HdfsError as e:
        print(f"--- LỖI HDFS: {e} ---")
        print("Vui lòng đảm bảo các file '.pkl' và '.json' đã được upload vào thư mục gốc / trên HDFS.")
        MODEL_LOAD_SUCCESS = False

    except Exception as e:
        print(f"--- LỖI TẢI MÔ HÌNH: {type(e).__name__}: {e} ---")
        traceback.print_exc()
        MODEL_LOAD_SUCCESS = False

def get_spark_session(hdfs_rpc_url):
    """Tạo hoặc trả về Spark Session đã tồn tại."""
    global global_spark_session
    if global_spark_session is None:
        try:
            global_spark_session = SparkSession.builder \
                .appName("HDFSWebAppAnalysis") \
                .master("local[*]") \
                .config("spark.hadoop.fs.defaultFS", hdfs_rpc_url) \
                .config("spark.driver.extraJavaOptions", "--add-opens=java.base/java.lang=ALL-UNNAMED") \
                .config("spark.executor.extraJavaOptions", "--add-opens=java.base/java.lang=ALL-UNNAMED") \
                .config("spark.driver.maxResultSize", "4g") \
                .config("spark.local.dir", os.environ['TMPDIR']) \
                .config("spark.hadoop.hadoop.tmp.dir", os.environ['TMPDIR']) \
                .getOrCreate()
            print("--- Spark Session đã khởi tạo thành công ---")
        except Exception as e:
            print(f"--- LỖI KHỞI TẠO SPARK: {type(e).__name__}: {e} ---")
            traceback.print_exc()
            return None
    return global_spark_session


# --- CHỨC NĂNG TẠO BIỂU ĐỒ ---

def generate_plot(top_data, file_path, title, x_label, y_label, data_key, label_key, horizontal=True):
    """Tạo biểu đồ chung cho các phân tích độ trễ/tỷ lệ."""
    labels = [row[label_key] for row in top_data]
    values = [row[data_key] for row in top_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color='#ff7f0e') if horizontal else ax.bar(labels, values, color='#ff7f0e') 
    
    for bar in bars:
        if horizontal:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom')
            
    ax.set_xlabel(x_label)
    ax.set_title(f'{title} trong {file_path}', fontsize=14)
    if horizontal: ax.invert_yaxis()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"

def generate_word_count_plot(top_data, file_path):
    """Tạo biểu đồ Word Count và trả về Base64 String."""
    words = [row['word'] for row in top_data]
    counts = [row['count'] for row in top_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words, counts, color='#007bff')
    
    ax.set_xlabel('Tần suất (Count)')
    ax.set_title(f'Top 10 Từ Phổ Biến trong {file_path}', fontsize=14)
    ax.invert_yaxis()
    
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"

# HÀM TẠO BIỂU ĐỒ FEATURE IMPORTANCE
def generate_feature_importance_plot(model, title):
    """Trích xuất và trực quan hóa Feature Importance từ mô hình XGBoost Booster."""
    try:
        # Lấy importance (weight) từ Booster
        importance = model.get_score(importance_type='weight') if hasattr(model, 'get_score') else {}

        # Chuyển sang danh sách dictionary
        plot_data_list = []
        for feature, score in importance.items():
            plot_data_list.append({'FeatureName': feature, 'Score': float(score)})

        # Nếu rỗng, thử dùng attribute feature_names nếu có
        if not plot_data_list and hasattr(model, 'feature_names') and model.feature_names:
            plot_data_list = [{'FeatureName': name, 'Score': 0.0} for name in model.feature_names]

        # Sắp xếp giảm dần
        plot_data_list.sort(key=lambda x: x['Score'], reverse=True)

        # Mapping tên đặc trưng sang tiếng Việt (tuỳ biến)
        feature_name_map = {
            'Month': 'Tháng',
            'DayofMonth': 'Ngày trong tháng',
            'DayOfWeek': 'Thứ trong tuần',
            'TaxiOut': 'Thời gian lăn (TaxiOut)',
            'Distance': 'Khoảng cách bay'
        }

        labels = [row['FeatureName'] for row in plot_data_list]
        values = [row['Score'] for row in plot_data_list]
        display_labels = [feature_name_map.get(l, l) for l in labels]

        # Cấu hình biểu đồ (kích thước động theo số feature)
        plt.rcParams.update({'font.size': 11})
        height = max(4, 0.6 * len(values))
        fig, ax = plt.subplots(figsize=(12, height))

        # Màu gradient nếu có nhiều bar
        if values:
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(values)))
        else:
            colors = ['#1f77b4']

        bars = ax.barh(display_labels, values, color=colors)

        # Thêm nhãn giá trị ở cuối mỗi thanh
        max_score = max(values) if values else 1
        for bar in bars:
            width = bar.get_width()
            pct = (width / max_score) * 100 if max_score > 0 else 0
            ax.text(width + max_score * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f} ({pct:.1f}%)",
                    va='center', ha='left', fontsize=10, fontweight='bold')

        ax.set_xlabel('Điểm tầm quan trọng (weight / F-score)', fontsize=12, fontweight='bold')
        ax.set_title(f"{title}\nĐộ quan trọng của các đặc trưng", fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        data = base64.b64encode(buf.getbuffer()).decode('ascii')
        return f"data:image/png;base64,{data}", plot_data_list

    except Exception as e:
        print(f"LỖI TẠO BIỂU ĐỒ FEATURE IMPORTANCE: {e}")
        traceback.print_exc()
        return None, None


# --- ROUTES CHO QUẢN LÝ LƯU TRỮ (Giữ nguyên) ---

@app.route('/')
@app.route('/browse')
@app.route('/browse/<path:path>')
def browse(path='/'):
    """Hiển thị danh sách file/thư mục trong HDFS"""
    try:
        full_path = '/' + path.lstrip('/')
        files = client.list(full_path, status=True)

        entries = []
        if full_path != '/':
            parent = os.path.dirname(full_path.rstrip('/'))
            if parent == '': parent = '/'
            back_path = '' if parent == '/' else parent
            entries.append({'name': '...', 'type': 'directory', 'path': back_path})

        for f, meta in files:
            entries.append({
                'name': f,
                'type': meta['type'].lower(),
                'path': os.path.join(full_path, f).replace('\\', '/')
            })

        breadcrumbs = full_path.strip('/').split('/')
        crumbs = []
        cur = ''
        for b in breadcrumbs:
            cur += '/' + b
            crumbs.append({'name': b or 'root', 'path': cur})
        
        entries.sort(key=lambda x: (x['type'] != 'directory', x['name']))

        return render_template('storage.html', entries=entries, current_path=full_path, breadcrumbs=crumbs)
    except Exception as e:
        return f"Lỗi khi duyệt HDFS: {e}"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Lấy danh sách file (có thể nhiều)
        files = request.files.getlist('file')
        path = request.form.get('path', '/')
        
        if not files or len(files) == 0:
            return jsonify({'message': 'Không có file nào được chọn để upload'}), 400
        
        uploaded_files = []
        
        for file in files:
            hdfs_path = ('/' + os.path.join(path, file.filename).lstrip('/')).replace('\\', '/')
            client.write(hdfs_path, file, overwrite=True)
            uploaded_files.append(hdfs_path)
        
        return jsonify({
            'message': f'Upload thành công {len(uploaded_files)} file',
            'uploaded_files': uploaded_files
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'message': f"Lỗi khi upload: {type(e).__name__}: {e}"
        }), 500


@app.route('/mkdir', methods=['POST'])
def make_directory():
    try:
        current_path = request.form.get('current_path', '/')
        new_dir_name = request.form.get('dir_name')
        if not new_dir_name:
            return jsonify({'message': 'Tên thư mục không được để trống'}), 400
        hdfs_path = os.path.join(current_path, new_dir_name).replace('\\', '/')
        client.makedirs(hdfs_path) 
        return jsonify({'message': f'Tạo thư mục thành công: {hdfs_path}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f"Lỗi tạo thư mục HDFS: {type(e).__name__}: {e}"}), 500

@app.route('/download/<path:path>')
def download(path):
    try:
        full_path = '/' + path.lstrip('/')
        with client.read(full_path) as reader:
            data = BytesIO(reader.read())
        filename = os.path.basename(path)
        return send_file(data, as_attachment=True, download_name=filename)
    except Exception as e:
        traceback.print_exc()
        return f"Lỗi tải file: {e}"

@app.route('/download_zip/<path:path>')
def download_zip(path):
    full_path = '/' + path.lstrip('/')
    dir_name = os.path.basename(full_path)
    zip_filename = f"{dir_name}.zip"

    with tempfile.TemporaryFile() as temp_zip_file:
        with zipfile.ZipFile(temp_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            try:
                def zip_dir(hdfs_path, current_dir_name):
                    for name, status in client.list(hdfs_path, status=True):
                        item_path = os.path.join(hdfs_path, name).replace('\\', '/')
                        arcname = os.path.join(current_dir_name, name).replace('\\', '/')

                        if status['type'] == 'FILE':
                            with client.read(item_path) as reader:
                                data = reader.read()
                            zipf.writestr(arcname, data)
                        elif status['type'] == 'DIRECTORY':
                            zip_dir(item_path, arcname)

                zip_dir(full_path, dir_name)

            except Exception as e:
                traceback.print_exc()
                return f"Lỗi khi nén thư mục HDFS: {e}", 500

        temp_zip_file.seek(0)
        
        return send_file(
            BytesIO(temp_zip_file.read()),
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )

@app.route('/delete/<path:path>', methods=['DELETE'])
def delete_file(path):
    try:
        hdfs_path = '/' + path.lstrip('/')
        client.delete(hdfs_path, recursive=True)
        return jsonify({'message': f'Xóa thành công: {hdfs_path}'})
    except HdfsError as e:
        traceback.print_exc()
        return jsonify({'message': f'Lỗi HDFS khi xóa: {e}'}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': f'Lỗi không xác định khi xóa: {e}'}), 500

# --- ROUTES CHO XỬ LÝ DỮ LIỆU (ANALYSIS) ---

@app.route('/analysis')
def analysis_form():
    try:
        # files_for_analysis là list of tuples: (name, metadata)
        files_for_analysis = [f for f in client.list('/', status=True) if f[1]['type'] == 'FILE']
        return render_template('analysis.html', files=files_for_analysis) 
    except Exception as e:
        return f"Lỗi khi truy cập HDFS cho phân tích: {e}"

# --- HÀM DỰ ĐOÁN (CẦN PHẢI CÓ) ---
def perform_prediction(row, result_lines):
    # Đảm bảo các mô hình đã được tải thành công
    global DELAY_MODEL, CAUSE_MODEL, LABEL_ENCODER
    
    # CÁC CỘT ĐẦU VÀO CỦA MÔ HÌNH DỰ ĐOÁN
    feature_cols = ["Month", "DayofMonth", "DayOfWeek", "TaxiOut", "Distance"]
    
    # 1. Chuẩn bị Feature Vector (sử dụng các cột đã làm sạch)
    # Kiểm tra xem row là Series hay DataFrame
    if isinstance(row, pd.Series):
        features_dict = row[feature_cols].apply(lambda x: x if pd.notna(x) else None).to_dict()
        df_row = row.to_frame().T
    elif isinstance(row, pd.DataFrame):
        features_dict = row[feature_cols].iloc[0].apply(lambda x: x if pd.notna(x) else None).to_dict()
        df_row = row
    else:
        result_lines.append("Lỗi: Dữ liệu đầu vào cho dự đoán không đúng định dạng.")
        return None

    # Chuyển đổi thành DMatrix cho XGBoost
    try:
        data_matrix = xgb.DMatrix(df_row[feature_cols].values, feature_names=feature_cols)
    except Exception as e:
        result_lines.append(f"Lỗi tạo DMatrix: {e}")
        return None
        
    # 2. Dự đoán Delay/No Delay (Binary Classification)
    delay_proba = DELAY_MODEL.predict(data_matrix)[0] # Xác suất trễ (label 1)
    
    delay_prediction_status = "KHÔNG BỊ TRỄ"
    if delay_proba >= 0.5:
        delay_prediction_status = "CÓ KHẢ NĂNG BỊ TRỄ"
        
    delay_probability_text = f"{delay_proba*100:.2f}%"

    # 3. Dự đoán Nguyên nhân Trễ (Multi-class Classification)
    predicted_cause = "Không áp dụng"
    if delay_prediction_status == "CÓ KHẢ NĂNG BỊ TRỄ":
        cause_prediction_raw = CAUSE_MODEL.predict(data_matrix)
        predicted_cause_index = cause_prediction_raw.argmax()
        
        # Giải mã nhãn
        try:
            predicted_cause_encoded = LABEL_ENCODER.classes_[predicted_cause_index]
            cause_classes_map = {
                'CarrierDelay': 'Do Hãng hàng không',
                'LateAircraftDelay': 'Do Máy bay đến trễ',
                'NASDelay': 'Do Hệ thống không lưu (NAS)',
                'SecurityDelay': 'Do An ninh',
                'WeatherDelay': 'Do Thời tiết',
            }
            predicted_cause = cause_classes_map.get(predicted_cause_encoded, predicted_cause_encoded)

        except Exception as e:
            result_lines.append(f"Lỗi giải mã nhãn nguyên nhân: {e}")
            predicted_cause = f"Lỗi giải mã (Index: {predicted_cause_index})"
    else:
        predicted_cause = "Không có dự đoán nguyên nhân do mô hình cho rằng chuyến bay không trễ"

    # 4. Trích xuất Thực tế (Actuals)
    # Lấy các cột thực tế từ file mới
    actual_arr_delay = row.get('ArrDelay')
    
    actual_status = "ĐÚNG GIỜ"
    # Điều kiện trễ thực tế: ArrDelay > 15 phút
    actual_is_delayed = actual_arr_delay is not None and pd.notna(actual_arr_delay) and actual_arr_delay > 15
    
    if actual_is_delayed:
        actual_status = f"TRỄ THỰC TẾ (> 15 phút)"
    elif actual_arr_delay is not None and pd.notna(actual_arr_delay) and actual_arr_delay > 0:
        actual_status = f"TRỄ NHẸ THỰC TẾ (0 < ArrDelay ≤ 15)"

        
    # Trích xuất chi tiết nguyên nhân thực tế
    actual_cause_detail = "Không áp dụng"
    if actual_is_delayed:
        delay_causes = {
            "CarrierDelay": row.get('CarrierDelay', 0.0),
            "WeatherDelay": row.get('WeatherDelay', 0.0),
            "NASDelay": row.get('NASDelay', 0.0),
            "SecurityDelay": row.get('SecurityDelay', 0.0),
            "LateAircraftDelay": row.get('LateAircraftDelay', 0.0)
        }
        
        # Chỉ xét các nguyên nhân > 0
        actual_causes_list = [f"<b>{k}</b>: {v:.0f}p" for k, v in delay_causes.items() if v is not None and v > 0]
        actual_cause_detail = "<br>".join(actual_causes_list) or "Không có nguyên nhân chi tiết được ghi nhận (Tất cả đều 0)"
    
    # 5. So sánh Dự đoán và Thực tế
    match_status = "Không thể xác định"
    if actual_arr_delay is not None and pd.notna(actual_arr_delay):
        # Trùng khớp nếu cả hai cùng dự đoán là trễ (>15p) hoặc cùng dự đoán không trễ (<=15p)
        if (delay_prediction_status == "CÓ KHẢ NĂNG BỊ TRỄ" and actual_is_delayed) or \
           (delay_prediction_status == "KHÔNG BỊ TRỄ" and not actual_is_delayed):
             match_status = "✅ TRÙNG KHỚP"
        else:
             match_status = "❌ KHÔNG TRÙNG KHỚP"
    else:
        actual_status = "KHÔNG CÓ ArrDelay GỐC"


    # 6. Trả về kết quả
    return {
        'delay_prediction': delay_prediction_status,
        'delay_probability': delay_probability_text,
        'cause_prediction': predicted_cause,
        'actual_status': actual_status,
        'actual_arr_delay': f"{actual_arr_delay:.2f}p" if actual_arr_delay is not None and pd.notna(actual_arr_delay) else "N/A",
        'actual_cause_detail': actual_cause_detail,
        'match_status': match_status,
        'processed_features': "\n".join([f"{k}: {v}" for k, v in features_dict.items() if v is not None])
    }
    
@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    """Thực hiện một tác vụ phân tích phân tán bằng Apache Spark."""
    file_path = request.form.get('file_path')
    analysis_type = request.form.get('analysis_type')

    spark_session = get_spark_session(HDFS_RPC_URL) 
    
    if spark_session is None:
        return render_template('result.html', lines=["LỖI: Không thể khởi tạo Spark Session. Vui lòng kiểm tra cài đặt PySpark, JAVA_HOME, và SPARK_HOME."], analysis_type=analysis_type), 500

    if not file_path:
        return jsonify({'message': 'Vui lòng chọn file để phân tích'}), 400
        
    result_lines = []
    plot_base64 = None 
    prediction_data = None 
    hdfs_full_path = file_path 

    # --- ĐỊNH NGHĨA CÁC LOẠI PHÂN TÍCH CSV ---
    csv_analysis_types = [
        'avg_delay_by_origin', 
        'percentage_delayed_flights', 
        'delay_causes_breakdown', 
        'delay_prediction', 
        'feature_importance_analysis'
    ]

    try:
        if analysis_type in csv_analysis_types:
            
            # Đọc CSV 
            raw_df = spark_session.read.csv(hdfs_full_path, header=True)
            
            # --- LÀM SẠCH VÀ ÉP KIỂU TẤT CẢ CÁC CỘT CẦN THIẾT ---
            
            # CÁC CỘT CẦN THIẾT TỪ CẤU TRÚC MỚI (Số học)
            delay_cols = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
            # CÁC CỘT DỰ ĐOÁN TỪ CẤU TRÚC MỚI (Số học)
            prediction_cols = ["Month", "DayofMonth", "DayOfWeek", "TaxiOut", "Distance"]
            # Cột Origin để phân tích theo sân bay (Chuỗi)
            analysis_cols = ["Origin"]
            
            all_needed_cols = delay_cols + prediction_cols + analysis_cols
            
            df = raw_df
            
            # Lặp qua các cột số và ép kiểu/làm sạch
            columns_to_clean = delay_cols + prediction_cols
            for col_name in columns_to_clean:
                if col_name in df.columns:
                    # Ghi đè cột gốc bằng cột đã làm sạch và ép kiểu Double
                    df = df.withColumn(
                        col_name,
                        regexp_replace(col(col_name).cast(StringType()), "[^\\d\\.\\-]", "").cast(DoubleType())
                    )
                else:
                    result_lines.append(f"CẢNH BÁO: Không tìm thấy cột '{col_name}' trong file. Phân tích/Dự đoán bị ảnh hưởng.")
            
            # Lấy dòng đầu tiên để hiển thị log
            with client.read(hdfs_full_path, encoding='utf-8') as reader:
                try:
                    all_lines = reader.readlines()
                    original_raw_line = all_lines[1].strip() if len(all_lines) > 1 else "Không thể đọc dòng dữ liệu gốc đầu tiên."
                except:
                    original_raw_line = "Không thể đọc dòng dữ liệu gốc đầu tiên."
            
            # DataFrame sạch chỉ giữ lại các cột cần thiết sau khi đã ép kiểu
            df_clean = df.select(*all_needed_cols).dropna(subset=prediction_cols + ["ArrDelay"])
            
            # ----------------------------------------------------
            # BẮT ĐẦU CHUỖI IF/ELIF CHO CÁC LOẠI PHÂN TÍCH CSV KHÁC NHAU
            # ----------------------------------------------------

            # --- LOGIC CHO feature_importance_analysis (ĐÃ SỬA LỖI CÚ PHÁP: elif -> if) ---
            if analysis_type == 'feature_importance_analysis':
                if not MODEL_LOAD_SUCCESS:
                    result_lines.append("LỖI: Không thể tải mô hình (DELAY_MODEL hoặc CAUSE_MODEL). Vui lòng kiểm tra file trên HDFS.")
                    plot_base64 = None
                else:
                    result_lines.append("--- BƯỚC 1: PHÂN TÍCH TẦM QUAN TRỌNG CỦA ĐẶC TRƯNG ---")
                    
                    # 1. PHÂN TÍCH TẦM QUAN TRỌNG CỦA ĐẶC TRƯNG (Vẽ 2 biểu đồ Features)
                    delay_plot_base64, delay_data = generate_feature_importance_plot(
                        DELAY_MODEL, 
                        "Tầm quan trọng Đặc trưng - Dự đoán Độ trễ (Delay/No Delay)"
                    )
                    cause_plot_base64, cause_data = generate_feature_importance_plot(
                        CAUSE_MODEL, 
                        "Tầm quan trọng Đặc trưng - Dự đoán Nguyên nhân Trễ (Multi-Class)"
                    )
                    
                    # Lưu 2 biểu đồ Feature Importance vào log để hiển thị chi tiết
                    result_lines.append("\nChi tiết Feature Importance Model Delay:")
                    for item in delay_data: result_lines.append(f"  - {item['FeatureName']}: {item['Score']:.4f}")
                    result_lines.append("\nChi tiết Feature Importance Model Cause:")
                    for item in cause_data: result_lines.append(f"  - {item['FeatureName']}: {item['Score']:.4f}")

                    
                    result_lines.append("\n--- BƯỚC 2: ÁP DỤNG MÔ HÌNH VÀ TÍNH TỶ LỆ TRÊN DỮ LIỆU NỀN ---")

                    # 2. ÁP DỤNG MÔ HÌNH LÊN DỮ LIỆU SẠCH (df_clean)
                    
                    # Chuyển đổi df_clean sang Pandas để chạy dự đoán hàng loạt (hiệu quả hơn)
                    df_pd = df_clean.select(*df_clean.columns).toPandas()
                    
                    if df_pd.empty:
                        result_lines.append("LỖI DỮ LIỆU: Không có dữ liệu hợp lệ để áp dụng mô hình.")
                        plot_base64 = None
                    else:
                        feature_order = prediction_cols # Sử dụng các cột dự đoán mới
                        input_matrix = xgb.DMatrix(df_pd[feature_order].values, feature_names=feature_order)
                        
                        # Dự đoán Delay/No Delay (Xác suất trễ)
                        delay_probs = DELAY_MODEL.predict(input_matrix)
                        df_pd['PredictedDelay'] = delay_probs
                        df_pd['IsDelayed'] = (df_pd['PredictedDelay'] >= 0.5).astype(int) # 1: Delayed, 0: On-time
                        
                        # Dự đoán Cause 
                        cause_preds_raw = CAUSE_MODEL.predict(input_matrix)
                        predicted_cause_indices = cause_preds_raw.argmax(axis=1)
                        
                        # Giải mã nhãn (LabelEncoder)
                        predicted_causes_encoded = LABEL_ENCODER.inverse_transform(predicted_cause_indices)
                        
                        # Map nhãn nguyên nhân
                        cause_classes_map = {
                            'CarrierDelay': 'Hãng hàng không',
                            'LateAircraftDelay': 'Máy bay đến trễ',
                            'NASDelay': 'Hệ thống không lưu (NAS)',
                            'SecurityDelay': 'An ninh',
                            'WeatherDelay': 'Thời tiết',
                        }
                        
                        df_pd['PredictedCause'] = [
                            cause_classes_map.get(predicted_causes_encoded[i], 'Unknown') 
                            if df_pd.loc[i, 'IsDelayed'] == 1 else 'Not Delayed' 
                            for i in range(len(df_pd))
                        ]
                        
                        # 3. THỐNG KÊ VÀ TẠO BIỂU ĐỒ KẾT QUẢ
                        
                        # 3a. Tỷ lệ Delay
                        total_records = len(df_pd)
                        predicted_delay_count = df_pd['IsDelayed'].sum()
                        predicted_delay_percentage = (predicted_delay_count / total_records) * 100
                        
                        delay_stats = [
                            {'Status': 'Delayed', 'Percentage': predicted_delay_percentage},
                            {'Status': 'On-time', 'Percentage': 100 - predicted_delay_percentage}
                        ]
                        
                        # Biểu đồ 3: Tỷ lệ Delayed/On-time
                        result_lines.append(f"\nTổng số bản ghi đã áp dụng mô hình: {total_records}")
                        result_lines.append(f"Tỷ lệ Dự đoán Bị trễ: {predicted_delay_percentage:.2f}%")
                        delay_rate_plot = generate_plot(delay_stats, file_path, "Tỷ lệ Dự đoán Bị Trễ/Đúng giờ", "Tỷ lệ (%)", "Trạng thái", "Percentage", "Status", horizontal=False) # Vẽ cột dọc
                        
                        # 3b. Tỷ trọng Nguyên nhân Delay
                        cause_counts = df_pd[df_pd['IsDelayed'] == 1]['PredictedCause'].value_counts()
                        
                        cause_breakdown = []
                        for cause, count in cause_counts.items():
                            percentage = (count / predicted_delay_count) * 100 if predicted_delay_count > 0 else 0
                            cause_breakdown.append({'Cause': cause, 'Percentage': percentage})
                            
                        # Biểu đồ 4: Tỷ trọng Nguyên nhân (chỉ hiển thị nếu có trễ)
                        if predicted_delay_count > 0:
                            cause_breakdown_plot = generate_plot(cause_breakdown, file_path, "Tỷ trọng Nguyên nhân Độ trễ Dự đoán", "Tỷ lệ (%)", "Nguyên nhân", "Percentage", "Cause")
                            result_lines.append("\nChi tiết Tỷ trọng Nguyên nhân Dự đoán:")
                            for item in cause_breakdown: result_lines.append(f"  - {item['Cause']}: {item['Percentage']:.2f}%")
                        else:
                            cause_breakdown_plot = None
                            result_lines.append("\nKhông có bản ghi nào được dự đoán là bị trễ nên không phân tích nguyên nhân.")


                        # Gộp tất cả 4 biểu đồ vào dictionary để gửi về result.html
                        plot_base64 = {
                            'delay_rate_plot': delay_rate_plot,
                            'delay_cause_plot': cause_breakdown_plot,
                            'feature_delay': delay_plot_base64,
                            'feature_cause': cause_plot_base64
                        }
            
            # --- LOGIC CHO delay_prediction ---
            elif analysis_type == 'delay_prediction':
                if not MODEL_LOAD_SUCCESS:
                    result_lines.append("LỖI: Không thể tải mô hình. Chức năng dự đoán không thể thực hiện.")
                else:
                    result_lines.append("Đang thực hiện Dự đoán Độ trễ và Nguyên nhân cho bản ghi đầu tiên...")
                    
                    # Lấy dòng đầu tiên hợp lệ và chuyển sang Pandas
                    first_row_pd = df_clean.limit(1).toPandas()
                    
                    if not first_row_pd.empty:
                        # Gọi hàm helper
                        prediction_data = perform_prediction(first_row_pd.iloc[0], result_lines)
                        prediction_data['original_raw_line'] = original_raw_line
                    else:
                        result_lines.append("Không có dữ liệu hợp lệ trong file để dự đoán.")


            # --- CÁC PHÂN TÍCH THỐNG KÊ CÒN LẠI (Đã cập nhật để dùng cột mới) ---
            elif analysis_type == 'avg_delay_by_origin':
                result_df = df_clean.groupBy("Origin").agg(
                    spark_round(avg(col("ArrDelay")), 2).alias("AvgArrivalDelay")
                ).filter(col("AvgArrivalDelay").isNotNull()).orderBy(col("AvgArrivalDelay").desc())

                top_10 = result_df.limit(10).collect()
                result_lines = [f"{row['Origin']}: {row['AvgArrivalDelay']} phút" for row in top_10]
                plot_base64 = generate_plot(top_10, file_path, "Độ Trễ Đến Trung Bình (Top 10)", "Thời gian trễ trung bình (phút)", "Sân bay (Origin)", "AvgArrivalDelay", "Origin")

            elif analysis_type == 'percentage_delayed_flights':
                total_flights = df_clean.count()
                delay_threshold = 15.0
                # ArrDelay đã được làm sạch và ép kiểu nên dùng trực tiếp
                delayed_flights_df = df_clean.withColumn("IsDelayed", when(col("ArrDelay") > delay_threshold, 1).otherwise(0))
                
                result_df = delayed_flights_df.groupBy("Origin").agg(
                    spark_round((spark_count(when(col("IsDelayed") == 1, 1)) / spark_count("*")) * 100, 2).alias("DelayPercentage")
                ).filter(col("DelayPercentage").isNotNull()).orderBy(col("DelayPercentage").desc())

                top_10 = result_df.limit(10).collect()
                result_lines = [f"{row['Origin']}: {row['DelayPercentage']}%" for row in top_10]
                plot_base64 = generate_plot(top_10, file_path, "Tỷ lệ Chuyến bay Bị Trễ (Top 10)", "Tỷ lệ (%)", "Sân bay (Origin)", "DelayPercentage", "Origin")

            elif analysis_type == 'delay_causes_breakdown':
                delay_causes = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
                # Các cột nguyên nhân đã được làm sạch và ép kiểu nên dùng trực tiếp
                df_delayed = df_clean.filter(col("ArrDelay") > 0).fillna(0.0, subset=delay_causes)
                
                # Tính tổng từng loại nguyên nhân
                agg_exprs = [sum(col(c)).alias(f"Sum_{c}") for c in delay_causes]
                total_delay_sum = df_delayed.agg(*agg_exprs).collect()[0]
                
                # Tính tổng trong Python sau khi đã collect (dùng built-in sum — tránh xung đột với pyspark.sql.functions.sum)
                total_sum = builtins.sum(float(total_delay_sum[f"Sum_{c}"]) for c in delay_causes)

                if total_sum > 0:
                    breakdown_data = []
                    for cause in delay_causes:
                        value = total_delay_sum[f"Sum_{cause}"]
                        percentage = (value / total_sum) * 100 if value is not None else 0
                        breakdown_data.append({'Cause': cause, 'Percentage': percentage})
                    
                    breakdown_data.sort(key=lambda x: x['Percentage'], reverse=True)
                    
                    result_lines = [f"{item['Cause']}: {item['Percentage']:.2f}%" for item in breakdown_data]
                    plot_base64 = generate_plot(breakdown_data, file_path, "Tỷ trọng Nguyên nhân Độ trễ", "Tỷ lệ (%)", "Nguyên nhân", "Percentage", "Cause")
                else:
                    result_lines.append("Không tìm thấy tổng độ trễ dương để phân tích nguyên nhân.")
            
            # Khối này không cần thiết vì đã có check ở đầu hàm, nhưng giữ lại để bắt lỗi nếu có
            else:
                 raise ValueError("Loại phân tích CSV không hợp lệ đã vượt qua bộ lọc ban đầu.")

        # --- LOGIC CHO LINE COUNT & WORD COUNT (TEXT FILES) ---
        elif analysis_type == 'line_count':
            count = spark_session.sparkContext.textFile(hdfs_full_path).count()
            result_lines.append(f"File '{file_path}' có tổng cộng {count} dòng.")

        elif analysis_type == 'word_count':
            lines = spark_session.sparkContext.textFile(hdfs_full_path).take(1001) 
            if len(lines) > 1:
                lines = lines[1:]
            rdd = spark_session.sparkContext.parallelize(lines)
            
            word_counts = rdd.flatMap(lambda line: split(line, '\s+')) \
                            .filter(lambda word: word != "") \
                            .map(lambda word: (lower(word), 1)) \
                            .reduceByKey(lambda a, b: a + b) \
                            .map(lambda x: (x[1], x[0])) \
                            .sortByKey(False) \
                            .take(10)
                            
            top_10 = [{'word': word, 'count': count} for count, word in word_counts]
            result_lines = [f"{i+1}. {item['word']}: {item['count']}" for i, item in enumerate(top_10)]
            plot_base64 = generate_word_count_plot(top_10, file_path)

        else:
            return jsonify({'message': 'Loại phân tích không hợp lệ'}), 400

    except ValueError as ve:
        return jsonify({'message': f'{ve}'}), 400
        
    except HdfsError as e:
        result_lines.append(f"LỖI HDFS: Không thể truy cập file {hdfs_full_path}. Lỗi: {e}")
        traceback.print_exc()
        
    except Exception as e:
        result_lines.append(f"LỖỖI PHÂN TÍCH: {type(e).__name__}: {e}")
        traceback.print_exc()

    return render_template('result.html', 
                            lines=result_lines, 
                            analysis_type=analysis_type, 
                            plot_base64=plot_base64, 
                            prediction_data=prediction_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)