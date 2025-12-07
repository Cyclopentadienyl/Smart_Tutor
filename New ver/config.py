import os

# --- 路徑設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# 確保目錄存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 檔案名稱 ---
STUDENT_DATA_FILE = os.path.join(RAW_DATA_DIR, "students_simulated.csv")
MODEL_FILE_CLUSTERING = os.path.join(MODEL_DIR, "kmeans_model.pkl")
MODEL_FILE_PREDICTION = os.path.join(MODEL_DIR, "predictor_model.pkl")

# --- 資料欄位定義 (對應您的新 CSV) ---
# 必須與您上傳的檔案 header 完全一致
COL_STUDENT_ID = "Student_ID"
COL_NAME = "Name"
COL_ACCURACY = "Accuracy"              # 數值: 0.85
COL_AVG_TIME = "avg_completion_time"   # 數值: 15.5
COL_SCORE_HISTORY = "avg_score"        # 字串: "[90, 85, 70]" (注意：這是列表字串)
COL_ERROR_TYPES = "Error_Types"        # 字串: "{'reading': 1, 'vocab': 0}"
COL_LEARNING_PACE = "Learning_Pace"    # 數值: 10.0
COL_ATTENDANCE = "Attendance"          # 數值: 0.95
COL_HW_COMPLETION = "HW_Completion"    # 數值: 1.0

# 系統生成的分析結果欄位
COL_GROUP = "Group"
COL_RECOMMENDED_LEVEL = "Recommended_Level"

# --- 系統參數 ---
RANDOM_SEED = 42
N_CLUSTERS = 3
