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
TASK_LOG_FILE = os.path.join(RAW_DATA_DIR, "task_log.csv")
MODEL_FILE_CLUSTERING = os.path.join(MODEL_DIR, "kmeans_model.pkl")
MODEL_FILE_PREDICTION = os.path.join(MODEL_DIR, "predictor_model.pkl")

# --- 資料欄位定義 (接口規範) ---
# 輸入特徵 (Input Features)
COL_STUDENT_ID = "student_id"
COL_AVG_SCORE = "avg_score"           # 平均成績
COL_AVG_TIME = "avg_completion_time"  # 平均完成時間 (分)
COL_WEAKNESS = "weakness_tag"         # 弱點標籤 (如: 代數, 幾何)
COL_PERSONALITY = "personality_type"  # 個性類型 (如: 積極, 被動)
COL_PROGRESS = "current_progress"     # 目前進度 (單元 ID)

# 輸出/預測目標 (Outputs/Targets)
COL_GROUP = "assigned_group"          # 分群結果 (A/B/C)
COL_RECOMMENDED_LEVEL = "rec_level"   # 推薦難度 (Easy/Medium/Hard)

# --- 系統參數 ---
RANDOM_SEED = 42
N_CLUSTERS = 3  # 預設分群數量