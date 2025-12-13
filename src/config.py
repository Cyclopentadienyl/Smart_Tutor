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
MODEL_FILE_CLUSTERING = os.path.join(MODEL_DIR, "xgb_proficiency_model.pkl")
MODEL_FILE_PREDICTION = os.path.join(MODEL_DIR, "predictor_model.pkl")

# --- 資料欄位定義 (接口規範) ---
# 輸入特徵 (Input Features)
COL_STUDENT_ID = "student_id"
COL_NAME = "name"
COL_ACCURACY = "accuracy"  # 答題正確率 (0-1)
COL_AVG_TIME = "avg_completion_time"  # 平均完成時間 (分鐘)
COL_LEARNING_PACE = "learning_pace"  # 學習步調
COL_ATTENDANCE = "attendance_rate"  # 出席率 (0-1)
COL_HW_COMPLETION = "homework_completion_rate"  # 作業完成率 (0-1)
COL_SCORE_HISTORY = "score_history"  # 歷史分數列表
COL_AVG_SCORE = "avg_score"  # 平均分數（替代方案，若無 score_history）
COL_ERROR_TYPES = "error_types"  # 錯誤類型 dict 字串
COL_RISK_LEVEL = "Risk_Status"  # 風險等級 (🔴/🟡/🟢)
COL_CURRENT_TOPIC = "Current_Topic" # 當前進度章節
COL_LAST_ACTIVE = "Last_Active"     # 最後上線時間

# 輸出/預測目標 (Outputs/Targets)
COL_GROUP = "Group"  # 以預測結果填充的分組標籤
COL_RECOMMENDED_LEVEL = "Recommended_Level"  # 推薦難度 (Easy/Medium/Hard)

# --- 系統參數 ---
RANDOM_SEED = 42
