import os

import numpy as np
import pandas as pd

from src import config


class DataManager:
    """負責資料讀取與生成 (含自動清洗空值功能)。"""

    def __init__(self):
        self.student_file = config.STUDENT_DATA_FILE

    def generate_mock_data(self, num_students: int = 100) -> pd.DataFrame:
        """生成模擬資料。"""
        print(f"[Info] 正在生成新格式資料 (N={num_students})...")
        np.random.seed(config.RANDOM_SEED)

        ids = [f"S{str(i + 1).zfill(3)}" for i in range(num_students)]
        names = [f"Student_{str(i + 1).zfill(3)}" for i in range(num_students)]

        accuracies = np.round(np.random.uniform(0.4, 0.99, num_students), 2)
        times = 30.0 - (accuracies * 20.0) + np.random.normal(0, 2, num_students)
        times = np.round(np.clip(times, 5.0, 40.0), 1)
        paces = np.round(np.random.uniform(5.0, 20.0, num_students), 1)
        attendances = np.round(np.random.uniform(0.6, 1.0, num_students), 2)
        hw_completions = np.round(np.random.uniform(0.5, 1.0, num_students), 2)

        score_histories_str: list[str] = []
        error_types_str: list[str] = []

        for _ in range(num_students):
            scores = np.random.randint(60, 100, 5).tolist()
            score_histories_str.append(str(scores))

            errors = {
                "reading": int(np.random.randint(0, 5)),
                "vocab": int(np.random.randint(0, 5)),
                "logic": int(np.random.randint(0, 5)),
            }
            error_types_str.append(str(errors))

        df = pd.DataFrame(
            {
                config.COL_STUDENT_ID: ids,
                config.COL_NAME: names,
                config.COL_ACCURACY: accuracies,
                config.COL_AVG_TIME: times,
                config.COL_SCORE_HISTORY: score_histories_str,
                config.COL_ERROR_TYPES: error_types_str,
                config.COL_LEARNING_PACE: paces,
                config.COL_ATTENDANCE: attendances,
                config.COL_HW_COMPLETION: hw_completions,
            }
        )

        os.makedirs(os.path.dirname(self.student_file), exist_ok=True)
        df.to_csv(self.student_file, index=False)
        return df

    def load_data(self) -> pd.DataFrame:
        """讀取本地資料並清洗空值。"""
        if not os.path.exists(self.student_file):
            return self.generate_mock_data()

        try:
            df = pd.read_csv(self.student_file)
            original_len = len(df)
            df = df.dropna(how="any")
            if len(df) < original_len:
                print(f"[Info] 已自動過濾 {original_len - len(df)} 筆空值資料")

            if config.COL_ACCURACY not in df.columns:
                print("[Warning] 舊格式資料，重新生成...")
                return self.generate_mock_data()

            return df
        except Exception as exc:
            print(f"[Error] {exc}，重新生成...")
            return self.generate_mock_data()

    def save_results(self, df: pd.DataFrame) -> str:
        df.to_csv(self.student_file, index=False)
        return "資料已更新並儲存。"

    def load_uploaded_file(self, file_path: str) -> pd.DataFrame:
        """讀取上傳檔案並立即清洗空值。"""
        if file_path is None:
            return pd.DataFrame()
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
            else:
                return pd.DataFrame()

            df = df.dropna(how="all")
            df = df.dropna(subset=[config.COL_ACCURACY, config.COL_AVG_TIME])

            return df
        except Exception as exc:
            print(f"[Error] 讀取失敗: {exc}")
            return pd.DataFrame()