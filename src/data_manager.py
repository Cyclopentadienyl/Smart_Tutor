import pandas as pd
import numpy as np
import os
from src import config

class DataManager:
    """
    負責處理所有資料的讀取、寫入與模擬生成。
    """
    def __init__(self):
        self.student_file = config.STUDENT_DATA_FILE

    def generate_mock_data(self, num_students=100):
        """
        生成模擬的學生資料，用於初始化系統。
        包含：成績、時間、弱點、個性等。
        """
        np.random.seed(config.RANDOM_SEED)
        
        ids = [f"S{str(i).zfill(3)}" for i in range(num_students)]
        
        # 模擬數據分佈
        # 1. 成績 (0-100)
        scores = np.random.normal(loc=70, scale=15, size=num_students)
        scores = np.clip(scores, 0, 100).astype(int)
        
        # 2. 完成時間 (10-60分鐘) - 成績越好通常越快(負相關)，但也加入隨機性
        times = 60 - (scores * 0.4) + np.random.normal(0, 5, num_students)
        times = np.clip(times, 5, 90).astype(int)
        
        # 3. 類別型資料 (弱點、個性)
        weaknesses = np.random.choice(['Algebra', 'Geometry', 'Calculus', 'Statistics'], num_students)
        personalities = np.random.choice(['Proactive', 'Passive', 'Anxious', 'Steady'], num_students)
        
        df = pd.DataFrame({
            config.COL_STUDENT_ID: ids,
            config.COL_AVG_SCORE: scores,
            config.COL_AVG_TIME: times,
            config.COL_WEAKNESS: weaknesses,
            config.COL_PERSONALITY: personalities,
            config.COL_PROGRESS: np.random.randint(1, 10, num_students) # 目前進度單元 1-10
        })
        
        # 存檔
        df.to_csv(self.student_file, index=False)
        print(f"[Info] 模擬資料已生成：{self.student_file}")
        return df

    def load_data(self):
        """讀取學生資料，如果不存在則自動生成"""
        if not os.path.exists(self.student_file):
            return self.generate_mock_data()
        return pd.read_csv(self.student_file)

    def save_results(self, df):
        """儲存包含分群或預測結果的資料表"""
        df.to_csv(self.student_file, index=False)
        return "資料已更新並儲存。"