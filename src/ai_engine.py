import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from src import config
import torch # 雖然目前主要用 Sklearn，但引入 Torch 以符合您的環境要求

class TutorAI:
    """
    AI 核心類別：負責執行分群 (Clustering) 與 推薦 (Recommendation)。
    """
    def __init__(self):
        self.kmeans_model = None
        self.rf_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] AI Engine initialized on device: {self.device}")

    def train_clustering(self, df: pd.DataFrame):
        """
        使用 K-Means 根據 '成績' 和 '時間' 進行學生分群。
        """
        # 準備訓練資料 (只取數值型特徵)
        X = df[[config.COL_AVG_SCORE, config.COL_AVG_TIME]].values
        
        # 初始化並訓練 K-Means
        self.kmeans_model = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_SEED)
        self.kmeans_model.fit(X)
        
        # 保存模型
        joblib.dump(self.kmeans_model, config.MODEL_FILE_CLUSTERING)
        
        # 回傳分群標籤
        labels = self.kmeans_model.labels_
        return labels

    def predict_difficulty(self, score, time):
        """
        規則或模型預測：根據狀態推薦難度 (簡單示範規則，未來可換成訓練好的 RF 模型)
        """
        # 這裡展示一個簡單的邏輯，未來可用 train 好的模型取代
        if score > 80 and time < 20:
            return "Hard (Challenge)"
        elif score < 60:
            return "Easy (Review)"
        else:
            return "Medium (Standard)"

    def run_analysis_pipeline(self, df: pd.DataFrame):
        """
        執行完整的 AI 流程：分群 -> 標記 -> 推薦
        """
        if df.empty:
            return df, "錯誤：無資料"

        # 1. 執行分群
        labels = self.train_clustering(df)
        
        # 將結果寫回 DataFrame
        df[config.COL_GROUP] = labels
        
        # 2. 根據分群與規則產生推薦
        # 將分群數字轉為可讀文字 (假設 Group 0 是成績最好的，這需視分群結果而定，這裡先簡化)
        # 實際專案中需要分析 Cluster Center 來決定哪個群是資優群
        
        # 對每一列執行推薦預測
        df[config.COL_RECOMMENDED_LEVEL] = df.apply(
            lambda row: self.predict_difficulty(row[config.COL_AVG_SCORE], row[config.COL_AVG_TIME]), 
            axis=1
        )
        
        return df, "分析完成！模型已重新訓練並更新推薦結果。"