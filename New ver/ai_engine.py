import pandas as pd
import numpy as np
import joblib
import os
import ast
# 移除 K-Means 與 Random Forest
# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier

# 引入 XGBoost 與 LabelEncoder
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from src import config

class TutorAI:
    def __init__(self):
        self.xgb_model = None
        self.label_encoder = None
        self.le_file_path = os.path.join(config.MODEL_DIR, "label_encoder.pkl")

    def _get_expert_label(self, row):
        # 專家規則 (Ground Truth)
        acc = row.get(config.COL_ACCURACY, 0)
        att = row.get(config.COL_ATTENDANCE, 0)
        hw = row.get(config.COL_HW_COMPLETION, 0)
        score_sum = acc + att + hw
        
        if score_sum > 2.5: return "Hard (Challenge)"
        elif score_sum > 1.8: return "Medium (Standard)"
        else: return "Easy (Review)"

    def _preprocess_data(self, df):
        """特徵工程與清洗 (保留防呆修正)"""
        df_proc = df.copy()
        
        # 1. 解析 Error_Types
        def parse_errors(val, key):
            try:
                if isinstance(val, str): d = ast.literal_eval(val)
                else: d = val
                return float(d.get(key, 0))
            except: return 0.0

        if config.COL_ERROR_TYPES in df_proc.columns:
            df_proc['err_reading'] = df_proc[config.COL_ERROR_TYPES].apply(lambda x: parse_errors(x, 'reading'))
            df_proc['err_vocab'] = df_proc[config.COL_ERROR_TYPES].apply(lambda x: parse_errors(x, 'vocab'))
            df_proc['err_logic'] = df_proc[config.COL_ERROR_TYPES].apply(lambda x: parse_errors(x, 'logic'))
        else:
            df_proc['err_reading'] = 0.0
            df_proc['err_vocab'] = 0.0
            df_proc['err_logic'] = 0.0

        # 2. 解析 avg_score
        def calc_mean_score(val):
            try:
                if isinstance(val, str): lst = ast.literal_eval(val)
                else: lst = val
                if isinstance(lst, list) and len(lst) > 0: return float(np.mean(lst))
                return 0.0
            except: return 0.0
        
        if config.COL_SCORE_HISTORY in df_proc.columns:
            df_proc['mean_score'] = df_proc[config.COL_SCORE_HISTORY].apply(calc_mean_score)
        else:
            df_proc['mean_score'] = 0.0

        # 3. 確保數值欄位為 Float
        num_cols = [config.COL_ACCURACY, config.COL_AVG_TIME, config.COL_LEARNING_PACE, 
                   config.COL_ATTENDANCE, config.COL_HW_COMPLETION]
        
        for col in num_cols:
            if col in df_proc.columns:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        
        # 過濾空值
        critical_cols = num_cols + ['mean_score', 'err_reading', 'err_vocab', 'err_logic']
        valid_check_cols = [c for c in critical_cols if c in df_proc.columns]
        df_proc = df_proc.dropna(subset=valid_check_cols)

        return df_proc

    def _get_feature_columns(self):
        return [config.COL_ACCURACY, config.COL_AVG_TIME, config.COL_LEARNING_PACE, 
                config.COL_ATTENDANCE, config.COL_HW_COMPLETION,
                'mean_score', 'err_reading', 'err_vocab', 'err_logic']

    def train_prediction_model(self, df: pd.DataFrame):
        """訓練 XGBoost 模型"""
        df_proc = self._preprocess_data(df)
        if df_proc.empty: return df
        
        X = df_proc[self._get_feature_columns()]
        
        # 對齊 index
        df_aligned = df.loc[df_proc.index].copy()
        
        # 準備標籤 (Y)
        if config.COL_RECOMMENDED_LEVEL not in df_aligned.columns:
            y_str = df_aligned.apply(self._get_expert_label, axis=1)
            df_aligned[config.COL_RECOMMENDED_LEVEL] = y_str
        else:
            y_str = df_aligned[config.COL_RECOMMENDED_LEVEL]

        # 【關鍵修改】XGBoost 需要數值型的 Label (0, 1, 2)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_str)

        # 建立 XGBoost Classifier
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=config.RANDOM_SEED,
            eval_metric='mlogloss'
        )
        self.xgb_model.fit(X, y_encoded)
        
        # 儲存模型與編碼器
        joblib.dump(self.xgb_model, config.MODEL_FILE_PREDICTION)
        joblib.dump(self.label_encoder, self.le_file_path)
        
        return df_aligned

    def predict_single(self, acc, time, pace, att, hw, e_read, e_vocab, e_logic, m_score):
        # 載入模型與編碼器
        if self.xgb_model is None:
            if os.path.exists(config.MODEL_FILE_PREDICTION):
                self.xgb_model = joblib.load(config.MODEL_FILE_PREDICTION)
            else: return "⚠️ 模型尚未訓練"
            
        if self.label_encoder is None:
            if os.path.exists(self.le_file_path):
                self.label_encoder = joblib.load(self.le_file_path)

        # 準備輸入資料
        input_data = pd.DataFrame({
            config.COL_ACCURACY: [acc], config.COL_AVG_TIME: [time],
            config.COL_LEARNING_PACE: [pace], config.COL_ATTENDANCE: [att],
            config.COL_HW_COMPLETION: [hw], 'mean_score': [m_score],
            'err_reading': [e_read], 'err_vocab': [e_vocab], 'err_logic': [e_logic]
        })
        
        try:
            # 預測 (得到的是 0, 1, 2)
            pred_idx = self.xgb_model.predict(input_data[self._get_feature_columns()])[0]
            # 轉回字串 (Hard, Medium...)
            if self.label_encoder:
                return self.label_encoder.inverse_transform([pred_idx])[0]
            return str(pred_idx)
        except Exception as e: return f"Error: {e}"

    def run_analysis_pipeline(self, df: pd.DataFrame):
        if df.empty: return df, "無資料"
        log = []
        
        # 1. 訓練 XGBoost
        df_clean = self.train_prediction_model(df)
        log.append(f"✅ XGBoost 訓練完成 (有效資料: {len(df_clean)} 筆)")
        
        # 2. 進行預測 (自我驗證)
        df_proc = self._preprocess_data(df_clean)
        
        # 預測並轉碼
        y_pred_idx = self.xgb_model.predict(df_proc[self._get_feature_columns()])
        predicted_labels = self.label_encoder.inverse_transform(y_pred_idx)
        
        df_clean[config.COL_RECOMMENDED_LEVEL] = predicted_labels
        
        # 3. 更新 Group (既然不分群了，就用預測結果當作分組，方便繪圖)
        df_clean[config.COL_GROUP] = predicted_labels
        log.append("✅ XGBoost 推論完成")
        
        return df_clean, "\n".join(log)

    def run_inference_only(self, df: pd.DataFrame):
        if df.empty: return df, "❌ 檔案為空"
        log = []
        
        df_proc = self._preprocess_data(df)
        if df_proc.empty: return df, "❌ 資料清洗後為空"
        
        df_clean = df.loc[df_proc.index].copy()
        
        # 載入模型
        if os.path.exists(config.MODEL_FILE_PREDICTION) and os.path.exists(self.le_file_path):
            self.xgb_model = joblib.load(config.MODEL_FILE_PREDICTION)
            self.label_encoder = joblib.load(self.le_file_path)
            
            # 預測
            y_pred_idx = self.xgb_model.predict(df_proc[self._get_feature_columns()])
            predicted_labels = self.label_encoder.inverse_transform(y_pred_idx)
            
            df_clean[config.COL_RECOMMENDED_LEVEL] = predicted_labels
            df_clean[config.COL_GROUP] = predicted_labels # 用預測結果填補 Group
            
            log.append("✅ 舊 XGBoost 模型推論完成")
        else:
            df_clean[config.COL_RECOMMENDED_LEVEL] = "Unknown"
            df_clean[config.COL_GROUP] = "Unknown"
            log.append("⚠️ 無法載入模型")

        return df_clean, "\n".join(log)