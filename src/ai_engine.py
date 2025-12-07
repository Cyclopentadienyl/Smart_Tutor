import ast
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src import config


class TutorAI:
    def __init__(self):
        self.xgb_model: xgb.XGBClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.le_file_path = os.path.join(config.MODEL_DIR, "label_encoder.pkl")

    def _get_expert_label(self, row: pd.Series) -> str:
        """依據簡單規則生成監督標籤。"""
        acc = row.get(config.COL_ACCURACY, 0)
        att = row.get(config.COL_ATTENDANCE, 0)
        hw = row.get(config.COL_HW_COMPLETION, 0)
        score_sum = acc + att + hw

        if score_sum > 2.5:
            return "Hard (Challenge)"
        if score_sum > 1.8:
            return "Medium (Standard)"
        return "Easy (Review)"

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析錯誤型別與分數列表並清洗數值欄位。"""
        df_proc = df.copy()

        def parse_errors(val, key):
            try:
                if isinstance(val, str):
                    parsed = ast.literal_eval(val)
                else:
                    parsed = val
                return float(parsed.get(key, 0))
            except Exception:
                return 0.0

        if config.COL_ERROR_TYPES in df_proc.columns:
            df_proc["err_reading"] = df_proc[config.COL_ERROR_TYPES].apply(
                lambda x: parse_errors(x, "reading")
            )
            df_proc["err_vocab"] = df_proc[config.COL_ERROR_TYPES].apply(
                lambda x: parse_errors(x, "vocab")
            )
            df_proc["err_logic"] = df_proc[config.COL_ERROR_TYPES].apply(
                lambda x: parse_errors(x, "logic")
            )
        else:
            df_proc["err_reading"] = 0.0
            df_proc["err_vocab"] = 0.0
            df_proc["err_logic"] = 0.0

        def calc_mean_score(val):
            try:
                if isinstance(val, str):
                    scores = ast.literal_eval(val)
                else:
                    scores = val
                if isinstance(scores, list) and scores:
                    return float(np.mean(scores))
                return 0.0
            except Exception:
                return 0.0

        if config.COL_SCORE_HISTORY in df_proc.columns:
            df_proc["mean_score"] = df_proc[config.COL_SCORE_HISTORY].apply(calc_mean_score)
        elif config.COL_AVG_SCORE in df_proc.columns:
            df_proc["mean_score"] = pd.to_numeric(
                df_proc[config.COL_AVG_SCORE], errors="coerce"
            ).fillna(0.0)
        else:
            df_proc["mean_score"] = 0.0

        num_cols = [
            config.COL_ACCURACY,
            config.COL_AVG_TIME,
            config.COL_LEARNING_PACE,
            config.COL_ATTENDANCE,
            config.COL_HW_COMPLETION,
        ]

        for col in num_cols:
            if col in df_proc.columns:
                df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

        critical_cols = num_cols + ["mean_score", "err_reading", "err_vocab", "err_logic"]
        valid_check_cols = [col for col in critical_cols if col in df_proc.columns]
        df_proc = df_proc.dropna(subset=valid_check_cols)

        return df_proc

    def _get_feature_columns(self) -> list[str]:
        return [
            config.COL_ACCURACY,
            config.COL_AVG_TIME,
            config.COL_LEARNING_PACE,
            config.COL_ATTENDANCE,
            config.COL_HW_COMPLETION,
            "mean_score",
            "err_reading",
            "err_vocab",
            "err_logic",
        ]

    def train_prediction_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """訓練 XGBoost 分類模型以預測推薦難度。"""
        df_proc = self._preprocess_data(df)
        if df_proc.empty:
            return df

        features = df_proc[self._get_feature_columns()]
        df_aligned = df.loc[df_proc.index].copy()

        if config.COL_RECOMMENDED_LEVEL not in df_aligned.columns:
            labels = df_aligned.apply(self._get_expert_label, axis=1)
            df_aligned[config.COL_RECOMMENDED_LEVEL] = labels
        else:
            labels = df_aligned[config.COL_RECOMMENDED_LEVEL]

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=config.RANDOM_SEED,
            eval_metric="mlogloss",
        )
        self.xgb_model.fit(features, y_encoded)

        joblib.dump(self.xgb_model, config.MODEL_FILE_PREDICTION)
        joblib.dump(self.label_encoder, self.le_file_path)

        return df_aligned

    def predict_single(
        self,
        acc: float,
        time: float,
        pace: float,
        att: float,
        hw: float,
        e_read: float,
        e_vocab: float,
        e_logic: float,
        m_score: float,
    ) -> str:
        if self.xgb_model is None:
            if os.path.exists(config.MODEL_FILE_PREDICTION):
                self.xgb_model = joblib.load(config.MODEL_FILE_PREDICTION)
            else:
                return "⚠️ 模型尚未訓練"

        if self.label_encoder is None and os.path.exists(self.le_file_path):
            self.label_encoder = joblib.load(self.le_file_path)

        input_data = pd.DataFrame(
            {
                config.COL_ACCURACY: [acc],
                config.COL_AVG_TIME: [time],
                config.COL_LEARNING_PACE: [pace],
                config.COL_ATTENDANCE: [att],
                config.COL_HW_COMPLETION: [hw],
                "mean_score": [m_score],
                "err_reading": [e_read],
                "err_vocab": [e_vocab],
                "err_logic": [e_logic],
            }
        )

        try:
            pred_idx = self.xgb_model.predict(input_data[self._get_feature_columns()])[0]
            if self.label_encoder:
                return self.label_encoder.inverse_transform([pred_idx])[0]
            return str(pred_idx)
        except Exception as exc:
            return f"Error: {exc}"

    def run_analysis_pipeline(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        if df.empty:
            return df, "無資料"

        df_clean = self.train_prediction_model(df)
        log: list[str] = [f"✅ XGBoost 訓練完成 (有效資料: {len(df_clean)} 筆)"]

        df_proc = self._preprocess_data(df_clean)
        predictions = self.xgb_model.predict(df_proc[self._get_feature_columns()])
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        df_clean.loc[df_proc.index, config.COL_RECOMMENDED_LEVEL] = predicted_labels
        df_clean.loc[df_proc.index, config.COL_GROUP] = predicted_labels
        log.append("✅ XGBoost 推論完成")

        return df_clean, "\n".join(log)

    def run_inference_only(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        if df.empty:
            return df, "❌ 檔案為空"

        df_proc = self._preprocess_data(df)
        if df_proc.empty:
            return df, "❌ 資料清洗後為空"

        df_clean = df.loc[df_proc.index].copy()

        if os.path.exists(config.MODEL_FILE_PREDICTION) and os.path.exists(self.le_file_path):
            self.xgb_model = joblib.load(config.MODEL_FILE_PREDICTION)
            self.label_encoder = joblib.load(self.le_file_path)

            predictions = self.xgb_model.predict(df_proc[self._get_feature_columns()])
            predicted_labels = self.label_encoder.inverse_transform(predictions)

            df_clean[config.COL_RECOMMENDED_LEVEL] = predicted_labels
            df_clean[config.COL_GROUP] = predicted_labels
            log = "✅ 舊 XGBoost 模型推論完成"
        else:
            df_clean[config.COL_RECOMMENDED_LEVEL] = "Unknown"
            df_clean[config.COL_GROUP] = "Unknown"
            log = "⚠️ 無法載入模型"

        return df_clean, log