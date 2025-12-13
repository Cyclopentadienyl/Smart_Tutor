import ast
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from src import config


class TutorAI:
    def __init__(self):
        self.xgb_model: xgb.XGBClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.le_file_path = os.path.join(config.MODEL_DIR, "label_encoder.pkl")

    def batch_evaluate_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """為全班學生生成風險燈號與當前狀態"""
        df_out = df.copy()
        
        # 確保有預測結果
        if config.COL_RECOMMENDED_LEVEL not in df_out.columns:
            # 如果還沒跑過模型，先跑一次簡單規則
            df_out[config.COL_RECOMMENDED_LEVEL] = df_out.apply(self._get_expert_label, axis=1)

        def assess_risk(row):
            acc = row.get(config.COL_ACCURACY, 0)
            att = row.get(config.COL_ATTENDANCE, 0)
            
            # 風險邏輯：準確率低 或 出席率低 -> 紅燈
            if acc < 0.6 or att < 0.7:
                return "🔴 High Risk"
            elif acc < 0.75:
                return "🟡 Warning"
            else:
                return "🟢 On Track"

        df_out[config.COL_RISK_LEVEL] = df_out.apply(assess_risk, axis=1)
        
        # 模擬分配「當前章節」(隨機分配給 demo 用)
        # 在真實系統中，這會從資料庫讀取
        import random
        topics = ["M101-基礎代數", "M102-幾何圖形", "E201-閱讀理解", "M103-進階應用"]
        df_out[config.COL_CURRENT_TOPIC] = [random.choice(topics) for _ in range(len(df_out))]
        
        return df_out

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

    def train_prediction_model(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        use_cv: bool = False,
        cv_folds: int = 5,
        return_details: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
        """
        訓練 XGBoost 分類模型以預測推薦難度。

        參數:
            df: 輸入資料
            test_size: 驗證集比例 (0.0-0.5)
            n_estimators: Boosting 迭代次數
            max_depth: 樹的最大深度
            learning_rate: 學習率
            subsample: 每次迭代的樣本採樣比例
            colsample_bytree: 每棵樹的特徵採樣比例
            use_cv: 是否執行交叉驗證
            cv_folds: 交叉驗證的折數
            return_details: 是否返回訓練詳情（用於進階UI）

        返回:
            如果 return_details=False: 返回 df_aligned (向後兼容)
            如果 return_details=True: 返回 (df_aligned, training_info)
        """
        df_proc = self._preprocess_data(df)
        if df_proc.empty:
            if return_details:
                return df, {}
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

        # 初始化訓練信息字典
        training_info = {}

        # Train/Val Split (用於可視化訓練過程)
        X_train, X_val, y_train, y_val = train_test_split(
            features, y_encoded,
            test_size=test_size,
            random_state=config.RANDOM_SEED,
            stratify=y_encoded
        )

        # 設定 XGBoost 模型
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            random_state=config.RANDOM_SEED,
            eval_metric=["mlogloss", "merror"],
            early_stopping_rounds=10
        )

        # 訓練模型（記錄 train & val 的 metrics）
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # 儲存模型
        joblib.dump(self.xgb_model, config.MODEL_FILE_PREDICTION)
        joblib.dump(self.label_encoder, self.le_file_path)

        # 如果需要返回詳細信息
        if return_details:
            # 1. 訓練歷程
            training_info['history'] = self.xgb_model.evals_result()

            # 2. 驗證集準確率
            y_pred = self.xgb_model.predict(X_val)
            training_info['val_accuracy'] = accuracy_score(y_val, y_pred)

            # 3. 特徵重要性
            feature_names = self._get_feature_columns()
            importance = self.xgb_model.feature_importances_
            training_info['feature_importance'] = dict(zip(feature_names, importance))

            # 4. Cross-Validation (可選)
            if use_cv:
                # 創建一個不帶 early_stopping 的臨時模型用於 CV
                # 因為 cross_val_score 內部 fit() 時不會提供 eval_set
                cv_model = xgb.XGBClassifier(
                    n_estimators=int(n_estimators),
                    learning_rate=float(learning_rate),
                    max_depth=int(max_depth),
                    subsample=float(subsample),
                    colsample_bytree=float(colsample_bytree),
                    random_state=config.RANDOM_SEED,
                    eval_metric="mlogloss"
                    # 注意：這裡不設置 early_stopping_rounds
                )

                cv_scores = cross_val_score(
                    cv_model, features, y_encoded,
                    cv=cv_folds,
                    scoring='accuracy'
                )
                training_info['cv_scores'] = cv_scores.tolist()
                training_info['cv_mean'] = float(np.mean(cv_scores))
                training_info['cv_std'] = float(np.std(cv_scores))

            return df_aligned, training_info

        # 向後兼容：默認只返回 DataFrame
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