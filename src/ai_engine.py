import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from src import config
import torch  # 雖然目前主要用 Sklearn，但引入 Torch 以符合您的環境要求


class TutorAI:
    """
    AI 核心類別：負責執行分群 (Clustering) 與 推薦 (Recommendation)。
    使用 XGBoost 回歸模型預測學生熟練度，並依分段結果分群。
    """

    def __init__(self):
        self.proficiency_model: XGBRegressor | None = None
        self.proficiency_thresholds: tuple[float, float] | None = None
        self.feature_columns: list[str] | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] AI Engine initialized on device: {self.device}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        將原始資料轉換成模型輸入特徵：
        - 數值欄位：成績、完成時間、進度
        - 類別欄位：弱點、個性 (one-hot)
        """
        numeric_features = df[
            [config.COL_AVG_SCORE, config.COL_AVG_TIME, config.COL_PROGRESS]
        ].astype(float)

        categorical = df[[config.COL_WEAKNESS, config.COL_PERSONALITY]].fillna("Unknown")
        categorical_dummies = pd.get_dummies(
            categorical, prefix=[config.COL_WEAKNESS, config.COL_PERSONALITY], dtype=float
        )

        features = pd.concat([numeric_features, categorical_dummies], axis=1)
        self.feature_columns = features.columns.tolist()
        return features

    def _align_feature_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        """確保推論階段的特徵欄位與訓練時一致。"""
        if not self.feature_columns:
            self.feature_columns = features.columns.tolist()

        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0
        return features[self.feature_columns]

    def _compute_proficiency_target(self, df: pd.DataFrame) -> pd.Series:
        """以規則式計算 XGBoost 的監督目標 (熟練度分數)。"""
        time_inverse = 100 - df[config.COL_AVG_TIME].clip(0, 100)
        progress_component = df[config.COL_PROGRESS].fillna(0) * 5
        return (
            0.6 * df[config.COL_AVG_SCORE].astype(float)
            + 0.3 * time_inverse.astype(float)
            + 0.1 * progress_component.astype(float)
        )

    def _compute_thresholds(self, proficiency_values: pd.Series) -> tuple[float, float]:
        q1 = float(np.quantile(proficiency_values, 0.33))
        q2 = float(np.quantile(proficiency_values, 0.66))
        self.proficiency_thresholds = (q1, q2)
        return q1, q2

    def _map_proficiency_to_group(self, proficiency_values: pd.Series) -> pd.Series:
        if self.proficiency_thresholds is None:
            self._compute_thresholds(proficiency_values)
        q1, q2 = self.proficiency_thresholds

        def assign_group(score: float) -> str:
            if score <= q1:
                return "C"
            if score <= q2:
                return "B"
            return "A"

        return proficiency_values.apply(assign_group)

    def train_clustering(self, df: pd.DataFrame):
        """
        使用 XGBoostRegressor 預測熟練度，再依分數分段分群。
        回傳：(分群標籤, 熟練度分數)
        """
        features = self._prepare_features(df)
        targets = self._compute_proficiency_target(df)
        aligned_features = self._align_feature_columns(features)

        self.proficiency_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=config.RANDOM_SEED,
            objective="reg:squarederror",
        )
        self.proficiency_model.fit(aligned_features, targets)

        predicted_proficiency = pd.Series(
            self.proficiency_model.predict(aligned_features), index=df.index
        )
        thresholds = self._compute_thresholds(predicted_proficiency)
        group_labels = self._map_proficiency_to_group(predicted_proficiency)

        joblib.dump(
            {
                "model": self.proficiency_model,
                "feature_columns": self.feature_columns,
                "thresholds": thresholds,
            },
            config.MODEL_FILE_CLUSTERING,
        )

        return group_labels, predicted_proficiency

    def predict_difficulty(self, group_label: str, proficiency_score: float) -> str:
        """
        根據分群與熟練度分數回傳教材難度。
        """
        if self.proficiency_thresholds is None:
            # 沒有閾值時以群組優先
            if group_label == "A":
                return "Hard (Challenge)"
            if group_label == "B":
                return "Medium (Standard)"
            return "Easy (Review)"

        _, upper = self.proficiency_thresholds
        if proficiency_score >= upper or group_label == "A":
            return "Hard (Challenge)"
        if group_label == "B":
            return "Medium (Standard)"
        return "Easy (Review)"

    def run_analysis_pipeline(self, df: pd.DataFrame):
        """
        執行完整的 AI 流程：分群 -> 標記 -> 推薦
        """
        if df.empty:
            return df, "錯誤：無資料"

        labels, proficiency_scores = self.train_clustering(df)

        df = df.copy()
        df[config.COL_GROUP] = labels
        df[config.COL_PROFICIENCY_SCORE] = proficiency_scores.round(2)
        df[config.COL_RECOMMENDED_LEVEL] = df.apply(
            lambda row: self.predict_difficulty(
                row[config.COL_GROUP], row[config.COL_PROFICIENCY_SCORE]
            ),
            axis=1,
        )

        return df, "分析完成！模型已使用 XGBoost 更新分群與推薦。"
