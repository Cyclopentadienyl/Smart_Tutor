import importlib.util
import os
import tempfile
import unittest

import pandas as pd

from src import config

XGBOOST_AVAILABLE = importlib.util.find_spec("xgboost") is not None

if XGBOOST_AVAILABLE:
    from src.ai_engine import TutorAI


@unittest.skipUnless(XGBOOST_AVAILABLE, "xgboost is required for TutorAI tests")
class TutorAITestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_model_dir = config.MODEL_DIR
        self.original_model_file = config.MODEL_FILE_PREDICTION

        config.MODEL_DIR = self.temp_dir.name
        config.MODEL_FILE_PREDICTION = os.path.join(self.temp_dir.name, "predictor_model.pkl")
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        self.ai = TutorAI()
        self.sample_df = pd.DataFrame(
            {
                config.COL_STUDENT_ID: ["S001", "S002", "S003", "S004", "S005"],
                config.COL_NAME: ["A", "B", "C", "D", "E"],
                config.COL_ACCURACY: [0.9, 0.75, 0.55, 0.82, 0.6],
                config.COL_AVG_TIME: [10.0, 20.0, 35.0, 22.0, 30.0],
                config.COL_LEARNING_PACE: [12.0, 11.0, 8.0, 10.0, 9.0],
                config.COL_ATTENDANCE: [0.95, 0.8, 0.7, 0.88, 0.75],
                config.COL_HW_COMPLETION: [0.9, 0.85, 0.6, 0.8, 0.65],
                config.COL_SCORE_HISTORY: [str([80, 85, 90]), str([70, 75, 78]), str([60, 62, 65]), str([82, 83, 81]), str([68, 70, 72])],
                config.COL_ERROR_TYPES: [
                    str({"reading": 1, "vocab": 0, "logic": 1}),
                    str({"reading": 2, "vocab": 1, "logic": 1}),
                    str({"reading": 3, "vocab": 2, "logic": 2}),
                    str({"reading": 1, "vocab": 1, "logic": 1}),
                    str({"reading": 2, "vocab": 2, "logic": 1}),
                ],
            }
        )

    def tearDown(self):
        config.MODEL_DIR = self.original_model_dir
        config.MODEL_FILE_PREDICTION = self.original_model_file
        self.temp_dir.cleanup()

    def test_run_analysis_pipeline_adds_predictions(self):
        processed_df, log = self.ai.run_analysis_pipeline(self.sample_df)

        self.assertIn(config.COL_RECOMMENDED_LEVEL, processed_df.columns)
        self.assertIn(config.COL_GROUP, processed_df.columns)
        self.assertGreater(processed_df[config.COL_RECOMMENDED_LEVEL].notna().sum(), 0)
        self.assertIn("推論完成", log)

    def test_predict_single_returns_known_label(self):
        trained_df = self.ai.train_prediction_model(self.sample_df)
        self.assertIn(config.COL_RECOMMENDED_LEVEL, trained_df.columns)

        prediction = self.ai.predict_single(0.8, 15.0, 11.0, 0.9, 0.85, 1, 1, 1, 85)
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, set(self.ai.label_encoder.classes_))


if __name__ == "__main__":
    unittest.main()