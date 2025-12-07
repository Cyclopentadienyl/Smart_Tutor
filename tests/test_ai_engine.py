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
        self.original_model_file = config.MODEL_FILE_CLUSTERING
        config.MODEL_FILE_CLUSTERING = os.path.join(self.temp_dir.name, "temp_model.pkl")

        self.sample_df = pd.DataFrame(
            {
                config.COL_STUDENT_ID: ["S001", "S002", "S003", "S004", "S005"],
                config.COL_AVG_SCORE: [90, 75, 60, 82, 55],
                config.COL_AVG_TIME: [15, 25, 45, 20, 50],
                config.COL_PROGRESS: [8, 6, 4, 7, 3],
                config.COL_WEAKNESS: ["Algebra", "Geometry", "Statistics", "Algebra", "Calculus"],
                config.COL_PERSONALITY: ["Proactive", "Passive", "Anxious", "Steady", "Passive"],
            }
        )

    def tearDown(self):
        config.MODEL_FILE_CLUSTERING = self.original_model_file
        self.temp_dir.cleanup()

    def test_train_clustering_generates_groups(self):
        ai = TutorAI()
        labels, proficiency = ai.train_clustering(self.sample_df.copy())

        self.assertEqual(len(labels), len(self.sample_df))
        self.assertEqual(len(proficiency), len(self.sample_df))
        self.assertTrue(set(labels).issubset({"A", "B", "C"}))
        self.assertIsNotNone(ai.proficiency_thresholds)

    def test_run_analysis_pipeline_adds_columns(self):
        ai = TutorAI()
        processed_df, message = ai.run_analysis_pipeline(self.sample_df.copy())

        self.assertIn(config.COL_GROUP, processed_df.columns)
        self.assertIn(config.COL_RECOMMENDED_LEVEL, processed_df.columns)
        self.assertIn(config.COL_PROFICIENCY_SCORE, processed_df.columns)
        self.assertTrue(message.startswith("分析完成"))


if __name__ == "__main__":
    unittest.main()
