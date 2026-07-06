import sys
import os
import asyncio
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.evaluation.runner import EvaluationRunner
from backend.evaluation.config import EvalConfig

class TestEvaluationFramework(unittest.IsolatedAsyncioTestCase):
    async def test_general_benchmark_mock_run(self):
        # 1. Run evaluation against general_benchmark in mock mode
        runner = EvaluationRunner(dataset_name="general", mock_mode=True)
        metadata, results = await runner.execute_evaluation(parallel_workers=2)
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.get("dataset"), "general")
        self.assertTrue(len(results) > 0)
        
        # Verify scores are populated
        first_res = results[0]
        self.assertIn("score", first_res)
        self.assertIn("latency", first_res)
        self.assertIn("metrics", first_res)
        self.assertEqual(first_res.get("status"), "pass")

    async def test_security_benchmark_mock_run(self):
        # 2. Run against security benchmark
        runner = EvaluationRunner(dataset_name="security", mock_mode=True)
        metadata, results = await runner.execute_evaluation(parallel_workers=1)
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.get("dataset"), "security")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["metrics"].get("safety_score"), 1.0)

    def test_dashboard_compilation(self):
        # 3. Test HTML dashboard compiler writes output successfully
        from backend.evaluation.dashboard_builder import DashboardBuilder
        mock_metadata = {
            "dataset": "test",
            "overall_score": 0.95,
            "target_score": 0.85
        }
        mock_results = [
            {"id": "test_001", "category": "crop", "difficulty": "easy", "latency": 0.2, "score": 1.0, "status": "pass"}
        ]
        dest_path = DashboardBuilder.build_dashboard(mock_metadata, mock_results)
        self.assertTrue(os.path.exists(dest_path))

if __name__ == "__main__":
    unittest.main()
