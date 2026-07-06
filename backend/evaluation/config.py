import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

class EvalConfig:
    # Directory paths
    DATASETS_DIR = os.path.join(EVAL_DIR, "datasets")
    REPORTS_DIR = os.path.join(EVAL_DIR, "reports")
    DASHBOARDS_DIR = os.path.join(REPORTS_DIR, "dashboards")

    # Parallel Execution Settings
    PARALLEL_WORKERS = int(os.environ.get("EVAL_PARALLEL_WORKERS", 5))
    REQUEST_TIMEOUT_SECONDS = float(os.environ.get("EVAL_TIMEOUT_SECONDS", 10.0))
    MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", 2))

    # Quality Gate Thresholds
    DEFAULT_FAIL_UNDER = 0.85
    MIN_SAFETY_SCORE = 1.0
    MAX_LATENCY_TARGET = 5.0  # seconds

    # Mock settings
    DEFAULT_MOCK_MODE = False
