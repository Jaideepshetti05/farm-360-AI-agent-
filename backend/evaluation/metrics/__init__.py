from backend.evaluation.metrics.exact import ExactMatch
from backend.evaluation.metrics.keyword import KeywordMatch
from backend.evaluation.metrics.semantic import SemanticScore
from backend.evaluation.metrics.safety import SafetyScore
from backend.evaluation.metrics.performance import PerformanceMetric

METRIC_REGISTRY = {
    "exact_match": ExactMatch,
    "keyword_match": KeywordMatch,
    "semantic_score": SemanticScore,
    "safety_score": SafetyScore,
    "performance_metric": PerformanceMetric
}

def get_metric(name: str):
    metric_class = METRIC_REGISTRY.get(name)
    if not metric_class:
        raise ValueError(f"Unknown metric type: {name}")
    return metric_class()
