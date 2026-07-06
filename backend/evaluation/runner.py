import os
import sys
import json
import time
import argparse
import asyncio
from loguru import logger
import datetime
from typing import Tuple

# Root import injection
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.evaluation.config import EvalConfig
from backend.services.prompt_service import PromptService
from backend.evaluation.metrics import get_metric
from backend.evaluation.dashboard_builder import DashboardBuilder

class EvaluationRunner:
    def __init__(self, dataset_name: str, mock_mode: bool = False, provider: str = "gemini"):
        self.dataset_name = dataset_name
        self.mock_mode = mock_mode
        self.provider = provider
        self.dataset_file = os.path.join(EvalConfig.DATASETS_DIR, f"{dataset_name}_benchmark.json")

    def load_dataset(self) -> dict:
        """Loads and validates dataset scheme."""
        if not os.path.exists(self.dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "test_cases" not in data:
            raise ValueError(f"Invalid dataset schema: missing 'test_cases' in {self.dataset_name}")
        return data

    async def execute_mock_completion(self, prompt: str) -> str:
        """Simulates completion text locally for testing framework runs."""
        await asyncio.sleep(0.1)  # simulate fast latency
        q = prompt.lower()
        if "crop" in q or "rice" in q or "assam" in q:
            return "Kharif season rice requires basal NPK fertilizer before sowing. Actions checklist: 1. NPK basal application, 2. Rice sowing."
        if "dairy" in q:
            return "Nutrition management includes bypass protein and deworming cattle along with compound fodder feed."
        return "Generic mock explanation advisory."

    async def run_single_case(self, case: dict) -> dict:
        """Executes a single benchmark prompt run."""
        inputs = case.get("inputs", {})
        query = inputs.get("query", "")
        
        # 1. Compile prompt using PromptService
        start_time = time.time()
        try:
            category = case.get("category", "")
            if category.startswith("vision_"):
                prompt, config = PromptService.render_and_validate(
                    category,
                    {
                        "predictions": inputs.get("predictions", ""),
                        "profile": inputs.get("profile", ""),
                        "language": inputs.get("language", "English")
                    }
                )
            else:
                prompt, config = PromptService.render_and_validate(
                    "general_assistant",
                    {
                        "profile": json.dumps(inputs.get("user_profile", {}), ensure_ascii=False),
                        "ml_context": inputs.get("ml_context", "")
                    }
                )
            placeholder_score = 1.0
        except Exception as e:
            logger.error(f"Prompt compilation failed for case {case.get('id')}: {e}")
            prompt = f"Error: {e}"
            placeholder_score = 0.0
            
        # 2. Query provider / mock completion
        response = ""
        latency = 0.0
        if placeholder_score > 0.0:
            try:
                if self.mock_mode:
                    response = await self.execute_mock_completion(prompt)
                else:
                    from backend.provider_manager import provider_manager
                    messages = [
                        {"role": "system", "content": "You are Farm360 AI expert agricultural extension officer."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response_chunks = []
                    for token in provider_manager.stream_completion(messages):
                        response_chunks.append(token)
                    response = "".join(response_chunks)
            except Exception as le:
                logger.error(f"LLM query failed: {le}")
                response = f"⚠️ Connection error: {le}"
                
            latency = time.time() - start_time
            
        case["actual_latency"] = latency
        
        # 3. Calculate metrics
        scores = {}
        enabled_metrics = ["keyword_match", "semantic_score", "safety_score", "performance_metric"]
        for metric_name in enabled_metrics:
            try:
                m_eval = get_metric(metric_name)
                score = await m_eval.evaluate(response, case)
                scores[metric_name] = score
            except Exception as me:
                logger.warning(f"Metric {metric_name} failed: {me}")
                scores[metric_name] = 0.0
                
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        rules = case.get("evaluation_rules", {})
        target = rules.get("fail_under", EvalConfig.DEFAULT_FAIL_UNDER)
        status = "pass" if avg_score >= target else "fail"
        
        return {
            "id": case.get("id"),
            "category": case.get("category"),
            "difficulty": case.get("difficulty"),
            "latency": latency,
            "score": avg_score,
            "status": status,
            "metrics": scores,
            "response": response
        }

    async def execute_evaluation(self, parallel_workers: int = 5) -> Tuple[dict, list]:
        """Runs parallel evaluation tasks using worker queues."""
        dataset_data = self.load_dataset()
        test_cases = dataset_data.get("test_cases", [])
        
        sem = asyncio.Semaphore(parallel_workers)
        
        async def worker(case):
            async with sem:
                return await self.run_single_case(case)
                
        tasks = [worker(case) for case in test_cases]
        results = await asyncio.gather(*tasks)
        
        overall_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        
        metadata = {
            "dataset": self.dataset_name,
            "dataset_version": dataset_data.get("version", "1.0.0"),
            "provider": self.provider,
            "mock_mode": self.mock_mode,
            "overall_score": overall_score,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return metadata, results

def main():
    parser = argparse.ArgumentParser(description="Farm360 AI - Evaluation Pipeline CLI")
    parser.add_argument("--dataset", type=str, default="general", help="Dataset name prefix")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM Provider preference")
    parser.add_argument("--mock", action="store_true", help="Enable mock runner completion mode")
    parser.add_argument("--parallel", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--fail-under", type=float, default=0.85, help="Fail-under gate score threshold")
    
    args = parser.parse_args()
    
    runner = EvaluationRunner(
        dataset_name=args.dataset,
        mock_mode=args.mock,
        provider=args.provider
    )
    
    logger.info(f"Starting Evaluation on dataset '{args.dataset}' (Mock mode: {args.mock})...")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        metadata, results = loop.run_until_complete(runner.execute_evaluation(args.parallel))
    finally:
        loop.close()
        
    metadata["target_score"] = args.fail_under
    
    # 1. Save JSON Report
    os.makedirs(EvalConfig.REPORTS_DIR, exist_ok=True)
    report_json_path = os.path.join(EvalConfig.REPORTS_DIR, f"eval_report_{args.dataset}.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=4)
    logger.success(f"[Report] JSON details written to {report_json_path}")
    
    # 2. Build HTML Dashboard
    dashboard_path = DashboardBuilder.build_dashboard(metadata, results)
    logger.success(f"[Report] HTML dashboard generated at {dashboard_path}")
    
    # 3. Check Quality Gate thresholds
    overall = metadata.get("overall_score", 0.0)
    logger.info(f"Average Evaluation Score: {overall * 100:.2f}% (Target threshold: {args.fail_under * 100:.2f}%)")
    
    if overall < args.fail_under:
        logger.error("[Gate Check] FAILURE: Target quality threshold violated.")
        sys.exit(1)
        
    logger.success("[Gate Check] PASS: Quality standards satisfied.")
    sys.exit(0)

if __name__ == "__main__":
    main()
