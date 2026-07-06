import time
import asyncio
from typing import Dict, Any, Tuple
from loguru import logger
from backend.router.registry import AdvisorRegistry
from backend.router.advisor_result import AdvisorResult

class IntentRouter:
    def __init__(self):
        self.registry = AdvisorRegistry()

    async def route(self, query: str, context: dict) -> AdvisorResult:
        """
        Concurrently evaluates query fit across all registered advisors,
        selects the best advisor, executes its logic, and returns the result.
        """
        start_routing = time.time()
        advisors = self.registry.get_advisors()
        
        # Concurrently evaluate fit scores
        async def eval_advisor(adv):
            try:
                score = await adv.evaluate_fit(query, context)
                return adv, score
            except Exception as e:
                logger.error(f"[Router] Error evaluating fit for {adv.name}: {e}")
                return adv, 0.0

        tasks = [eval_advisor(adv) for adv in advisors]
        eval_results = await asyncio.gather(*tasks)
        
        # Find best advisor
        best_adv = None
        best_score = -1.0
        
        for adv, score in eval_results:
            if score > best_score:
                best_score = score
                best_adv = adv
                
        # Fallback to GeneralAdvisor if fit score is too low
        if best_score < 0.30:
            logger.info(f"[Router] Best fit score ({best_score:.2f}) too low. Falling back to GeneralAdvisor.")
            best_adv = self.registry.get_by_name("GeneralAdvisor")
            best_score = 0.25
            
        routing_time_ms = (time.time() - start_routing) * 1000.0
        logger.info(f"[Router] Routed query to {best_adv.name} (score: {best_score:.2f}, decision time: {routing_time_ms:.2f}ms)")
        
        # Execute selected advisor
        result = await best_adv.execute(query, context)
        
        # Inject routing performance metadata
        result.metadata["routing_decision_time_ms"] = routing_time_ms
        result.metadata["router_fit_score"] = best_score
        
        return result
