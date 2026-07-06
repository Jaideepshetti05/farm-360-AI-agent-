import time
import json
from typing import Dict, Any
from loguru import logger
from backend.router.interfaces import AdvisorInterface
from backend.router.advisor_result import AdvisorResult
from backend.services.prompt_service import PromptService
from backend.services.context_builder import PromptContextService
from backend.provider_manager import provider_manager

class BaseAdvisor(AdvisorInterface):
    def __init__(self, name: str, prompt_key: str):
        self.name = name
        self.prompt_key = prompt_key

    def metadata(self) -> dict:
        return {
            "advisor_name": self.name,
            "prompt_key": self.prompt_key,
            "version": "1.0.0"
        }

    async def execute(self, query: str, context: dict) -> AdvisorResult:
        start_time = time.time()
        
        user_profile = context.get("user_profile", {})
        ml_context = context.get("ml_context", "")
        recent_history = context.get("recent_history", [])
        summary_text = context.get("summary_text")
        
        # Check if advisor uses RAG
        rag_context = ""
        if self.name in ["CropAdvisor", "DiseaseAdvisor", "AnimalAdvisor", "WeatherAdvisor", "MarketAdvisor"]:
            try:
                from backend.rag import RAGService
                rag_service = RAGService()
                rag_context = await rag_service.get_context(query)
            except Exception as re:
                logger.warning(f"[{self.name}] RAG lookup skipped: {re}")
                
        if rag_context:
            ml_context = f"{ml_context}\n\n{rag_context}".strip()

        # Build message history using dynamic context compilation
        messages = PromptContextService.build_prompt_context(
            system_prompt_template=self.prompt_key,
            user_profile=user_profile,
            ml_context=ml_context,
            recent_history=recent_history,
            summary_text=summary_text,
            max_context_tokens=4096
        )
        
        messages.append({"role": "user", "content": query})
        
        response_chunks = []
        try:
            for token in provider_manager.stream_completion(messages):
                response_chunks.append(token)
            response = "".join(response_chunks)
        except Exception as e:
            logger.error(f"[{self.name}] LLM execution failed: {e}")
            response = f"⚠️ Connection error: {e}"
            
        elapsed = time.time() - start_time
        
        return AdvisorResult(
            response_text=response,
            confidence=1.0,
            advisor_name=self.name,
            execution_time=elapsed,
            metadata=self.metadata()
        )

class GeneralAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("GeneralAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        return 0.25  # baseline fit

class CropAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("CropAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["crop", "yield", "sow", "fertiliz", "seed", "plant", "grow"]):
            return 0.85
        return 0.0

class DiseaseAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("DiseaseAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["disease", "leaf", "spot", "blast", "mold", "yellow", "insect", "pest"]):
            return 0.90
        return 0.0

class AnimalAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("AnimalAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["cow", "cattle", "buffalo", "animal", "veterinary", "livestock", "sheep", "goat"]):
            return 0.85
        return 0.0

class DairyAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("DairyAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["milk", "dairy", "milking", "fat", "snf", "collection"]):
            return 0.85
        return 0.0

class WeatherAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("WeatherAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["weather", "rain", "monsoon", "forecast", "temp"]):
            return 0.85
        return 0.0

class MarketAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("MarketAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["price", "cost", "market", "mandi", "rate", "inr"]):
            return 0.85
        return 0.0

class VisionAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("VisionAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        if context.get("image_path"):
            return 0.95
        return 0.0

class PredictionAdvisor(BaseAdvisor):
    def __init__(self):
        super().__init__("PredictionAdvisor", "general_assistant")

    async def evaluate_fit(self, query: str, context: dict) -> float:
        q = query.lower()
        if any(k in q for k in ["forecast", "trend", "yield prediction"]):
            return 0.85
        return 0.0
