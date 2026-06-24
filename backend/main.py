# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from loguru import logger
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from api_gateway.model_wrapper import Farm360API
from media_pipeline.image_processor import MediaPipeline
from decision_engine.logic import DecisionEngine
from external_apis.weather import WeatherClient
from memory.session import MemoryManager
from feedback.feedback_logger import FeedbackSystem
from agent_core.explainability import format_model_prediction
from backend.config import settings


# ---------------------------------------------------------------------------
# System prompt  (prose + markdown -- NO JSON output)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """You are Farm360 AI, an expert agricultural advisor specializing in Indian farming operations.
You combine deep agronomic knowledge with practical field expertise in agronomy, horticulture, soil science,
plant pathology, irrigation engineering, dairy farming, and agricultural economics.

FARMER PROFILE:
{profile}

{ml_context}

RESPONSE RULES:
1. Respond in clear, natural, flowing PROSE -- never output raw JSON unless the user specifically asks for it.
2. Use markdown to organize your response:
   - **Bold** for key terms, quantities, and important values
   - Bullet lists for options or tips
   - Numbered lists for step-by-step procedures
   - ## Headers to separate major sections in longer answers
   - Tables (markdown) for comparing varieties, costs, schedules
3. Be SPECIFIC: include exact crop variety names, dosages (kg/ha, ml/litre), timing (days after sowing), and costs (INR).
4. Be PRACTICAL and immediately actionable for an Indian farmer.
5. NEVER ask clarifying questions -- provide the most complete and helpful answer right away.
6. Keep a warm, expert, conversational tone -- like a trusted agricultural extension officer talking to a farmer.
"""


class Farm360Agent:
    def __init__(self, use_mock_llm: bool = False, model_base_path: str = None):
        logger.info("Initializing Farm360 Intelligence Modules...")
        self.api      = Farm360API(model_base_path=model_base_path)
        self.media    = MediaPipeline()
        self.decision = DecisionEngine()
        self.weather  = WeatherClient()
        self.memory   = MemoryManager()
        self.feedback = FeedbackSystem()

        self.session_id   = "default_session"
        self.user_id      = "farmer_1"
        self.use_mock_llm = use_mock_llm
        self.has_llm      = False
        self.client       = None

        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM client using Gemini via OpenAI SDK."""
        if self.use_mock_llm:
            logger.info("Mock LLM mode — using deterministic fallback responses.")
            return

        if not settings.google_api_key:
            logger.info("No GOOGLE_API_KEY — using fallback responses.")
            return

        try:
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=settings.google_api_key,
                timeout=15.0,
            )
            self.has_llm = True
            logger.success("Gemini LLM configured successfully.")
        except Exception as e:
            logger.warning(f"Gemini setup failed: {e}")

    # -----------------------------------------------------------------------
    # ML context builder
    # -----------------------------------------------------------------------
    def _collect_ml_context(self, query: str, image_path: str = None) -> str:
        parts = []
        q = query.lower()

        if any(k in q for k in ["yield", "crop", "production", "harvest", "plant", "grow", "season"]):
            try:
                pred = self.api.predict_crop_yield("Rice", "Kharif", "Assam", 100.0, 150.0, 200.0, 10.0)
                parts.append(f"Crop yield model: {format_model_prediction('crop_yield', pred)}")
            except Exception as e:
                logger.warning(f"Crop model skipped: {e}")

        if any(k in q for k in ["dairy", "milk", "cattle", "cow", "buffalo", "livestock"]):
            try:
                pred = self.api.predict_dairy_production([2024, 2025, 2026])
                parts.append(f"Dairy forecast: {format_model_prediction('dairy_forecast', pred)}")
            except Exception as e:
                logger.warning(f"Dairy model skipped: {e}")

        if image_path:
            try:
                tensor = self.media.process_image(image_path)
                vision = self.api.predict_crop_disease_from_image(tensor)
                parts.append(f"Vision analysis of uploaded image: {format_model_prediction('crop_disease_vision', vision)}")
            except Exception as e:
                logger.warning(f"Vision model skipped: {e}")

        if parts:
            return "RELEVANT ML MODEL DATA (incorporate into your answer):\n" + "\n".join(f"- {p}" for p in parts)
        return ""

    # -----------------------------------------------------------------------
    # Message builder
    # -----------------------------------------------------------------------
    def _build_messages(self, query: str, image_path: str = None, model: str = None) -> list:
        profile    = self.memory.get_user_profile(self.user_id)
        ml_context = self._collect_ml_context(query, image_path)

        system = SYSTEM_PROMPT_TEMPLATE.format(
            profile=json.dumps(profile, ensure_ascii=False),
            ml_context=ml_context,
        )

        history  = self.memory.get_chat_history(self.session_id)
        messages = [{"role": "system", "content": system}]
        for m in history[-6:]:
            messages.append({
                "role": "user" if m["role"] == "user" else "assistant",
                "content": m["content"],
            })
        messages.append({"role": "user", "content": query})
        return messages

    # -----------------------------------------------------------------------
    # STREAMING prose (generator -- yields text tokens)
    # -----------------------------------------------------------------------
    def stream_query_prose(self, query: str, image_path: str = None, model: str = None):
        """
        Generator that yields text tokens one by one from the LLM stream.
        This powers the real-time ChatGPT-style token streaming.
        """
        target_model = "gemini-1.5-flash"  # Override frontend model to Gemini
        messages     = self._build_messages(query, image_path, target_model)

        logger.info(f"[STREAM] model={target_model} query={query[:60]!r}")

        full_response = ""
        try:
            stream = self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
                extra_headers={
                    "HTTP-Referer": "https://farm360.app",
                    "X-Title": "Farm360 AI",
                },
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield delta

        except Exception as e:
            logger.exception(f"LLM streaming error: {e}")
            error_message = f"⚠️ **LLM API Request Failed**\n\nThe AI provider returned an error: `{str(e)}`\n\nPlease check your API keys or rate limits."
            yield error_message
            full_response = error_message
        finally:
            if full_response:
                self.memory.add_message(self.session_id, "user", query)
                self.memory.add_message(self.session_id, "assistant", full_response)

    # -----------------------------------------------------------------------
    # Blocking chat (for /chat endpoint)
    # -----------------------------------------------------------------------
    def chat_blocking(self, query: str, image_path: str = None, model: str = None) -> str:
        if not self.has_llm:
            return "⚠️ LLM is not configured. Please add a valid API key."
        return "".join(self.stream_query_prose(query, image_path, model))

    # -----------------------------------------------------------------------
    # Rich fallback (when no LLM is available)
    # -----------------------------------------------------------------------
    def _fallback_prose(self, query: str) -> str:
        q = query.lower()
        parts = [
            "**Farm360 AI** -- offline advisory mode (LLM unavailable, showing knowledge-base response)\n",
            "---\n",
        ]

        if any(k in q for k in ["yield", "crop", "plant", "grow", "season", "june", "july", "kharif"]):
            parts.append(
                "## Recommended Crops for the Kharif Season\n\n"
                "Based on typical Indian agroclimatic conditions, here are the best options:\n\n"
                "| Crop | Best Variety | Avg Yield | Duration |\n"
                "|------|-------------|-----------|----------|\n"
                "| Rice | IR-64, Basmati 370 | 5-7 t/ha | 130-140 days |\n"
                "| Maize | DKC-9144 | 8-10 t/ha | 90-100 days |\n"
                "| Soybean | JS-335 | 2.5-3 t/ha | 95-100 days |\n"
                "| Cotton | Bt-Bollgard II | 20-25 q/ha | 180-200 days |\n\n"
                "## Key Pre-Sowing Actions\n\n"
                "1. **Soil test** -- send soil to the nearest KVK lab for NPK + micronutrient analysis\n"
                "2. **Seed treatment** -- treat seeds with Thiram 75 WS @ 3 g/kg seed before sowing\n"
                "3. **Basal fertilizer** -- apply FYM at 10 t/ha two weeks before transplanting\n"
                "4. **Check local mandi prices** -- align variety selection to current market demand\n"
            )
        elif any(k in q for k in ["disease", "yellow", "spot", "blight", "fungal", "pest", "insect", "brown", "wilt"]):
            parts.append(
                "## Common Causes of Leaf Yellowing and Spotting\n\n"
                "**Nutritional deficiencies** (most common):\n"
                "- **Nitrogen deficiency** -- uniform yellowing starting at older leaves; apply Urea 46% @ 26 kg/acre as top-dressing\n"
                "- **Iron deficiency** -- interveinal chlorosis (yellow between green veins); foliar spray FeSO4 @ 5 g/L, 2-3 times\n"
                "- **Magnesium** -- similar to iron but lower leaves first; apply MgSO4 @ 500 g/100L water\n\n"
                "**Fungal diseases**:\n"
                "- **Leaf blast** (rice) -- diamond-shaped lesions with grey centre; spray Tricyclazole 75WP @ 6 g/10L\n"
                "- **Brown spot** -- oval brown lesions; apply Mancozeb 75WP @ 20 g/10L\n\n"
                "## Recommended Diagnostic Steps\n\n"
                "1. Upload a clear close-up photo of the affected leaf using the paperclip button\n"
                "2. Check soil pH (target: 6.0-7.0 for most crops)\n"
                "3. Isolate a 5-plant sample and monitor for 48 hours to track spread direction\n"
            )
        elif any(k in q for k in ["dairy", "milk", "cattle", "cow", "buffalo", "livestock"]):
            parts.append(
                "## Increasing Dairy Cattle Milk Yield\n\n"
                "**Nutrition (highest impact)**:\n"
                "- Feed balanced compound cattle feed at **1 kg per 2.5 L milk produced**\n"
                "- Provide **30-35 kg green fodder** per day (Napier grass, berseem, sorghum)\n"
                "- Supplement with bypass protein (soybean meal @ 200-300 g/day per cow)\n\n"
                "**Breed improvement**:\n"
                "- Cross-breed desi cows with **HF or Jersey** to increase potential to 15-20 L/day\n"
                "- Use **AI (artificial insemination)** from proven bulls via BAIF or state livestock dept\n\n"
                "**Management**:\n"
                "- Maintain strict 12-hour milking intervals\n"
                "- Complete milking within 7 minutes per cow (prevents residual milk)\n"
                "- Quarterly deworming; annual FMD + BQ + HS vaccination\n\n"
                "> **Expected improvement**: Proper nutrition alone typically improves yield by 15-25% within 60 days.\n"
            )
        else:
            parts.append(
                "## Agricultural Best Practices for Indian Farming\n\n"
                "**Soil Health**\n"
                "- Test NPK and micronutrients every 2 years at your nearest KVK (Krishi Vigyan Kendra)\n"
                "- Target organic matter > 0.8%; add FYM or vermicompost annually\n"
                "- Maintain pH between 6.0-7.0 for most crops\n\n"
                "**Water Management**\n"
                "- Adopt drip or sprinkler irrigation to reduce water consumption by 30-40%\n"
                "- Apply mulching to reduce soil moisture evaporation by up to 25%\n\n"
                "**Integrated Pest Management**\n"
                "- Use pheromone traps for early pest monitoring\n"
                "- Prefer neem-based biopesticides (Azadirachtin 0.03% EC) before chemical intervention\n\n"
                "**Market Access**\n"
                "- Register on **e-NAM portal** for price discovery across 1,000+ mandis\n"
                "- Join an FPO (Farmer Producer Organisation) for collective bargaining power\n\n"
                "> To get AI-powered personalized advice, ensure your OpenRouter API key is active in the `.env` file.\n"
            )

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # Legacy compatibility
    # -----------------------------------------------------------------------
    def chat(self, query: str, image_path: str = None) -> dict:
        """Legacy JSON interface (kept for backward compatibility)."""
        text = self.chat_blocking(query, image_path)
        return {
            "summary": "Farm360 AI Response",
            "analysis": text,
            "recommendations": [],
            "action_steps": [],
            "missing_data_warning": None,
        }

    def process_query_deterministic(self, query, image_path=None):
        return self.chat(query, image_path)


if __name__ == "__main__":
    for d in ["logs", "feedback"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    logger.info("FARM360 AI -- TEST MODE")
    agent = Farm360Agent(use_mock_llm=False)
    print("Streaming test response:")
    for token in agent.stream_query_prose("What crops should I plant in June in Punjab, India?"):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()