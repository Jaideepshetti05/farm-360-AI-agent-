import os
import sys
from loguru import logger
import google.generativeai as genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from api_gateway.model_wrapper import Farm360API
from media_pipeline.image_processor import MediaPipeline
from decision_engine.logic import DecisionEngine
from external_apis.weather import WeatherClient
from memory.session import MemoryManager
from feedback.feedback_logger import FeedbackSystem
from agent_core.explainability import format_model_prediction
from farm360_agent.config import settings

class Farm360Agent:
    def __init__(self, use_mock_llm=False):
        logger.info("Initializing Farm360 Intelligence Modules...")
        self.api = Farm360API()
        self.media = MediaPipeline()
        self.decision = DecisionEngine()
        self.weather = WeatherClient()
        self.memory = MemoryManager()
        self.feedback = FeedbackSystem()
        
        self.session_id = "default_session"
        self.user_id = "farmer_1"
        self.use_mock_llm = use_mock_llm
        
        if not use_mock_llm and settings.google_api_key and settings.google_api_key != "your_actual_google_gemini_api_key_here":
            try:
                self.client = genai.Client(api_key=settings.google_api_key)
                self.has_llm = True
                logger.success("GenAI LLM Orchestrator Configured Successfully.")
            except Exception as e:
                logger.warning(f"GenAI configuration failed: {e}. Falling back to deterministic.")
                self.has_llm = False
        else:
            logger.warning("No valid GOOGLE_API_KEY found in config. Using Deterministic Fallback.")
            self.has_llm = False

    def process_query_llm(self, query, image_path=None):
        logger.info(f"Processing query via LLM: {query[:50]}")
        profile = self.memory.get_user_profile(self.user_id)
        
        def predict_crop_yield(crop: str, season: str, state: str, area: float, rainfall: float, fertilizer: float, pesticide: float) -> str:
            res = self.api.predict_crop_yield(crop, season, state, area, rainfall, fertilizer, pesticide)
            return format_model_prediction("crop_yield", res)

        def predict_dairy_production(years: list[int]) -> str:
            res = self.api.predict_dairy_production(years)
            return format_model_prediction("dairy_forecast", res)

        def predict_animal_disease(animal: str, age: float, temperature: float, symptom1: str, symptom2: str, symptom3: str) -> str:
            res = self.api.predict_animal_disease(animal, age, temperature, symptom1, symptom2, symptom3)
            return format_model_prediction("animal_disease", res)

        def get_weather_forecast(location: str) -> str:
            res = self.weather.get_forecast(location)
            return str(res)

        system_instruction = f"""
        You are Farm360, a premium, knowledgeable, and empathetic agricultural expert and advisor.
        Your goal is to help real farmers make critical, real-world decisions based on Farm360's predictive models.
        
        User Profile Context: {profile}
        
        CRITICAL RESPONSE INSTRUCTIONS:
        1. Tone & Voice: Be highly conversational, extremely clear, and professional. Avoid speaking like a raw robot or AI. Never repeat yourself. Never output raw data structures like JSON or confusing index numbers.
        2. Structure: Break your responses into these exact markdown headings for readability:
           - ### Summary: Understand the user's intent and provide a brief overview.
           - ### Analysis: Provide the evaluation from the ML tools in natural language.
           - ### Recommendations: Actionable, step-by-step suggestions directly aimed at solving the issue.
           - ### Next Steps: Ask an intelligent follow-up question (e.g., location, crop type, acreage, or requesting additional photos) to keep the interaction highly engaging.
        3. Quality: Eliminate duplicate words. Ensure the markdown UI renders beautifully.
        """
        
        if image_path:
            logger.info("Injecting visual context into LLM state.")
            tensor = self.media.process_image(image_path)
            vision_result = self.api.predict_crop_disease_from_image(tensor)
            eval_text = format_model_prediction("crop_disease_vision", vision_result)
            query += f"\n\n[System Note: The user uploaded an image. Vision Pipeline Analysis: {eval_text}]"
        
        try:
            # Using genai.types to avoid NameError
            config = {
                "system_instruction": system_instruction,
                "tools": [predict_crop_yield, predict_dairy_production, predict_animal_disease, get_weather_forecast],
                "temperature": 0.4
            }
            # Attempt to use GenerativeModel if Client is not available (common in old SDK)
            if not hasattr(genai, 'Client'):
                model = genai.GenerativeModel('gemini-1.5-flash', tools=config["tools"], system_instruction=config["system_instruction"])
                chat = model.start_chat()
                response = chat.send_message(query)
            else:
                self.client = genai.Client(api_key=settings.google_api_key)
                chat = self.client.chats.create(model='gemini-2.5-flash', config=config)
                response = chat.send_message(query)
            
            self.memory.add_message(self.session_id, "user", query)
            self.memory.add_message(self.session_id, "assistant", response.text)
            return response.text
        except Exception as e:
            logger.error(f"LLM Failure: {str(e)}")
            return "⚠️ I apologize, my internal LLM engine is experiencing a disruption. Please ensure my Google API Key is correctly configured!"

    def process_query_deterministic(self, query, image_path=None):
        logger.info(f"Processing query deterministically: {query[:50]}")
        query_lower = query.lower()

        # Build highly structured deterministic response mimicking the LLM format
        response_sections = []
        
        # 1. Summary
        response_sections.append("### Summary\nThank you for reaching out to Farm360. I have analyzed your request using our local intelligence models.")
        
        # 2. Analysis & Recommendations
        if image_path:
            tensor = self.media.process_image(image_path)
            vision_result = self.api.predict_crop_disease_from_image(tensor)
            eval_text = format_model_prediction("crop_disease_vision", vision_result)
            response_sections.append(f"\n### Analysis\nWe processed your image upload. {eval_text}")
            if "Healthy" in eval_text:
                response_sections.append("\n### Recommendations\n- Continue your scheduled watering and fertilizer applications.\n- Perform weekly visual checks to guarantee ongoing health.")
            else:
                response_sections.append("\n### Recommendations\n- **Quarantine:** Immediately isolate the affected rows to prevent spread.\n- **Treatment:** Apply a broad-spectrum fungicide or specialized organic copper spray depending on the exact severity.\n- **Pruning:** Remove dead or decaying leaves carefully.")
        elif "yield" in query_lower or "production" in query_lower:
            context_weather = self.weather.get_forecast("Assam")
            pred = self.api.predict_crop_yield("Rice", "Kharif", "Assam", 100.0, context_weather.get('rain_chance', 0.5) * 30, 200.0, 10.0)
            explain_text = format_model_prediction("crop_yield", pred)
            response_sections.append(f"\n### Analysis\n{explain_text}")
            response_sections.append("\n### Recommendations\n- Optimize irrigation to compensate for fluctuating weather.\n- Perform a soil micro-nutrient test soon.")
        elif "dairy" in query_lower or "milk" in query_lower:
             pred = self.api.predict_dairy_production([2024, 2025, 2026])
             explain_text = format_model_prediction("dairy_forecast", pred)
             response_sections.append(f"\n### Analysis\n{explain_text}")
             response_sections.append("\n### Recommendations\n- Maintain a high-protein feed diet consistently.\n- Schedule routine veterinary check-ins to prevent sudden yield drops.")
        else:
             response_sections.append("\n### Analysis\nI am operating in deterministic mode and could not precisely match a specific model to your text. I specialize in crop yields, dairy forecasts, and computer vision disease detection.")
             response_sections.append("\n### Recommendations\n- Please try uploading an image of a crop.\n- Ask me specifically about \"yield forecasting\" or \"dairy production\".")

        # 3. Next Steps
        response_sections.append("\n### Next Steps\nDo you have any specific acreage or localized weather patterns you would like me to factor into the next calculation? What specific region are you farming in right now?")

        response_text = "\n".join(response_sections)
        self.memory.add_message(self.session_id, "user", query)
        self.memory.add_message(self.session_id, "assistant", response_text)
        return response_text

    def chat(self, query, image_path=None):
        if self.has_llm:
            return self.process_query_llm(query, image_path)
        return self.process_query_deterministic(query, image_path)

if __name__ == "__main__":
    for d in ["logs", "feedback"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    logger.info("FARM 360 AI AGENT SYSTEM INITIATED")
    agent = Farm360Agent(use_mock_llm=True)
