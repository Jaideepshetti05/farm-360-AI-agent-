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
        
        # Debug: Log API key status (without exposing the key)
        api_key_status = "SET" if settings.google_api_key else "NOT SET"
        logger.debug(f"GOOGLE_API_KEY environment variable: {api_key_status}")
        print(f"[DEBUG] GOOGLE_API_KEY loaded from env: {api_key_status}")
        
        # Initialize GenAI LLM if API key is available and not in mock mode
        if not use_mock_llm and settings.google_api_key and "your_actual" not in settings.google_api_key:
            try:
                genai.configure(api_key=settings.google_api_key)
                self.has_llm = True
                logger.success("GenAI LLM Orchestrator Configured Successfully.")
                print("[DEBUG] ✅ Real AI responses ENABLED - Using Gemini model")
            except Exception as e:
                logger.warning(f"GenAI configuration failed: {e}. Falling back to deterministic.")
                self.has_llm = False
                print(f"[DEBUG] ❌ GenAI setup failed: {e}")
        else:
            if use_mock_llm:
                logger.info("Mock LLM mode enabled - using deterministic responses")
                print("[DEBUG] ℹ️ Mock LLM mode enabled")
            else:
                logger.warning("No valid GOOGLE_API_KEY found or placeholder used. Using Deterministic Fallback.")
                print("[DEBUG] ❌ No valid GOOGLE_API_KEY found - using fallback")
            self.has_llm = False

    def process_query_llm(self, query, image_path=None):
        logger.info(f"Processing query via LLM: {query[:50]}")
        profile = self.memory.get_user_profile(self.user_id)
        
        # Tool definitions for Gemini
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

        # 🎯 STRONG EXPERT PROMPT - PRODUCTION READY
        system_instruction = f"""
You are the Farm360 AI Agricultural Expert, a senior agricultural consultant and data scientist. 
Your objective is to provide professional, actionable, and data-driven advice to farmers in India.

USER CONTEXT:
- Role: Farmer
- Domain: Indian Agriculture (Kharif, Rabi, Zaid cycles)
- Current Profile: {profile}

CORE PRINCIPLES:
1. NO CHATBOT BEHAVIOR: Do not say "Hello", "How can I help", "Are you looking for", or "Would you like". 
2. DIRECTNESS: Provide immediate, direct answers. If the user asks about a disease, diagnose it. If they ask about crops, suggest them.
3. STRUCTURED LOGIC: Use the Farm360 tools (yield prediction, disease diagnosis, weather) to back your claims.
4. EXPERT TONE: Sound like a professional agronomist. Use practical agricultural terminology (e.g., NPK levels, irrigation cycles, pest thresholds).

REQUIRED OUTPUT FORMAT (JSON ONLY):
You MUST return ONLY a valid JSON object with these fields:
- "summary": A concise (1 sentence) professional overview of the solution.
- "analysis": A deep-dive professional assessment (2-4 sentences) integrating tool outputs and agricultural science.
- "recommendations": A list of 3-5 specific, technical actions for the farmer.
- "crop_suggestions": A list of 2-3 specific crops (with varieties if possible) suitable for the context.
- "next_steps": A list of 3 concrete, immediate actions to take on the farm.

DO NOT output any text before or after the JSON.
"""
        
        if image_path:
            logger.info("Injecting visual context into LLM state.")
            tensor = self.media.process_image(image_path)
            vision_result = self.api.predict_crop_disease_from_image(tensor)
            eval_text = format_model_prediction("crop_disease_vision", vision_result)
            query += f"\n\n[System Note: VISUAL DATA DETECTED. Vision Model Diagnosis: {eval_text}. Integrate this into your structured analysis.]"
        
        try:
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                tools=[predict_crop_yield, predict_dairy_production, predict_animal_disease, get_weather_forecast],
                system_instruction=system_instruction
            )
            
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(query, generation_config=generation_config)
            
            raw_text = response.text.strip()
            # Handle potential markdown code blocks if the model ignores response_mime_type
            if raw_text.startswith("```"):
                import re
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
                if match:
                    raw_text = match.group(1)

            import json
            try:
                structured_data = json.loads(raw_text)
                self.memory.add_message(self.session_id, "user", query)
                self.memory.add_message(self.session_id, "assistant", raw_text)
                return structured_data
            except Exception:
                logger.warning("LLM produced invalid JSON, returning raw text as summary.")
                return {
                    "summary": "Direct Agricultural Insight",
                    "analysis": raw_text[:500],
                    "recommendations": ["Review current crop status", "Consult local expert"],
                    "crop_suggestions": ["Rice", "Mustard"],
                    "next_steps": ["Check soil moisture", "Verify input details"]
                }

        except Exception as e:
            logger.error(f"LLM Failure: {str(e)}")
            return self.process_query_deterministic(query, image_path)

    def _generate_smart_fallback_response(self, query, image_path=None):
        """Structured fallback logic when LLM is unavailable."""
        query_lower = query.lower()
        context = self._analyze_agricultural_query(query_lower)
        
        res = {
            "summary": context.get('summary', "Agricultural analysis based on local diagnostic models."),
            "analysis": f"Currently operating in high-performance deterministic mode. Analysis indicates focus on {context.get('crop', 'general agriculture')} in {context.get('location', 'India')}.",
            "recommendations": [],
            "crop_suggestions": [],
            "next_steps": []
        }

        if image_path:
            try:
                tensor = self.media.process_image(image_path)
                vision_result = self.api.predict_crop_disease_from_image(tensor)
                eval_text = format_model_prediction("crop_disease_vision", vision_result)
                res["analysis"] += f" Visual analysis results: {eval_text}."
                
                if "Healthy" in eval_text:
                    res["recommendations"] = ["Maintain watering schedule", "Monitor for pests weekly", "Continue NPK balance"]
                else:
                    res["recommendations"] = ["Isolate affected area", "Apply targeted fungicide", "Prune damaged tissue"]
            except:
                res["analysis"] += " Image processing failed."

        elif "yield" in query_lower:
            res["recommendations"] = ["Optimize irrigation", "Soil nutrient test", "Precision fertilization"]
            res["crop_suggestions"] = ["High-yield Rice variety IR64", "Basmati 370"]
        else:
            res["recommendations"] = ["Regular crop monitoring", "Weather-based planning", "Soil health assessment"]
            res["crop_suggestions"] = ["Mustard", "Wheat"]

        res["next_steps"] = [
            "Conduct soil NPK testing",
            "Review local weather forecast",
            "Document any new symptoms"
        ]
        
        return res
    
    def _analyze_agricultural_query(self, query_lower: str) -> dict:
        """Extract context without conversational filler."""
        context = {
            'location': 'India',
            'crop': 'Crops',
            'season': 'Kharif'
        }
        
        locations = ['assam', 'punjab', 'haryana', 'maharashtra', 'karnataka', 'tamil nadu', 'west bengal', 'gujarat', 'rajasthan', 'madhya pradesh', 'uttar pradesh']
        for loc in locations:
            if loc in query_lower:
                context['location'] = loc.title()
                break
        
        crops = ['rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'pulses', 'soybean', 'groundnut', 'mustard']
        for crop in crops:
            if crop in query_lower:
                context['crop'] = crop.title()
                break
        
        if 'yield' in query_lower:
            context['summary'] = f"Direct productivity analysis for {context['crop']} in {context['location']}."
        elif 'disease' in query_lower:
            context['summary'] = f"Diagnostic health assessment for your {context['crop']}."
        else:
            context['summary'] = "Expert agricultural advisory for Indian farming operations."
            
        return context

    def process_query_deterministic(self, query, image_path=None):
        """Always return structured JSON."""
        logger.info(f"Processing query deterministically: {query[:50]}")
        return self._generate_smart_fallback_response(query, image_path)

    def chat(self, query, image_path=None):
        if self.has_llm:
            return self.process_query_llm(query, image_path)
        return self.process_query_deterministic(query, image_path)

if __name__ == "__main__":
    for d in ["logs", "feedback"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    logger.info("FARM 360 AI AGENT SYSTEM INITIATED")
    # By default, use real LLM if configured
    agent = Farm360Agent(use_mock_llm=False)
    test_query = "suggest better crops"
    print(f"Test Query: {test_query}")
    print(agent.chat(test_query))
