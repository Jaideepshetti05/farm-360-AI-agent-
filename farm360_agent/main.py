import os
import sys
from loguru import logger
import google.generativeai as genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from api_gateway.model_wrapper import Farm360API, LLMValidator
from media_pipeline.image_processor import MediaPipeline
from decision_engine.logic import DecisionEngine
from external_apis.weather import WeatherClient
from memory.session import MemoryManager
from feedback.feedback_logger import FeedbackSystem
from agent_core.explainability import format_model_prediction
from farm360_agent.config import settings
import json

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
        original_query = query
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

        def get_farm_data() -> str:
            return json.dumps({
                "farm_id": "F-001",
                "location": profile.get("location", "Unknown"),
                "primary_crop": profile.get("primary_crop", "Unknown"),
                "farm_size": profile.get("farm_size", "Unknown"),
                "active_animals": [
                    {"type": "Cow", "age": 4.5, "health_status": "Good"}
                ]
            })

        # 🎯 STRONG EXPERT PROMPT - PRODUCTION READY
        system_instruction = f"""
You are the Farm360 AI Agricultural ERP Assistant and Senior Agronomist. 
You provide enterprise-grade data analysis, precise agricultural advice, and diagnostic intelligence for modern farming operations.

## CURRENT CONTEXT
- Farmer Profile: {json.dumps(profile)}
- Note: Maintain conversational continuity with the recent history.

## MANDATORY TOOL ENFORCEMENT & RULES
1. If the user asks about CROP YIELD or PRODUCTION, you MUST CALL the predict_crop_yield tool.
2. If the user asks about ANIMAL HEALTH or DISEASE, you MUST CALL the predict_animal_disease tool.
3. If the user asks about MILK or DAIRY, you MUST CALL the predict_dairy_production tool.
4. If a tool expects specific required arguments not present in the query, DO NOT GUESS. Explicitly ask the user to provide the missing required parameters.
5. NEVER hallucinate farm data. MUST CALL get_farm_data to understand the user's current farm context before advising.
6. If a tool returns 'not available' or data is missing, output: "missing_data_warning": "Required data not available".

## REASONING & STRUCTURED JSON REQUIREMENT
You operate in two steps:
STEP 1: THINK (Analyze query and use tools)
STEP 2: RESPOND (Output final JSON)

Your response MUST be ONLY a valid JSON object matching exactly this schema:
{{
  "_reasoning_step": "Think step-by-step: what data do I need, which tools did I use, and what is the logical conclusion?",
  "summary": "1-sentence executive summary.",
  "insights": ["Insight 1", "Insight 2"],
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "action_steps": ["Action 1", "Action 2"],
  "missing_data_warning": "Any required data you need, or null if none."
}}
"""
        
        if image_path:
            logger.info("Injecting visual context into LLM state.")
            tensor = self.media.process_image(image_path)
            vision_result = self.api.predict_crop_disease_from_image(tensor)
            eval_text = format_model_prediction("crop_disease_vision", vision_result)
            query += f"\n\n[System Note: VISUAL DATA DETECTED. Vision Model Diagnosis: {eval_text}. MUST BE Integrated into your structured JSON analysis.]"
        
        try:
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            }

            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                tools=[predict_crop_yield, predict_dairy_production, predict_animal_disease, get_weather_forecast, get_farm_data],
                system_instruction=system_instruction
            )
            
            # Retrieve formatted history
            history_data = self.memory.get_chat_history(self.session_id)
            formatted_history = []
            for m in history_data:
                # Map role to API expected 'user' or 'model'
                role = "user" if m["role"] == "user" else "model"
                formatted_history.append({"role": role, "parts": [m["content"]]})
                
            chat = model.start_chat(history=formatted_history, enable_automatic_function_calling=True)
            
            max_retries = 2
            for attempt in range(max_retries):
                response = chat.send_message(query, generation_config=generation_config)
                try:
                    structured_data = LLMValidator.parse_and_validate(response.text)
                    self.memory.add_message(self.session_id, "user", original_query)
                    self.memory.add_message(self.session_id, "model", json.dumps(structured_data, indent=2))
                    return structured_data
                except ValueError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"LLM Output Validation Failed after retries: {e}")
                        raise
                    logger.warning(f"LLM validation failed on attempt {attempt+1}: {e}. Retrying.")
                    query = f"Your last response failed validation: {str(e)}. Please correct it and return ONLY the required JSON mapping."

        except Exception as e:
            logger.error(f"LLM Failure: {str(e)}")
            return self.process_query_deterministic(original_query, image_path)

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
