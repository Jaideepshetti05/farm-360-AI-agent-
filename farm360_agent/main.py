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

        # Enhanced agricultural AI expert prompt
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
4. Expertise: Act as an experienced agricultural consultant with deep knowledge of Indian farming practices, crop cycles, weather patterns, and livestock management.
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
            # Fallback to enhanced deterministic response with smart prompt
            return self._generate_smart_fallback_response(query, image_path)

    def _generate_smart_fallback_response(self, query, image_path=None):
        """
        Generate intelligent, structured agricultural responses when LLM is unavailable.
        Uses local ML models and rule-based reasoning to provide valuable insights.
        """
        logger.info("Generating smart fallback response with local intelligence")
        query_lower = query.lower()
        
        # Create smart agricultural prompt for analysis
        agricultural_context = self._analyze_agricultural_query(query_lower)
        
        # Build highly structured deterministic response mimicking the LLM format
        response_sections = []
        
        # 1. Summary - Context-aware and engaging
        summary = agricultural_context.get('summary', "Thank you for reaching out to Farm360. I have analyzed your request using our local intelligence models.")
        response_sections.append(f"### Summary\n{summary}")
        
        # 2. Analysis & Recommendations - Intelligent routing
        if image_path:
            try:
                tensor = self.media.process_image(image_path)
                vision_result = self.api.predict_crop_disease_from_image(tensor)
                eval_text = format_model_prediction("crop_disease_vision", vision_result)
                
                response_sections.append(f"\n### Analysis\n🔍 **Image Analysis Complete**: {eval_text}")
                
                if "Healthy" in eval_text or "healthy" in eval_text:
                    response_sections.append("\n### Recommendations\n✅ **Great news!** Your crop appears healthy. Here's how to maintain it:\n- Continue your scheduled watering and fertilizer applications\n- Perform weekly visual checks for early pest detection\n- Monitor soil moisture levels regularly\n- Document growth stages for future reference")
                else:
                    response_sections.append("\n### Recommendations\n⚠️ **Action Required**: Detected potential issues. Take these steps:\n- **Quarantine**: Immediately isolate affected rows to prevent spread\n- **Treatment**: Apply broad-spectrum fungicide or specialized organic copper spray\n- **Pruning**: Remove dead or decaying leaves carefully\n- **Monitoring**: Check neighboring plants daily for symptom progression\n- **Documentation**: Photograph affected areas for tracking")
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                response_sections.append("\n### Analysis\n⚠️ Image analysis encountered an issue. Please try uploading a clearer photo or describe the symptoms in detail.")
                response_sections.append("\n### Recommendations\n- Try retaking the image in better lighting\n- Focus on the affected plant parts\n- Include both close-up and wider context shots")
        
        elif "yield" in query_lower or "production" in query_lower or "crop" in query_lower:
            try:
                # Extract context or use defaults
                location = agricultural_context.get('location', 'Assam')
                crop = agricultural_context.get('crop', 'Rice')
                season = agricultural_context.get('season', 'Kharif')
                
                context_weather = self.weather.get_forecast(location)
                pred = self.api.predict_crop_yield(
                    crop, season, location, 
                    agricultural_context.get('area', 100.0), 
                    context_weather.get('rain_chance', 0.5) * 30, 
                    200.0, 10.0
                )
                explain_text = format_model_prediction("crop_yield", pred)
                
                response_sections.append(f"\n### Analysis\n📊 **Yield Forecast**: {explain_text}\n\n🌤️ **Current Weather Context**: {context_weather.get('description', 'Data unavailable')}")
                response_sections.append("\n### Recommendations\n🎯 **Optimization Strategies**:\n- Optimize irrigation to compensate for fluctuating weather patterns\n- Perform soil micro-nutrient testing for precision fertilization\n- Consider crop rotation benefits for soil health\n- Monitor pest activity during critical growth stages\n- Plan harvest timing based on maturity indices")
            except Exception as e:
                logger.error(f"Yield prediction error: {e}")
                response_sections.append("\n### Analysis\n📈 I specialize in crop yield forecasting using advanced regression models.")
                response_sections.append("\n### Recommendations\n- Provide specific details: crop type, location, acreage\n- Share recent weather patterns or irrigation data\n- Mention fertilizer and pesticide usage history")
        
        elif "dairy" in query_lower or "milk" in query_lower or "livestock" in query_lower:
            try:
                pred = self.api.predict_dairy_production([2024, 2025, 2026])
                explain_text = format_model_prediction("dairy_forecast", pred)
                
                response_sections.append(f"\n### Analysis\n🥛 **Production Forecast**: {explain_text}")
                response_sections.append("\n### Recommendations\n🐄 **Herd Optimization**:\n- Maintain high-protein feed diet consistently\n- Schedule routine veterinary check-ins quarterly\n- Implement stress reduction measures during peak production\n- Monitor water quality and availability\n- Track breeding cycles for optimal yield planning")
            except Exception as e:
                logger.error(f"Dairy prediction error: {e}")
                response_sections.append("\n### Analysis\n📊 I provide dairy production forecasts using time-series analysis.")
                response_sections.append("\n### Recommendations\n- Share herd size and current production data\n- Provide feeding schedule details\n- Mention any health concerns or breeding information")
        
        elif "disease" in query_lower or "pest" in query_lower or "symptom" in query_lower:
            response_sections.append("\n### Analysis\n🔬 **Disease Diagnostic System**: I can analyze animal diseases using symptomatic data.")
            response_sections.append("\n### Recommendations\n📋 **Provide These Details**:\n- Animal type and age\n- Body temperature reading\n- Specific symptoms observed (minimum 3)\n- Duration of symptoms\n- Recent environmental changes")
        
        elif "weather" in query_lower or "rain" in query_lower or "forecast" in query_lower:
            try:
                location = agricultural_context.get('location', 'Assam')
                context_weather = self.weather.get_forecast(location)
                
                response_sections.append(f"\n### Analysis\n🌤️ **Weather Forecast for {location}**:\n{context_weather}")
                response_sections.append("\n### Recommendations\n🌾 **Agricultural Actions**:\n- Plan irrigation based on predicted rainfall\n- Protect crops if heavy rain or storms forecasted\n- Optimize spraying schedules around wind conditions\n- Prepare drainage systems if excessive rain expected")
            except Exception as e:
                logger.error(f"Weather fetch error: {e}")
                response_sections.append("\n### Analysis\n🌤️ Weather data integration available for agricultural planning.")
                response_sections.append("\n### Recommendations\n- Specify your location for localized forecasts\n- Mention crop sensitivity to weather conditions")
        
        else:
            # General agricultural consultation
            response_sections.append("\n### Analysis\n🌱 **Farm360 Agricultural Intelligence**: I specialize in providing data-driven insights for modern farming operations.")
            response_sections.append("\n### Recommendations\n💡 **How I Can Help**:\n- **Crop Yield Forecasting**: Predict production based on multiple factors\n- **Disease Detection**: Analyze crop images or animal symptoms\n- **Dairy Production**: Forecast milk production trends\n- **Weather Integration**: Location-specific agricultural advice\n- **Livestock Health**: Disease prediction and prevention")
        
        # 3. Next Steps - Intelligent follow-up
        follow_up_questions = agricultural_context.get('follow_up', [
            "Do you have specific acreage or localized weather patterns to factor in?",
            "What specific region are you currently farming in?",
            "Would you like to share more details about your current crop or livestock?"
        ])
        
        next_steps = "\n### Next Steps\n❓ " + " ".join(follow_up_questions)
        response_sections.append(next_steps)

        response_text = "\n".join(response_sections)
        self.memory.add_message(self.session_id, "user", query)
        self.memory.add_message(self.session_id, "assistant", response_text)
        return response_text
    
    def _analyze_agricultural_query(self, query_lower: str) -> dict:
        """
        Analyze agricultural query to extract context and intent.
        Returns structured context for response generation.
        """
        context = {
            'summary': "Thank you for reaching out to Farm360. Based on your query, I'll provide insights using our local ML models.",
            'location': 'Assam',
            'crop': 'Rice',
            'season': 'Kharif',
            'area': 100.0,
            'follow_up': []
        }
        
        # Extract location mentions
        locations = ['assam', 'punjab', 'haryana', 'maharashtra', 'karnataka', 'tamil nadu', 
                    'west bengal', 'gujarat', 'rajasthan', 'madhya pradesh', 'uttar pradesh']
        for loc in locations:
            if loc in query_lower:
                context['location'] = loc.title()
                context['follow_up'].append(f"Are you farming in the {loc.title()} region specifically?")
                break
        
        # Extract crop mentions
        crops = ['rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'pulses', 'soybean', 
                'groundnut', 'rapeseed', 'mustard']
        for crop in crops:
            if crop in query_lower:
                context['crop'] = crop.title()
                context['follow_up'].append(f"What variety of {crop} are you cultivating?")
                break
        
        # Extract season mentions
        seasons = ['kharif', 'rabi', 'zaid', 'summer', 'winter', 'monsoon']
        for season in seasons:
            if season in query_lower:
                context['season'] = season.title()
                break
        
        # Dynamic follow-up questions based on query type
        if not context['follow_up']:
            context['follow_up'] = [
                "Could you share your specific location for more tailored advice?",
                "What crop varieties are you currently working with?",
                "Do you have historical yield data we could leverage?"
            ]
        
        # Customize summary based on query intent
        if 'yield' in query_lower:
            context['summary'] = f"🌾 **Crop Yield Analysis**: I'll help you optimize {context['crop']} production using predictive modeling."
        elif 'disease' in query_lower or 'pest' in query_lower:
            context['summary'] = "🔬 **Diagnostic Analysis**: Let's identify and address your crop or livestock health concerns."
        elif 'dairy' in query_lower or 'milk' in query_lower:
            context['summary'] = "🥛 **Dairy Intelligence**: I'll provide production forecasts and optimization strategies."
        
        return context

    def process_query_deterministic(self, query, image_path=None):
        logger.info(f"Processing query deterministically: {query[:50]}")
        
        # Use the enhanced smart fallback response generator
        return self._generate_smart_fallback_response(query, image_path)

    def chat(self, query, image_path=None):
        if self.has_llm:
            return self.process_query_llm(query, image_path)
        return self.process_query_deterministic(query, image_path)

if __name__ == "__main__":
    for d in ["logs", "feedback"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    logger.info("FARM 360 AI AGENT SYSTEM INITIATED")
    agent = Farm360Agent(use_mock_llm=True)
