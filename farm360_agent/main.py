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
        if not use_mock_llm and settings.google_api_key:
            try:
                genai.configure(api_key=settings.google_api_key)
                self.client = genai.Client(api_key=settings.google_api_key)
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
                logger.warning("No valid GOOGLE_API_KEY found. Using Deterministic Fallback.")
                print("[DEBUG] ❌ No GOOGLE_API_KEY found - using fallback")
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

        # Enhanced agricultural AI expert prompt - DIRECT ADVICE FOCUSED
        system_instruction = f"""
You are Farm360, a premium, knowledgeable, and empathetic agricultural expert and advisor.
Your goal is to help real farmers make critical, real-world decisions based on Farm360's predictive models.

User Profile Context: {profile}

CRITICAL RESPONSE INSTRUCTIONS:
1. Tone & Voice: Be highly conversational, extremely clear, and professional. Avoid speaking like a raw robot or AI. Never repeat yourself. Never output raw data structures like JSON or confusing index numbers.
2. Structure: Break your responses into these exact markdown headings for readability:
   - ### Summary: Provide a brief, direct overview (1-2 lines max).
   - ### Analysis: Explain what the user likely means and evaluate from ML tools in natural language.
   - ### Recommendations: Specific, actionable, step-by-step suggestions to solve the issue.
   - ### Crop Suggestions: If relevant, suggest specific crops with reasoning.
   - ### Next Steps: Concrete actions the user should take next (NOT questions).
3. Quality: Eliminate duplicate words. Ensure the markdown UI renders beautifully.
4. Expertise: Act as an experienced agricultural consultant with deep knowledge of Indian farming practices, crop cycles, weather patterns, and livestock management.
5. IMPORTANT: Do NOT ask clarification questions like "Are you looking to...", "Would you like...", "What do you want?". Instead, provide complete, practical advice immediately.
6. Farmer-First: Assume the user is a farmer who needs practical help, not a developer. Keep it simple, specific, and actionable.
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
            print(f"[DEBUG] LLM error: {e}")
            # Clear message that AI is disabled
            return "⚠️ **AI Service Temporarily Unavailable**\n\nI'm experiencing a technical issue with my AI engine. Please:\n1. Check your internet connection\n2. Verify GOOGLE_API_KEY is correctly configured\n3. Try again in a few moments\n\nFor immediate assistance, you can use our deterministic analysis mode by restarting with `use_mock_llm=True`."

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
                
                # Add Crop Suggestions
                response_sections.append(self._generate_crop_suggestions(crop, season, location))
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
            
            # Add general crop suggestions
            response_sections.append(self._generate_general_crop_suggestions())
        
        # 3. Next Steps - CONCRETE ACTIONS (not questions)
        next_steps_section = self._generate_actionable_next_steps(agricultural_context, query_lower)
        response_sections.append(next_steps_section)

        response_text = "\n".join(response_sections)
        self.memory.add_message(self.session_id, "user", query)
        self.memory.add_message(self.session_id, "assistant", response_text)
        return response_text
    
    def _generate_actionable_next_steps(self, context: dict, query_lower: str) -> str:
        """
        Generate concrete, actionable next steps instead of questions.
        Tells user WHAT TO DO, not WHAT TO ANSWER.
        """
        actions = []
        
        # Based on detected intent, provide specific actions
        if 'yield' in query_lower or 'production' in query_lower:
            actions = [
                "Measure your exact plot area in acres or hectares",
                "Record recent rainfall amounts using a rain gauge",
                "Test soil NPK levels at nearest agricultural center",
                "Document fertilizer application dates and quantities",
                "Take dated photos of crop growth stages weekly"
            ]
        elif 'dairy' in query_lower or 'milk' in query_lower:
            actions = [
                "Set up daily milk yield tracking spreadsheet",
                "Schedule veterinary health check this month",
                "Review and optimize cattle feed protein content",
                "Install water troughs for adequate hydration",
                "Maintain breeding cycle records for planning"
            ]
        elif 'disease' in query_lower or 'pest' in query_lower or 'symptom' in query_lower:
            actions = [
                "Isolate affected plants/animals immediately",
                "Photograph symptoms in good lighting today",
                "Collect leaf/tissue samples for lab testing",
                "Apply recommended fungicide/pesticide within 24 hours",
                "Monitor neighboring crops daily for spread signs"
            ]
        elif 'weather' in query_lower or 'rain' in query_lower:
            actions = [
                "Check IMD weather forecast for your district daily",
                "Prepare drainage channels before heavy rain",
                "Cover sensitive crops if storm predicted",
                "Adjust irrigation schedule based on rainfall",
                "Store fertilizers in dry location to prevent damage"
            ]
        else:
            # General farming guidance
            actions = [
                "Identify your primary farming objective (income vs subsistence)",
                "Assess available resources: land area, water access, labor",
                "Research high-value crops suitable for your climate zone",
                "Connect with local Krishi Vigyan Kendra for expert advice",
                "Start small-scale trials before full implementation"
            ]
        
        # Format as actionable statements
        action_text = "\n### Next Steps\n\nTake these practical actions:\n"
        for i, action in enumerate(actions[:5], 1):  # Limit to top 5 actions
            action_text += f"{i}. {action}\n"
        
        return action_text
    
    def _generate_crop_suggestions(self, current_crop: str, season: str, location: str) -> str:
        """
        Generate specific crop suggestions based on context.
        """
        # Crop rotation recommendations
        rotation_map = {
            'Rice': ['Wheat', 'Mustard', 'Vegetables'],
            'Wheat': ['Rice', 'Soybean', 'Groundnut'],
            'Cotton': ['Wheat', 'Chickpea', 'Tobacco'],
            'Maize': ['Potato', 'Wheat', 'Moong'],
            'Sugarcane': ['Wheat', 'Rice', 'Vegetables']
        }
        
        # Seasonal alternatives
        seasonal_crops = {
            'Kharif': ['Rice', 'Maize', 'Cotton', 'Soybean', 'Groundnut', 'Pulses'],
            'Rabi': ['Wheat', 'Mustard', 'Chickpea', 'Barley', 'Peas'],
            'Zaid': ['Vegetables', 'Fodder crops', 'Moong', 'Watermelon']
        }
        
        suggestion_text = "\n### Crop Suggestions\n\n"
        
        # Rotation suggestion
        if current_crop in rotation_map:
            rotations = rotation_map[current_crop]
            suggestion_text += f"**After {current_crop}, consider rotating with:**\n"
            for i, rot_crop in enumerate(rotations[:3], 1):
                suggestion_text += f"{i}. **{rot_crop}** - improves soil health and breaks pest cycles\n"
        
        # Seasonal alternatives
        if season in seasonal_crops:
            suggestion_text += f"\n**Other suitable {season} crops for your region:**\n"
            alternatives = [c for c in seasonal_crops[season] if c.lower() != current_crop.lower()][:3]
            for alt in alternatives:
                suggestion_text += f"- {alt}\n"
        
        return suggestion_text
    
    def _generate_general_crop_suggestions(self) -> str:
        """
        Provide general high-value crop recommendations for new farmers.
        """
        suggestion_text = "\n### Crop Suggestions\n\n"
        suggestion_text += "**High-value crops to consider based on Indian farming conditions:**\n\n"
        
        # Categorized by investment level
        suggestion_text += "**Low Investment, Quick Returns:**\n"
        suggestion_text += "- **Vegetables** (Tomato, Brinjal, Okra): 60-90 days harvest\n"
        suggestion_text += "- **Leafy greens** (Spinach, Coriander): 30-45 days harvest\n"
        suggestion_text += "- **Radish/Carrot**: 45-60 days harvest\n\n"
        
        suggestion_text += "**Medium Investment, Stable Income:**\n"
        suggestion_text += "- **Wheat/Rice**: Staple crops with guaranteed MSP\n"
        suggestion_text += "- **Mustard**: High oilseed demand, 90-120 days\n"
        suggestion_text += "- **Pulses** (Chickpea, Moong): Fix nitrogen, improve soil\n\n"
        
        suggestion_text += "**Higher Investment, Premium Returns:**\n"
        suggestion_text += "- **Cotton**: Cash crop, 150-180 days\n"
        suggestion_text += "- **Sugarcane**: Annual crop, buy-back agreements\n"
        suggestion_text += "- **Maize**: Growing industrial demand\n\n"
        
        suggestion_text += "**Recommendation:** Start with 1-2 crops you're familiar with, then diversify gradually."
        
        return suggestion_text
    
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
        """
        Deterministic fallback mode - used ONLY when LLM is unavailable.
        This provides basic agricultural analysis using local ML models.
        """
        logger.info(f"Processing query deterministically: {query[:50]}")
        
        # Clear notice that real AI is not enabled
        notice = "⚠️ **AI Mode Not Enabled**\n\nReal AI responses require a valid GOOGLE_API_KEY. Currently using local ML models only.\n\n"
        
        # Generate smart response using local intelligence
        smart_response = self._generate_smart_fallback_response(query, image_path)
        
        # Prepend the notice
        return notice + smart_response

    def chat(self, query, image_path=None):
        if self.has_llm:
            return self.process_query_llm(query, image_path)
        return self.process_query_deterministic(query, image_path)

if __name__ == "__main__":
    for d in ["logs", "feedback"]:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    logger.info("FARM 360 AI AGENT SYSTEM INITIATED")
    agent = Farm360Agent(use_mock_llm=True)
