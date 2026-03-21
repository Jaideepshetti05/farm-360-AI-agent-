# Farm360 AI Agent System

This project contains the orchestrator and integration layer that turns the raw ML prediction models into an interactive, multimodal AI Assistant.

## Features
- **LLM Orchestration**: Parses natural language queries, identifies necessary tools, and provides reasoned agricultural advice. Built with the `google-genai` SDK using Gemini Flash 2.5 context window.
- **Multimodal Support**: Upload images of diseased crops or animals. The Vision Pipeline pre-processes it to standard PyTorch tensors and passes it through the loaded `crop_disease_model`.
- **Memory Management**: Keeps a history of the turn-by-turn chat session, alongside persistent farm profile properties (location, crop type).
- **Decision Engine**: Custom logic to cross-examine multiple factors, e.g. "Do not apply chemical pesticide if rain forecast is > 70% chance."
- **Feedback Collection**: `feedback/feedback_logger.py` logs corrected predictions to help fine-tune future pipeline cycles.

## Architecture & Folders
- `agent_core/`: Contains the LLM connection code `main.py` and the `explainability.py` logic ensuring that data looks clear before passing to the model.
- `api_gateway/`: The `model_wrapper.py` loads `pickle`/`pth` models across Crop Yield, Cattle Disease, Dairy Forecasting, etc.
- `media_pipeline/`: `image_processor.py` transforms files to (`224x224`, `Normalize(...)`) before inference.
- `decision_engine/`: Business rules like logic thresholds.
- `memory/`: Short term array states.
- `external_apis/`: Weather API client wrapper.

## How to Run
1. Ensure your Virtual Environment is activated.
```bash
venv\Scripts\activate
```
2. Export your OpenWeather API Key and optionally your Gemini key:
```bash
set OPENWEATHER_API_KEY=YOUR_WEATHER_KEY
set GEMINI_API_KEY=YOUR_GEMINI_KEY
set FARM360_API_KEY=my_secure_password  # Default mapped to default-secret-key
```
3. Start the FastAPI Backend:
```bash
python farm360_agent\app.py
```
*(The backend will securely host itself asynchronously on localhost:8000)*

4. Open a new terminal window, activate the venv, and start the Streamlit UI:
```bash
streamlit run farm360_agent\app_frontend.py
```

## Example User Queries

**Query 1 (Text + Logic Integration)**
> User: "Can you predict my rice yield in Assam considering the weather?"
> Assistant: "Analyzed using Crop Yield Model: The predicted crop yield is 2341.20 units per acre... Note: High chance of rain coming. Delay pesticide application for 24 hours."

**Query 2 (Multimodal)**
> User uploads `blighted_leaf.jpg`
> Assistant: "Visual analysis indicates: Late Blight with High confidence. I recommend isolating the crop and treating with Mancozeb based on your past history."

## Limitations & Pending Improvements
- The PyTorch vision integration currently runs on CPU synchronously; this can act as a bottleneck on a web server setup.
- The `dairy_intelligence` model picks the latest version automatically, but this should be logged into an explicit model registry like MLFlow.
- External weather is currently mock-generated logic. You need an active OpenWeatherMap / Agrometeorology API key for live deployment.
