# 🌾 Farm360 AI — Intelligent Agricultural Assistant

Farm360 is a production-ready AI system that combines machine learning models with a conversational interface to help farmers make data-driven decisions in real time.

## 🚀 Features

- **Crop Yield Forecasting** — Predict crop production based on rainfall, fertilizer, season, and region
- **Dairy Production Intelligence** — Multi-year milk yield forecasts using time-series regression
- **Crop Disease Detection** — Computer vision (ResNet18) to classify 17 crop disease categories from images
- **Animal Disease Diagnosis** — ML classification of livestock conditions from clinical symptoms
- **Conversational AI** — Powered by Google Gemini 2.5 Flash with tool-calling for intelligent routing
- **Streaming Chat UI** — Real-time Server-Sent Events (SSE) streamed to a Next.js frontend

---

## 🏗️ Project Structure

```
ml-models/
├── farm360_agent/          # FastAPI backend + AI orchestration
│   ├── app.py              # Main FastAPI application with SSE streaming
│   ├── main.py             # Farm360Agent orchestrator (LLM + deterministic)
│   ├── config.py           # Pydantic settings (env vars, model paths)
│   ├── api_gateway/        # ML model loading & inference wrappers
│   ├── agent_core/         # Response formatting & explainability
│   ├── decision_engine/    # Rule-based decision logic
│   ├── media_pipeline/     # Image preprocessing for vision model
│   ├── memory/             # Session & conversation memory
│   ├── external_apis/      # Weather API integration
│   └── feedback/           # User feedback logging
│
├── farm360_nextjs/         # Next.js 16 frontend (React + TypeScript)
│   └── src/
│       ├── app/            # Next.js App Router pages
│       └── components/     # ChatCanvas, ChatInput, Sidebar
│
├── crop_regression/        # Crop yield prediction training scripts
├── crop_vision/            # ResNet18 crop disease vision training
├── dairy_module/           # Dairy production forecasting training
├── animal_module/          # Animal classification training
├── health_module/          # Animal disease diagnosis training
├── vision_v2/              # Vision model v2 experiments
│
├── Dockerfile              # Production Docker image (Python 3.11)
├── docker-compose.yml      # Multi-container orchestration
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Setup & Running Locally

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18+
- A Google Gemini API key (optional — the system falls back to deterministic mode)

### 1. Backend (FastAPI)

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp farm360_agent/.env.example farm360_agent/.env
# Fill in GOOGLE_API_KEY and FARM360_API_KEY in .env

# Start the backend server
cd farm360_agent
python app.py
# → Running at http://127.0.0.1:8000
```

### 2. Frontend (Next.js)

```bash
cd farm360_nextjs

# Create local environment file
echo "NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000" > .env.local
echo "NEXT_PUBLIC_FARM360_API_KEY=your-api-key-here" >> .env.local

# Install dependencies and run
npm install
npm run dev
# → Running at http://localhost:3000
```

---

## 🔐 Environment Variables

### `farm360_agent/.env`
```env
GOOGLE_API_KEY=your_google_gemini_api_key
FARM360_API_KEY=your_custom_backend_secret
```

### `farm360_nextjs/.env.local`
```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_FARM360_API_KEY=your_custom_backend_secret
```

> ⚠️ **Never commit `.env` files to version control.** These are listed in `.gitignore`.

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

This starts:
- FastAPI backend on port `8000`
- Next.js frontend on port `3000`

---

## 🧠 ML Models

The trained model weights are **not included** in this repository due to their size. Place them in the following paths before starting the backend:

| Model | Path |
|---|---|
| Crop Yield Regression | `crop_regression/models/crop_regression_model.pkl` |
| Dairy Forecasting | `models/dairy_regression_model.pkl` |
| Animal Disease Classifier | `health_module/models/animal_disease_model.pkl` |
| Crop Disease Vision (ResNet18) | `crop_vision/models/crop_disease_model.pth` |

To train the models from scratch, run the training scripts inside each module directory.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/chat` | POST | Non-streaming text chat |
| `/chat_stream` | POST | SSE streaming chat response |
| `/analyze_image` | POST | Multimodal image + text analysis |
| `/docs` | GET | Interactive API documentation |

All endpoints (except `/` and `/docs`) require `X-API-Key` header.

---

## 📚 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Models | scikit-learn, PyTorch (ResNet18) |
| LLM | Google Gemini 2.5 Flash |
| Frontend | Next.js 16, React 18, TypeScript |
| Streaming | Server-Sent Events (SSE) |
| Styling | Tailwind CSS |
| Logging | Loguru |
| Config | Pydantic Settings |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
