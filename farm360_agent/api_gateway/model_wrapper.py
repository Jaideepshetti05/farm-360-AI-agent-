import os
import joblib
import numpy as np
import pandas as pd
import pickle
from loguru import logger

# Pre-load PyTorch models aggressively during initialization
import torch
import torch.nn as nn
import torchvision.models as models

from config import settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Farm360API:
    def __init__(self):
        logger.info("Initializing ML Models...")
        
        # In Docker, models are mounted at /app/<model_folder>
        # Check if running in Docker by checking mount points
        docker_mount_base = "/app" if os.path.exists("/app") and os.path.isdir("/app") else BASE_DIR
        
        # 1. Crop Regression
        logger.info("Loading Crop Production Regression Model...")
        self.crop_model_path = os.path.join(docker_mount_base, settings.crop_model_path)
        
        if not os.path.exists(self.crop_model_path):
            logger.error(f"Missing critical crop model: {self.crop_model_path}")
            raise FileNotFoundError(f"Missing critical crop model: {self.crop_model_path}")
        self.crop_model = joblib.load(self.crop_model_path)

        # 2. Dairy Regression
        logger.info("Loading Dairy Intelligence Model...")
        self.dairy_model_path = os.path.join(docker_mount_base, settings.dairy_model_path)
        if not os.path.exists(self.dairy_model_path):
             logger.error(f"Missing dairy model: {self.dairy_model_path}")
             raise FileNotFoundError(f"Missing dairy model: {self.dairy_model_path}")
             
        with open(self.dairy_model_path, "rb") as f:
            self.dairy_model = pickle.load(f)

        # 3. Animal Disease Model
        logger.info("Loading Animal Disease Classification Model...")
        self.animal_model_dir = os.path.join(docker_mount_base, settings.animal_model_dir)
        
        if not os.path.exists(self.animal_model_dir):
            logger.error(f"Missing animal model directory: {self.animal_model_dir}")
            raise FileNotFoundError(f"Missing animal model directory: {self.animal_model_dir}")
            
        self.animal_model = joblib.load(os.path.join(self.animal_model_dir, "RandomForest_Tuned.pkl"))
        self.animal_scaler = joblib.load(os.path.join(self.animal_model_dir, "classification_scaler.pkl"))
        
        self.animal_encoders = {}
        for feat in ["Animal", "Symptom 1", "Symptom 2", "Symptom 3", "Disease"]:
            enc_file = os.path.join(self.animal_model_dir, f"{feat}_label_encoder.pkl")
            if not os.path.exists(enc_file):
                logger.error(f"Missing encoder {enc_file}")
                raise FileNotFoundError(f"Missing encoder {enc_file}")
            self.animal_encoders[feat] = joblib.load(enc_file)

        # 4. Crop Vision Model
        logger.info("Loading Crop Disease Vision Model (ResNet18)...")
        self.vision_model_path = os.path.join(docker_mount_base, settings.crop_vision_model_path)
        if not os.path.exists(self.vision_model_path):
             logger.error(f"Missing vision model weights: {self.vision_model_path}")
             raise FileNotFoundError(f"Missing vision model weights: {self.vision_model_path}")
             
        self.vision_model = models.resnet18()
        self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 17)
        self.vision_model.load_state_dict(torch.load(self.vision_model_path, map_location=torch.device('cpu'), weights_only=True))
        self.vision_model.eval()
        
        logger.success("All ML Models Successfully Initialized in Farm360API!")

    def predict_crop_yield(self, crop: str, season: str, state: str, area: float, rainfall: float, fertilizer: float, pesticide: float):
        try:
            df = pd.DataFrame({
                "Crop": [crop], "Season": [season], "State": [state], "Area": [area],
                "Annual_Rainfall": [rainfall], "Fertilizer": [fertilizer], "Pesticide": [pesticide]
            })
            log_pred = self.crop_model.predict(df)[0]
            production = np.expm1(log_pred)
            return {"predicted_production": float(production), "yield_per_area": float(production / area if area > 0 else 0)}
        except Exception as e:
            logger.error(f"Crop Yield Inference Failed: {str(e)}")
            raise e

    def predict_dairy_production(self, years: list):
        try:
            X = np.array([[y] for y in years])
            preds = self.dairy_model.predict(X)
            return {int(y): float(p) for y, p in zip(years, preds)}
        except Exception as e:
            logger.error(f"Dairy Inference Failed: {str(e)}")
            raise e

    def predict_crop_disease_from_image(self, image_tensor):
        try:
            with torch.no_grad():
                out = self.vision_model(image_tensor)
                _, pred = torch.max(out, 1)
            return {"class_index": int(pred.item()), "confidence": "High"}
        except Exception as e:
            logger.error(f"Vision Inference Failed: {str(e)}")
            raise e

    def predict_animal_disease(self, animal: str, age: float, temperature: float, symptom1: str, symptom2: str, symptom3: str):
        try:
            def safe_encode(feat_name, val):
                enc = self.animal_encoders[feat_name]
                return enc.transform([val])[0] if val in enc.classes_ else 0
                    
            df = pd.DataFrame({
                "Animal": [safe_encode("Animal", animal)], "Age": [age], "Temperature": [temperature],
                "Symptom 1": [safe_encode("Symptom 1", symptom1)], "Symptom 2": [safe_encode("Symptom 2", symptom2)],
                "Symptom 3": [safe_encode("Symptom 3", symptom3)]
            })
            
            df_scaled = self.animal_scaler.transform(df)
            pred_idx = self.animal_model.predict(df_scaled)[0]
            pred_disease = self.animal_encoders["Disease"].inverse_transform([pred_idx])[0]
            probs = self.animal_model.predict_proba(df_scaled)[0]
            return {"prediction": str(pred_disease), "confidence": float(max(probs))}
        except Exception as e:
            logger.error(f"Animal Inference Failed: {str(e)}")
            raise e
