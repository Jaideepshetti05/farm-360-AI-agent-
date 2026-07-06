import os
import joblib
import numpy as np
import pandas as pd
import pickle
import hashlib
from loguru import logger

# PyTorch imports omitted here; ModelRegistry manages vision model instantiation

from backend.config import settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Whitelist of allowed SHA-256 hashes for pickle/joblib files to prevent arbitrary code execution
ALLOWED_MODEL_HASHES = {
    "production_model_log.pkl": "96d985fc66f588101cc5e805c8794bebafa6c979ff8d92cdec80db920e56db52",
    "dairy_intelligence_v1_20260217_210257.pkl": "4b729de0b5dd8d123b9873abbdd56953c93ee2b6272ed73b5a4d2424c8dab406",
    "dairy_regression_model.pkl": "4b729de0b5dd8d123b9873abbdd56953c93ee2b6272ed73b5a4d2424c8dab406",
    "RandomForest_Tuned.pkl": "e195ba0d88edc2c1d5a18e67d1eaf070e7c7f17437a5d0c3a3ea730afc16a892",
    "classification_scaler.pkl": "1f814764235bdef6b3588d474b3d39cfe79369e7688cd0db43aaf2ba62ab40b6",
    "Animal_label_encoder.pkl": "cab14feeb07828cae99ef06ccbe22b1b0a73bcc274651b4dd62127083fd127e1",
    "Symptom 1_label_encoder.pkl": "38e171bed1c700d3e6882aff43020cfb8302b72f7d65002ae84598d23ed1eb0b",
    "Symptom 2_label_encoder.pkl": "38e171bed1c700d3e6882aff43020cfb8302b72f7d65002ae84598d23ed1eb0b",
    "Symptom 3_label_encoder.pkl": "38e171bed1c700d3e6882aff43020cfb8302b72f7d65002ae84598d23ed1eb0b",
    "Disease_label_encoder.pkl": "9d41c0c171213c808a1175f878667597cfd9f0be6c5129a5e40fbc18239c3df4"
}

def secure_verify_and_load(file_path: str, loader_fn):
    """
    Computes the SHA-256 hash of a file before deserialization and compares it
    with the whitelist of expected hashes to mitigate arbitrary code execution risk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
        
    filename = os.path.basename(file_path)
    expected_hash = ALLOWED_MODEL_HASHES.get(filename)
    if not expected_hash:
        logger.warning(f"[Security] No hardcoded SHA256 registered for {filename}.")
        raise ValueError(f"Security validation blocked loading: {filename} is not registered in the whitelist.")
        
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    actual_hash = sha.hexdigest()
    
    if actual_hash != expected_hash:
        logger.error(f"[Security] FILE INTEGRITY FAILURE: {filename} (hash mismatch). Expected {expected_hash}, got {actual_hash}!")
        raise ValueError(f"Model integrity validation failed: {filename} has been modified or corrupted.")
        
    logger.info(f"[Security] Model file integrity verified: {filename}")
    return loader_fn(file_path)


class Farm360API:
    def __init__(self, model_base_path: str = None):
        logger.info("Initializing ML Models...")

        # Determine base path: explicit arg > settings > fallback
        if model_base_path:
            resolved_base = model_base_path
        elif settings.model_base_path:
            resolved_base = settings.model_base_path
        else:
            resolved_base = BASE_DIR

        logger.info(f"Model base path: {resolved_base}")

        # 1. Crop Regression
        logger.info("Loading Crop Production Regression Model...")
        self.crop_model_path = os.path.join(resolved_base, settings.crop_model_path)

        if not os.path.exists(self.crop_model_path):
            logger.warning(f"Crop model not found: {self.crop_model_path}. Running in limited mode.")
            self.crop_model = None
        else:
            self.crop_model = secure_verify_and_load(self.crop_model_path, joblib.load)

        # 2. Dairy Regression
        logger.info("Loading Dairy Intelligence Model...")
        self.dairy_model_path = os.path.join(resolved_base, settings.dairy_model_path)
        if not os.path.exists(self.dairy_model_path):
             logger.warning(f"Dairy model not found: {self.dairy_model_path}. Running in limited mode.")
             self.dairy_model = None
        else:
             def load_pickle(path):
                 with open(path, "rb") as f:
                     return pickle.load(f)
             self.dairy_model = secure_verify_and_load(self.dairy_model_path, load_pickle)

        # 3. Animal Disease Model
        logger.info("Loading Animal Disease Classification Model...")
        self.animal_model_dir = os.path.join(resolved_base, settings.animal_model_dir)

        if not os.path.exists(self.animal_model_dir):
            logger.warning(f"Animal model directory not found: {self.animal_model_dir}. Running in limited mode.")
            self.animal_model = None
            self.animal_scaler = None
            self.animal_encoders = {}
        else:
            model_file = os.path.join(self.animal_model_dir, "RandomForest_Tuned.pkl")
            scaler_file = os.path.join(self.animal_model_dir, "classification_scaler.pkl")
            
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.warning(f"Animal model files incomplete in {self.animal_model_dir}. Running in limited mode.")
                self.animal_model = None
                self.animal_scaler = None
                self.animal_encoders = {}
            else:
                self.animal_model = secure_verify_and_load(model_file, joblib.load)
                self.animal_scaler = secure_verify_and_load(scaler_file, joblib.load)

                self.animal_encoders = {}
                for feat in ["Animal", "Symptom 1", "Symptom 2", "Symptom 3", "Disease"]:
                    enc_file = os.path.join(self.animal_model_dir, f"{feat}_label_encoder.pkl")
                    if not os.path.exists(enc_file):
                        logger.warning(f"Missing encoder {enc_file}")
                        self.animal_encoders = {}
                        self.animal_model = None
                        break
                    self.animal_encoders[feat] = secure_verify_and_load(enc_file, joblib.load)

        # 4. Crop Vision Model (delegated to lazy model_registry to prevent duplicate initialization)
        logger.info("Crop Disease Vision Model delegated to ModelRegistry.")
        logger.success("Farm360API Initialized (some models may be unavailable in this environment)")

    def predict_crop_yield(
        self, crop: str, season: str, state: str,
        area: float, rainfall: float, fertilizer: float, pesticide: float,
        crop_year: int = None
    ):
        try:
            if self.crop_model is None:
                return {"error": "Crop yield model not available in this environment", "predicted_production": None}

            import datetime
            if crop_year is None:
                crop_year = datetime.datetime.now().year

            df = pd.DataFrame({
                "Crop": [crop], "Crop_Year": [crop_year], "Season": [season], "State": [state], "Area": [area],
                "Annual_Rainfall": [rainfall], "Fertilizer": [fertilizer], "Pesticide": [pesticide]
            })
            log_pred = self.crop_model.predict(df)[0]
            production = np.expm1(log_pred)
            return {
                "predicted_production": float(production),
                "yield_per_area": float(production / area if area > 0 else 0),
            }
        except Exception as e:
            logger.error(f"Crop Yield Inference Failed: {str(e)}")
            return {"error": str(e), "predicted_production": None}


    def predict_dairy_production(self, years: list):
        try:
            if self.dairy_model is None:
                return {"error": "Dairy model not available in this environment"}

            X = np.array([[y] for y in years])
            preds = self.dairy_model.predict(X)
            return {int(y): float(p) for y, p in zip(years, preds)}
        except Exception as e:
            logger.error(f"Dairy Inference Failed: {str(e)}")
            return {"error": str(e)}

    def predict_crop_disease_from_image(self, image_tensor):
        """
        Run crop disease inference on a preprocessed image tensor.
        Returns named disease label + confidence %, not just a class index.
        """
        try:
            from backend.vision_service.registry import model_registry
            vision_model = model_registry.get("crop_disease", "latest")
            if vision_model is None:
                return {"error": "Vision model not available in this environment", "class_index": None}

            raw = vision_model.predict(image_tensor)
            if "error" in raw:
                return {"error": raw["error"], "class_index": None}

            predictions = raw.get("predictions", [])
            if not predictions:
                return {"error": "No predictions returned", "class_index": None}

            top = predictions[0]
            try:
                class_index = vision_model.classes.index(top["label"])
            except ValueError:
                class_index = 0

            return {
                "class_index": class_index,
                "label": top["label"],
                "display_name": top["display_name"],
                "confidence": top["confidence"],
                "confidence_pct": f"{round(top['confidence'] * 100, 1)}%",
                "top_3": [
                    {
                        "label": p["label"],
                        "display_name": p["display_name"],
                        "confidence": p["confidence"],
                        "confidence_pct": f"{round(p['confidence'] * 100, 1)}%",
                    }
                    for p in predictions
                ],
                "is_healthy": "Healthy" in top["label"],
            }
        except Exception as e:
            logger.error(f"Vision Inference Failed: {str(e)}")
            return {"error": str(e), "class_index": None}

    def predict_animal_disease(
        self, animal: str, age: float, temperature: float,
        symptom1: str, symptom2: str, symptom3: str
    ):
        try:
            if self.animal_model is None or self.animal_scaler is None or not self.animal_encoders:
                return {"error": "Animal disease model not available in this environment", "prediction": None}

            def safe_encode(feat_name, val):
                enc = self.animal_encoders[feat_name]
                return enc.transform([val])[0] if val in enc.classes_ else 0

            df = pd.DataFrame({
                "Animal": [safe_encode("Animal", animal)],
                "Age": [age],
                "Temperature": [temperature],
                "Symptom 1": [safe_encode("Symptom 1", symptom1)],
                "Symptom 2": [safe_encode("Symptom 2", symptom2)],
                "Symptom 3": [safe_encode("Symptom 3", symptom3)],
            })

            df_scaled = self.animal_scaler.transform(df)
            pred_idx = self.animal_model.predict(df_scaled)[0]
            pred_disease = self.animal_encoders["Disease"].inverse_transform([pred_idx])[0]
            probs = self.animal_model.predict_proba(df_scaled)[0]
            return {"prediction": str(pred_disease), "confidence": float(max(probs))}
        except Exception as e:
            logger.error(f"Animal Inference Failed: {str(e)}")
            return {"error": str(e), "prediction": None}