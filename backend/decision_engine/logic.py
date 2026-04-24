class DecisionEngine:
    """Fuses multi-model outputs and business logic."""
    def __init__(self):
        pass

    def evaluate_crop_health_and_yield(self, disease_pred, yield_pred, weather_data):
        """Produces advice based on disease and yield."""
        advice = []
        if disease_pred.get("class_index", -1) != 0: # 0 implies healthy typically
            advice.append("Disease detected in crop. Treat with appropriate fungicides.")
        
        expected_yield = yield_pred.get("yield_per_area", 0)
        advice.append(f"Expected yield is {expected_yield:.2f} per acre under current conditions.")
        
        if weather_data and weather_data.get("rain_chance", 0) > 70:
            advice.append("High chance of rain coming. Delay pesticide application for 24 hours.")
            
        return advice

    def evaluate_animal_health(self, disease_pred):
        """Actionable advice for animal diseases."""
        # Mapping logic etc.
        disease = disease_pred.get("prediction", "Unknown")
        return f"Animal shows signs of {disease}. Isolate and contact vet immediately."
