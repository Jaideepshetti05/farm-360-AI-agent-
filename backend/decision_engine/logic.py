from loguru import logger


class DecisionEngine:
    """Fuses multi-model outputs and business logic."""
    def __init__(self):
        pass

    def evaluate_crop_health_and_yield(self, disease_pred, yield_pred, weather_data=None):
        """Produces advice based on disease and yield."""
        advice = []

        if disease_pred and disease_pred.get("class_index", -1) != 0:  # 0 implies healthy typically
            advice.append("⚠️ **Disease detected** in crop. Treat with appropriate fungicides.")
            if "error" not in disease_pred:
                advice.append(f"Disease class: {disease_pred.get('class_index', 'unknown')}")

        if yield_pred:
            if "error" in yield_pred:
                advice.append(f"Yield prediction unavailable: {yield_pred['error']}")
            else:
                expected_yield = yield_pred.get("yield_per_area", 0)
                advice.append(f"Expected yield is **{expected_yield:.2f} units per acre** under current conditions.")

        if weather_data:
            if "error" in weather_data:
                advice.append(f"Weather data unavailable: {weather_data['error']}")
            else:
                rain_chance = weather_data.get("rain_chance", 0)
                if rain_chance > 70:
                    advice.append("🌧️ **High chance of rain** coming. Delay pesticide application for 24 hours.")

        if not advice:
            advice.append("✅ No significant issues detected. Continue with standard farm management practices.")

        return advice

    def evaluate_animal_health(self, disease_pred):
        """Actionable advice for animal diseases."""
        if not disease_pred or "error" in disease_pred:
            return f"Unable to evaluate animal health: {disease_pred.get('error', 'unknown error') if disease_pred else 'no data'}"

        disease = disease_pred.get("prediction", "Unknown")
        confidence = disease_pred.get("confidence", 0)
        return (
            f"🔍 Animal shows signs of **{disease}** (confidence: {float(confidence)*100:.1f}%). "
            "Immediate actions: **Isolate** the animal and **contact a veterinarian** for confirmation and treatment."
        )