def format_model_prediction(domain, raw_prediction):
    """
    Transforms raw JSON and numeric metrics into natural, highly readable text.
    Eliminates all 'data structure' jargon so the agricultural advisor sounds human.
    """
    if domain == "crop_yield":
        if "error" in raw_prediction:
            return f"Unfortunately, the yield projection engine encountered an issue: {raw_prediction['error']}. Please verify the environmental parameters such as rainfall and area."
        yield_per_area = raw_prediction.get('yield_per_area', 0)
        return (f"Based on the environmental factors provided, the anticipated yield is approximately **{yield_per_area:.2f} units per acre**. "
                "This serves as a strong baseline indicator. If this projection falls below your seasonal expectations, consider evaluating your soil nutrient profiles or adjusting irrigation and fertilizer schedules.")
                
    elif domain == "dairy_forecast":
        if "error" in raw_prediction:
            return f"Dairy yield forecasting is temporarily unavailable ({raw_prediction['error']}). Could you confirm the historical baseline data?"
            
        formatted_forecast = "\n".join([f"- **{year}**: {int(val):,} lbs" for year, val in raw_prediction.items() if str(year).isdigit()])
        return (f"Here is the projected dairy production trajectory over the evaluated timeframe:\n{formatted_forecast}\n\n"
                "Consistent nutrition, herd health management, and optimized milking protocols will be vital to meeting or exceeding these long-term targets.")
        
    elif domain == "crop_disease_vision":
        if "error" in raw_prediction:
            return f"Visual inspection could not be completed. (Error: {raw_prediction['error']}). Ensure the image is clear and well-lit."
        
        idx = raw_prediction.get('class_index', 0)
        
        # Standardized 17-class Agricultural Mapping
        labels = [
            "Healthy", "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Cherry Powdery Mildew",
            "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Grape Black Rot", 
            "Grape Esca", "Grape Leaf Blight", "Peach Bacterial Spot", "Pepper Bell Bacterial Spot",
            "Potato Early Blight", "Potato Late Blight", "Tomato Early Blight", "Tomato Late Blight"
        ]
        disease_name = labels[idx] if 0 <= idx < len(labels) else f"an Unrecognized Anomaly (Code {idx})"
        
        if idx == 0:
            return "Great news! Based on the visual analysis, your crop appears **Healthy**. Maintain your current fertilization and watering routines to ensure continued vitality."
        else:
            return (f"Based on the visual analysis, your crop appears significantly affected by **{disease_name}**, "
                    f"diagnosed with high confidence by our computer vision models. "
                    "Immediate isolation of the affected plants and targeted fungicidal/bactericidal treatments should be strongly evaluated.")
                
    elif domain == "animal_disease":
        if "error" in raw_prediction:
            return f"Animal diagnostic analysis encountered an issue: {raw_prediction['error']}."
        
        disease = raw_prediction.get("prediction", "Unknown")
        conf = raw_prediction.get("confidence", 0)
        conf_pct = f"{float(conf)*100:.1f}%" if isinstance(conf, (float, int)) else conf
        
        return (f"The diagnostic model indicates a strong probability of **{disease}** (Confidence: {conf_pct}). "
                "Closely monitor the animal for rapidly escalating symptoms, such as fever spikes or lethargy. Please consult your local veterinarian immediately for confirmation and a tailored antibiotic or quarantine regimen.")

    return str(raw_prediction)
