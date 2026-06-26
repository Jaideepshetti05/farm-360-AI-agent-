"""
Farm360 AI – Vision Explainer
================================
Converts raw vision predictions into farmer-friendly LLM explanations.

Pipeline:
    PredictionResult
        ↓ build_prompt()
    Formatted prompt string
        ↓ provider_manager.stream_completion()
    Explanation text (streaming or collected)
        ↓
    Explanation (Pydantic model)

Language support: en, hi, te, kn, mr, pa, bn, ta
"""
from __future__ import annotations

import json
from typing import List, Optional

from loguru import logger

from backend.vision_service.schemas import ClassPrediction, Explanation

# ── Language map ───────────────────────────────────────────────────────────────
_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi",
    "pa": "Punjabi",
    "bn": "Bengali",
    "ta": "Tamil",
    "gu": "Gujarati",
    "ml": "Malayalam",
}

# ── Urgency thresholds by task ─────────────────────────────────────────────────
_URGENCY_RULES = {
    "crop_disease": {
        "Late_Blight": "Critical",
        "Blast": "Critical",
        "Neck_Blast": "Critical",
        "Bacterial_Blight": "High",
        "Red_Rot": "High",
        "Blight": "High",
        "Rust": "Medium",
        "Gray_Leaf_Spot": "Medium",
        "Brown_Spot": "Medium",
        "Healthy": "Low",
    },
    "weed": {"default": "High"},
    "breed": {"default": "Low"},
    "body_condition": {"default": "Medium"},
    "fruit_grade": {"default": "Low"},
    "plant_id": {"default": "Low"},
    "detect": {"default": "Low"},
}

# ── Task-specific prompt templates ────────────────────────────────────────────
_TASK_PROMPTS = {
    "crop_disease": (
        "A farmer has uploaded a crop photo. The Farm360 vision AI detected:\n"
        "{predictions}\n\n"
        "Farmer profile: {profile}\n\n"
        "Please provide:\n"
        "1. A brief confirmation or clarification of the diagnosis (2-3 sentences)\n"
        "2. Immediate treatment steps with exact product names, dosages (kg/ha or g/L), "
        "   and costs in Indian Rupees\n"
        "3. Prevention strategy for the next season\n"
        "4. When to call an extension officer\n\n"
        "Keep the language simple for a rural farmer. Respond in {language}."
    ),
    "breed": (
        "A farmer photographed their animal. The Farm360 vision AI identified:\n"
        "{predictions}\n\n"
        "Farmer profile: {profile}\n\n"
        "Please provide:\n"
        "1. Key characteristics of this breed\n"
        "2. Average milk yield / draft capability\n"
        "3. Best management practices specific to this breed in Indian conditions\n"
        "4. Common health issues to watch for\n\n"
        "Respond in {language}."
    ),
    "weed": (
        "A farmer submitted a field photo. The Farm360 vision AI detected:\n"
        "{predictions}\n\n"
        "Farmer profile: {profile}\n\n"
        "Please provide:\n"
        "1. Identification of the weed species and its harm to the crop\n"
        "2. Recommended herbicide (pre/post-emergent), dosage, and cost in INR\n"
        "3. Manual/mechanical control options\n"
        "4. Integrated weed management strategy\n\n"
        "Respond in {language}."
    ),
    "fruit_grade": (
        "A farmer photographed their harvested fruit. The Farm360 vision AI graded it:\n"
        "{predictions}\n\n"
        "Farmer profile: {profile}\n\n"
        "Please provide:\n"
        "1. Explanation of this quality grade and what it means for pricing\n"
        "2. Storage recommendations to prevent further quality loss\n"
        "3. Best market channels for this grade\n"
        "4. How to improve quality in the next harvest\n\n"
        "Respond in {language}."
    ),
    "plant_id": (
        "A farmer photographed a plant for identification. The Farm360 vision AI detected:\n"
        "{predictions}\n\n"
        "Farmer profile: {profile}\n\n"
        "Please provide:\n"
        "1. Confirmation of plant identity and agricultural significance\n"
        "2. Whether it is a weed, crop, or beneficial plant\n"
        "3. Any agronomic value or risk\n\n"
        "Respond in {language}."
    ),
    "detect": (
        "A farmer uploaded a farm scene photo. The Farm360 vision AI detected:\n"
        "{predictions}\n\n"
        "Farmer profile: {profile}\n\n"
        "Please provide a brief analysis of what was found and any relevant "
        "agricultural advice. Respond in {language}."
    ),
}
_DEFAULT_PROMPT = _TASK_PROMPTS["detect"]


class VisionExplainer:
    """
    Builds prompts for and collects explanations from the LLM.
    Import the singleton `vision_explainer` at the bottom of this module.
    """

    def explain(
        self,
        task: str,
        predictions: List[ClassPrediction],
        user_profile: dict,
        lang: str = "en",
        provider_manager=None,
    ) -> Optional[Explanation]:
        """
        Build prompt from predictions and call the LLM via provider_manager.
        Returns an Explanation or None if LLM is unavailable.
        """
        if provider_manager is None or not predictions:
            return None

        language_name = _LANG_NAMES.get(lang, "English")
        urgency = self._determine_urgency(task, predictions)
        predictions_str = self._format_predictions(predictions)
        profile_str = json.dumps(user_profile, ensure_ascii=False, indent=2)

        prompt_template = _TASK_PROMPTS.get(task, _DEFAULT_PROMPT)
        prompt = prompt_template.format(
            predictions=predictions_str,
            profile=profile_str,
            language=language_name,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Farm360 AI, an expert agricultural advisor for Indian farmers. "
                    "Give practical, specific, actionable advice. Use simple language. "
                    "Always include exact product names, dosages, and prices in INR where relevant."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = ""
            for token in provider_manager.stream_completion(messages):
                response_text += token

            if not response_text or response_text.startswith("⚠️"):
                logger.warning("[Explainer] LLM returned empty or error response")
                return None

            recs = self._extract_recommendations(response_text)
            products = self._extract_products(response_text)

            return Explanation(
                text=response_text.strip(),
                language=lang,
                recommendations=recs,
                urgency=urgency,
                treatment_products=products,
            )

        except Exception as exc:
            logger.error(f"[Explainer] LLM call failed: {exc}")
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_predictions(predictions: List[ClassPrediction]) -> str:
        lines = []
        for p in predictions:
            pct = round(p.confidence * 100, 1)
            lines.append(f"  • {p.display_name} ({pct}% confidence, rank #{p.rank})")
        return "\n".join(lines)

    @staticmethod
    def _determine_urgency(task: str, predictions: List[ClassPrediction]) -> str:
        rules = _URGENCY_RULES.get(task, {})
        if not predictions:
            return "Low"
        top_label = predictions[0].label
        for keyword, urgency in rules.items():
            if keyword.lower() in top_label.lower():
                return urgency
        return rules.get("default", "Medium")

    @staticmethod
    def _extract_recommendations(text: str) -> List[str]:
        """Extract numbered or bulleted list items as recommendations."""
        lines = text.split("\n")
        recs = []
        for line in lines:
            stripped = line.strip()
            # Match numbered items: "1.", "2." or bullets "•", "-", "*"
            if stripped and (
                (len(stripped) > 3 and stripped[0].isdigit() and stripped[1] in ".)")
                or stripped.startswith(("•", "-", "*", "–"))
            ):
                clean = stripped.lstrip("0123456789.)-•*–").strip()
                if len(clean) > 15:
                    recs.append(clean)
        return recs[:6]  # Max 6 recommendations

    @staticmethod
    def _extract_products(text: str) -> List[str]:
        """Heuristic extraction of product names (words before dosage patterns)."""
        import re
        # Look for patterns like "Tricyclazole 75WP", "Mancozeb 75WP @ ..."
        products = re.findall(
            r"\b([A-Z][a-zA-Z]+(?:\s+\d+[A-Z]+)?)\s+@\s+\d",
            text
        )
        return list(dict.fromkeys(products))[:4]  # Deduplicate, max 4


# ── Singleton ──────────────────────────────────────────────────────────────────
vision_explainer = VisionExplainer()
