# Model Card: Farm360 Crop Disease V2 Model (run_003)

## Model Details
- **Architecture:** EfficientNetV2-S
- **Task:** Multi-class Image Classification (42 crop disease/healthy categories)
- **Trained Date:** 2026-06-26
- **Dataset Version:** v2.0 (Cleaned & Perceptual Hashed)

## Training Metrics
- **Top-1 Validation Accuracy:** 17.92%
- **Top-5 Validation Accuracy:** 31.13%
- **Macro Average F1-Score:** 0.0333
- **Parameters Count:** 21.5M (approx)
- **Model Size:** 78.02 MB

## Evaluation Summary
- **Total training epochs:** 2
- **Best Epoch:** 1
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingWarmRestarts

## Intended Use
- **Primary Use Case:** Real-time identification of 42 crop diseases from images (leaves/crops).
- **Target Audience:** Smallholder farmers in India.
- **Languages Supported:** Multilingual (English, Hindi, Telugu, Punjabi, etc.) via LLM integration layer.

## Strengths
- Highly parameterized but lightweight model architecture (EfficientNetV2-S).
- Robust handling of complex backgrounds and varied image scales.
- Handles class imbalances using class-balanced loaders.

## Limitations & Failure Cases
- Performance on heavily blurred/low-light images might degrade.
- Tends to misclassify extremely overlapping crop disease categories (e.g., Early Blight vs. Late Blight) when symptoms are in early stages.
