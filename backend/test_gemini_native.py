import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env from root
root_dir = Path(__file__).parent.parent
load_dotenv(root_dir / ".env")

def test_gemini():
    sys.stdout.reconfigure(encoding='utf-8')
    api_key = os.environ.get("GOOGLE_API_KEY_1")
    if not api_key:
        print("❌ GOOGLE_API_KEY_1 not found in .env. Please populate it first.")
        sys.exit(1)

    print("=" * 60)
    print("🌾 FARM360 — NATIVE GEMINI SDK TEST")
    print("=" * 60)
    print(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    print("🚀 Initializing google-genai client...")
    
    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        print(f"🤖 Model: {model}")
        print("📡 Endpoint: default native endpoint (generativelanguage.googleapis.com)")
        
        queries = [
            "What is rice blast disease? (answer in 1 short sentence)",
            "Write a Java binary search program (just the core method)",
            "Explain Kubernetes (answer in 1 short sentence)",
            "What is 123 × 456?"
        ]

        for idx, q in enumerate(queries, 1):
            print(f"\n[{idx}/4] Query: {q}")
            response = client.models.generate_content(
                model=model,
                contents=q,
            )
            print(f"✅ Response: {response.text.strip()}")
            
        print("\n" + "=" * 60)
        print("🎉 ALL PHASE 1 VALIDATION QUERIES PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_gemini()
