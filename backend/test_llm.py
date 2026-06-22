"""
Quick diagnostic: check if Farm360 LLM is working.
Run: python backend/test_llm.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("backend", ""))

from backend.config import settings

print("=" * 60)
print("  FARM360 LLM DIAGNOSTIC")
print("=" * 60)

# 1. Check API Key
key = settings.google_api_key
print(f"\n1. GOOGLE_API_KEY from .env:")
if not key:
    print("   ❌ NOT SET (None)")
elif "your_actual" in key:
    print(f"   ❌ PLACEHOLDER: {key}")
    print("   → You need to put a REAL Google Gemini API key in .env")
    print("   → Get one at: https://aistudio.google.com/apikey")
else:
    print(f"   ✅ REAL KEY SET (length: {len(key)}, starts: {key[:8]}...)")

# 2. Check package
print(f"\n2. google.generativeai package:")
try:
    import google.generativeai as genai
    print(f"   ✅ Installed (v{genai.__version__})")
    print(f"   Has GenerativeModel: {hasattr(genai, 'GenerativeModel')}")
except ImportError:
    print("   ❌ NOT INSTALLED")

# 3. Try initializing with real key
print(f"\n3. LLM Initialization Test:")
if key and "your_actual" not in key:
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("   ✅ genai.configure() succeeded")
        
        # 4. Send a test query
        print(f"\n4. Sending test query to Gemini...")
        response = model.generate_content("Say 'Farm360 LLM is working!' in one sentence.")
        print(f"   ✅ RESPONSE: {response.text}")
        print("\n" + "=" * 60)
        print("  ✅ LLM IS WORKING!")
        print("=" * 60)
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
else:
    print("   ❌ SKIPPED - No real API key configured")
    print("\n   To fix:")
    print("   1. Go to https://aistudio.google.com/apikey")
    print("   2. Copy your API key")
    print("   3. Edit .env file:")
    print("      GOOGLE_API_KEY=AIzaSyYourRealKeyHere")
    print("   4. Re-run this script")

print()
