"""
Diagnostic script to verify GOOGLE_API_KEY configuration and LLM initialization.
Run this to check if real AI responses are enabled.
"""

import sys
import os
from pathlib import Path

# Add farm360_agent to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

def diagnose_api_key():
    """Check if GOOGLE_API_KEY is properly configured."""
    
    print("=" * 80)
    print("🔍 FARM360 API KEY DIAGNOSTIC TOOL")
    print("=" * 80)
    
    # Check 1: Environment variable
    print("\n1️⃣ Checking GOOGLE_API_KEY environment variable...")
    env_key = os.getenv("GOOGLE_API_KEY")
    
    if env_key:
        print(f"   ✅ Found in environment: {env_key[:10]}...{env_key[-5:]}")
        print(f"   Length: {len(env_key)} characters")
    else:
        print("   ❌ NOT FOUND in environment variables")
    
    # Check 2: .env file
    print("\n2️⃣ Checking .env file...")
    env_file_path = BASE_DIR.parent / ".env"
    
    if env_file_path.exists():
        print(f"   ✅ .env file found at: {env_file_path}")
        
        with open(env_file_path, 'r') as f:
            content = f.read()
            if "GOOGLE_API_KEY=" in content:
                print("   ✅ GOOGLE_API_KEY entry exists in .env")
                
                # Extract the value (without printing full key)
                for line in content.split('\n'):
                    if line.startswith("GOOGLE_API_KEY="):
                        key_value = line.split("=", 1)[1].strip()
                        if key_value and key_value != "your_actual_google_gemini_api_key_here":
                            print(f"   ✅ Value is set (not placeholder): {key_value[:10]}...{key_value[-5:]}")
                        else:
                            print("   ⚠️ Value is placeholder or empty")
                        break
            else:
                print("   ❌ GOOGLE_API_KEY not found in .env")
    else:
        print(f"   ❌ .env file NOT found at: {env_file_path}")
    
    # Check 3: Config loading
    print("\n3️⃣ Checking config.py loading...")
    try:
        from farm360_agent.config import settings
        
        if settings.google_api_key:
            print(f"   ✅ Loaded in settings: {settings.google_api_key[:10]}...{settings.google_api_key[-5:]}")
            print(f"   Length: {len(settings.google_api_key)} characters")
        else:
            print("   ❌ settings.google_api_key is None or empty")
            
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
    
    # Check 4: Agent initialization
    print("\n4️⃣ Testing Farm360Agent initialization...")
    try:
        from farm360_agent.main import Farm360Agent
        
        print("   Importing agent... ✅")
        
        # Test with mock LLM disabled (should try to use real AI)
        print("   Initializing with use_mock_llm=False...")
        agent = Farm360Agent(use_mock_llm=False)
        
        if agent.has_llm:
            print("   ✅ SUCCESS! Real AI responses ENABLED")
            print("   Status: has_llm = True")
        else:
            print("   ❌ Real AI responses DISABLED")
            print("   Status: has_llm = False")
            print("   Reason: Using deterministic fallback")
            
        # Test query
        print("\n5️⃣ Testing actual query...")
        test_query = "What is crop yield forecasting?"
        print(f"   Query: '{test_query}'")
        
        response = agent.chat(test_query)
        
        print(f"\n   Response preview (first 200 chars):")
        print(f"   {'-'*80}")
        print(f"   {response[:200]}...")
        print(f"   {'-'*80}")
        
        # Check if response indicates AI usage
        if "AI Mode Not Enabled" in response or "deterministic" in response.lower():
            print("\n   ⚠️ Response indicates fallback mode")
        else:
            print("\n   ✅ Response appears to be from real AI")
            
    except Exception as e:
        print(f"   ❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    issues = []
    
    if not env_key:
        issues.append("❌ GOOGLE_API_KEY not set in environment variables")
    
    if not env_file_path.exists():
        issues.append("❌ .env file missing")
    
    try:
        from farm360_agent.config import settings
        if not settings.google_api_key or settings.google_api_key == "your_actual_google_gemini_api_key_here":
            issues.append("❌ GOOGLE_API_KEY not loaded in settings or still placeholder")
    except:
        issues.append("❌ Cannot load settings object")
    
    try:
        from farm360_agent.main import Farm360Agent
        agent = Farm360Agent(use_mock_llm=False)
        if not agent.has_llm:
            issues.append("❌ Agent initialized without LLM capability")
    except:
        issues.append("❌ Cannot initialize agent")
    
    if issues:
        print("\n🚨 ISSUES FOUND:\n")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n💡 RECOMMENDED FIXES:\n")
        print("   1. Get your Google Gemini API key from: https://aistudio.google.com/apikey")
        print("   2. Add it to your .env file:")
        print("      GOOGLE_API_KEY=your_actual_api_key_here")
        print("   3. For Render deployment, add GOOGLE_API_KEY to environment variables")
        print("   4. Restart the application after setting the key")
    else:
        print("\n✅ ALL CHECKS PASSED!")
        print("🎉 Real AI responses are properly configured!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        diagnose_api_key()
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
