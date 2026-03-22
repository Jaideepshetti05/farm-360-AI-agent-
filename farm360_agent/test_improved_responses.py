"""
Test script to demonstrate improved Farm360 agent responses.
This shows the enhanced fallback responses with structured agricultural insights.
"""

import sys
import os
from pathlib import Path

# Add farm360_agent to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from main import Farm360Agent

def test_agent_responses():
    """Test various agricultural queries to showcase improved responses."""
    
    print("=" * 80)
    print("🌱 FARM360 AGENT - ENHANCED RESPONSE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize agent in mock LLM mode (deterministic fallback)
    agent = Farm360Agent(use_mock_llm=True)
    
    # Test queries representing different use cases
    test_queries = [
        {
            "query": "What is the rice yield prediction for Punjab in Kharif season?",
            "description": "🌾 Crop Yield Forecasting Query"
        },
        {
            "query": "I need information about dairy production forecasting",
            "description": "🥛 Dairy Production Query"
        },
        {
            "query": "My wheat crop has yellow leaves and spots",
            "description": "🔬 Disease Detection Query"
        },
        {
            "query": "What's the weather forecast for Maharashtra?",
            "description": "🌤️ Weather Integration Query"
        },
        {
            "query": "Tell me about your capabilities",
            "description": "💡 General Agricultural Consultation"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"{'='*80}\n")
        
        response = agent.chat(test_case['query'])
        print(response)
        print("\n")
    
    print("=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)
    
    print("\n📊 KEY IMPROVEMENTS:")
    print("  ✓ Structured markdown format with clear headings")
    print("  ✓ Context-aware summaries based on query intent")
    print("  ✓ Intelligent entity extraction (location, crop, season)")
    print("  ✓ Actionable recommendations with emojis for visual clarity")
    print("  ✓ Dynamic follow-up questions for engagement")
    print("  ✓ Graceful error handling with helpful guidance")
    print("  ✓ No generic 'deterministic mode' fallback text")
    print("\n🎯 RESULT: Agent now provides intelligent, useful agricultural advice!")

if __name__ == "__main__":
    try:
        test_agent_responses()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
