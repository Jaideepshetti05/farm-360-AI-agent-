"""
Test script to interact with the local Farm360 server.
"""

import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "secure-secret-key-1234"

def test_health():
    """Test health check endpoint."""
    print("=" * 80)
    print("🏥 TESTING HEALTH CHECK")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_chat_query(query):
    """Test simple chat endpoint."""
    print("=" * 80)
    print(f"💬 TESTING CHAT: '{query}'")
    print("=" * 80)
    
    response = requests.post(
        f"{BASE_URL}/chat",
        headers={"X-API-Key": API_KEY},
        data={"query": query}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: {data['query']}")
        print(f"\nResponse:\n{data['response']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_streaming_query(query):
    """Test streaming chat endpoint."""
    print("=" * 80)
    print(f"📺 TESTING STREAMING: '{query}'")
    print("=" * 80)
    
    response = requests.post(
        f"{BASE_URL}/chat_stream",
        headers={"X-API-Key": API_KEY},
        data={"query": query},
        stream=True
    )
    
    print(f"\nStreaming Response:\n")
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: ") and not decoded.endswith("[DONE]"):
                content = decoded[6:]  # Remove "data: " prefix
                print(content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    try:
        # Test 1: Health check
        test_health()
        
        # Test 2: Simple chat queries
        test_chat_query("suggest better crops for my farm")
        test_chat_query("how to increase milk production?")
        
        # Test 3: Streaming response
        test_streaming_query("my wheat crop has yellow leaves")
        
        print("=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("Make sure the server is running at http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")
