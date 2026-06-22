import requests

url = "http://127.0.0.1:8000/chat_stream"
data = {
    "query": "What crops should I plant in June in Assam, India?",
    "session_id": "default_session",
    "model": "google/gemma-4-26b-a4b-it:free"
}
headers = {
    "X-API-Key": "secure-secret-key-1234"
}

try:
    response = requests.post(url, data=data, headers=headers, stream=True)
    print("Status Code:", response.status_code)
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
except Exception as e:
    print("Error connecting to backend:", e)
