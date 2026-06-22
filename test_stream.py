import requests
import sys

def main():
    url = "http://127.0.0.1:8000/chat_stream"
    headers = {"X-API-Key": "secure-secret-key-1234"}
    data = {
        "query": "Hello",
        "session_id": "test",
        "model": "google/gemma-4-26b-a4b-it:free"
    }
    print("Sending POST request to:", url)
    try:
        with requests.post(url, headers=headers, data=data, stream=True, timeout=10) as r:
            print("Status Code:", r.status_code)
            for line in r.iter_lines():
                if line:
                    print(line.decode("utf-8"))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
