import requests

def query_ollama(prompt: str, host: str = "http://localhost", port: int = 11434) -> str:
    url = f"{host}:{port}/api/generate"
    payload = {"prompt": prompt, "model": "llama3.2", "stream": False}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("response", "No response received.")
