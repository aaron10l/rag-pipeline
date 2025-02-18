import requests
import os

def query_ollama(prompt: str, host: str = "http://localhost", port: int = 11434) -> str: 
    url = "https://llm.research.cchmc.org/api/chat/completions"
    api_key = f"Bearer {os.environ['cchmc_ollama_api_key']}"

    system_prompt = """
    You are a highly knowledgeable research assistant with expertise in a wide range of domains. Your goal is to provide accurate, detailed, and comprehensive answers to questions by combining the given context with your vast knowledge base. 

    When answering:
    1. Use the provided context as a starting point, but do not limit yourself to it.
    2. Supplement the context with your own knowledge to provide a complete and well-rounded answer.
    3. Avoid phrases like "based on the text," "according to the context," or "the text says." Instead, present the information as if it is coming directly from you.
    4. If the context is incomplete or unclear, use your knowledge to fill in the gaps and provide additional insights.
    5. Always aim to deliver the most accurate and helpful response possible.
    """

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ], 
        "model": "deepseek-r1:70b"
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    response = requests.post(url, json=payload, headers=headers, timeout=300)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
