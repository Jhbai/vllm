import requests
import json
from flask import Response, stream_with_context

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

def predict(paras):
    model_name = "google/gemma3-27b-it"
    # model_name = "Qwen/Qwen3-32B-Instruct"

    prompt = paras.get("prompt", "")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": True
    }

    try:
        response = requests.post(VLLM_API_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        def generate():
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk.strip():
                    yield chunk

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except requests.exceptions.RequestException as e:
        error_message = {"error": str(e)}
        return Response(json.dumps(error_message), status=500, mimetype="application/json")
