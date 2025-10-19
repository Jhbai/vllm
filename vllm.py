import subprocess
import sys

def start_vllm_service():
    model_name = "google/gemma3-27b-it"
    # model_name = "Qwen/Qwen3-32B-Instruct"

    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--quantization", "awq",
        "--tensor-parallel-size", "1",
        "--max-model-len", "4096",
        "--dtype", "auto"
    ]
    
    print(f"Starting vLLM service with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    process.wait()

if __name__ == "__main__":
    start_vllm_service()
