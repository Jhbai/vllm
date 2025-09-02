# ----- vllm (> v0.7.4) 安裝 ----- #
git clone https://github.com/vllm-project/vllm.git
cd vllm
git fetch origin main
git reset -hard origin/main
git pull --rebase
VLLM_USE_PRECOMPILED=1 pip install --editable .

# ----- 用docker image跑bash，再起service執行 ----- #
docker run -d --gpus all -p 8000:8000 -v "D:\LLM:/model" --name vllm-interactive-server vllm/vllm-openai:v0.8.1
docker exec -it vllm-interactive-server /bin/bash
pip install transformers==4.53.1

lsof -i :8000
kill -9 123

python -m vllm.entrypoints.openai.api_server --model /model/gemma/gemma3_4b --host 0.0.0.0 --port 8000
(vllm serve /model/gemma/gemma3_4b --model-impl transformers)

# ----- 用docker image直接執行 ----- #
docker run --rm --gpus all -p 8000:8000 -v "D:\LLM:/model" --name vllm-nemotron-server vllm/vllm-openai:v0.8.1 --model /model/Nemotron-Research-Reasoning-Qwen-1.5B/ --max-model-len 8192 --host 0.0.0.0
