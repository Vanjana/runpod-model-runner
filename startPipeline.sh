source /workspace/s3
source /workspace/venv/bin/activate

export HF_HOME=/workspace/models/_hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/models/_hf_cache

cd /workspace/app/src
python main.py
cd ..