PYTHONPATH="" CUDA_VISIBLE_DEVICES=2,3 python -m accelerate.commands.launch \
    --num_processes=2 \
    /data/gzb/code/DebateAgents/train/train.py