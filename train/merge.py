import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,

)
from peft import PeftModel
import wandb
import os
from datetime import datetime


MODEL_ID = "/data/gzb/modelzoo/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "/data/gzb/code/DebateAgents/output/run_123/checkpoint-56"
FINAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "ckpt_merged")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# 加载训练好的 LoRA 适配器
model_to_merge = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# 合并权重
print("begin merge...")
merged_model = model_to_merge.merge_and_unload()

# 保存完整模型
if not os.path.exists(FINAL_SAVE_DIR):
    os.makedirs(FINAL_SAVE_DIR)

merged_model.save_pretrained(FINAL_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(FINAL_SAVE_DIR)

print(f"Full model saved to: {FINAL_SAVE_DIR}")