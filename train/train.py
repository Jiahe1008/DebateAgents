import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import wandb
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# ================= 配置区域 =================
# 模型路径 (假设使用 Qwen2.5-1.5B，你可以换成 Qwen2.5-1.5B-Instruct)
MODEL_ID = "/data/gzb/modelzoo/Qwen2.5-1.5B-Instruct" 
# 你的数据集路径
DATA_PATH = "/data/gzb/code/DebateAgents/dataset/training_data_fixed.jsonl"
# 输出路径
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/data/gzb/code/DebateAgents/output/run_123"

# 3. 动态构建最终模型保存路径
# 结果会变成: .../output/run_20231122_143000/ckpt_merged
FINAL_SAVE_DIR = os.path.join(OUTPUT_DIR, "ckpt_merged")

# WandB 项目名称 (Run 的名字也可以加上时间)
WANDB_PROJECT = "debate-qwen-finetune"
WANDB_RUN_NAME = f"qwen-1.5b-debate-123"

# 训练超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 4          # 根据显存调整，1.5B模型显存占用很小，可以适当调大
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 10000   # 辩论上下文通常较长，建议设大一点
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# ================= 1. 初始化 WandB =================
# 确保你已经运行了 `wandb login` 或者设置了 WANDB_API_KEY 环境变量
wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

# ================= 2. 数据处理函数 =================
def formatting_prompts_func(examples):
    """
    将数据集格式化为模型输入的 Prompt。
    处理 input 为空的情况。
    """
    output_texts = []
    
    for instruction, input_text, output_text in zip(examples['instruction'], examples['input'], examples['output']):
        # 构建 Prompt 结构
        # 使用 Alpaca 风格或更清晰的辩论风格分隔符
        
        prompt = f"### Instruction:\n{instruction}\n\n"
        
        # 只有当 input 不为空时，才添加 Input 部分
        if input_text and input_text.strip() != "":
            prompt += f"### Input:\n{input_text}\n\n"
        
        prompt += f"### Response:\n{output_text}"
        
        # Qwen 的 EOS token 会由 Trainer 自动处理，但显式添加是个好习惯，
        # 这里 SFTTrainer 会自动把 text 字段作为训练目标
        output_texts.append(prompt)
        
    return output_texts

# 加载数据集
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.shuffle(seed=42)
print(f"Loaded {len(dataset)} examples.")

# ================= 3. 加载模型与 Tokenizer =================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,  # 1.5B 建议不量化以获得更好效果，除非显存不够
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Qwen通常没有pad_token，设为eos
tokenizer.padding_side = "right" # 训练通常使用 right padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16, # 推荐在 Ampere 架构显卡(3090/4090/A100)上使用
    trust_remote_code=True,
    # quantization_config=bnb_config # 如果需要量化请取消注释
)

# ================= 4. 配置 LoRA =================
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_RANK,
    bias="none",
    task_type="CAUSAL_LM",
    # Qwen 的核心模块，建议对所有线性层进行微调以获得最好效果
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# ================= 5. 设置训练参数 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    logging_steps=2,
    num_train_epochs=NUM_EPOCHS,
    save_strategy="epoch",       # 按步数保存 checkpoint
    # save_steps=100,              # 每 100 步保存一次 checkpoint
    evaluation_strategy="no",    # 如果你有验证集，可以设为 "steps"
    report_to="wandb",           # 记录到 wandb
    bf16=True,                   # 开启 bf16 加速 (如果是旧显卡如 2080ti 改为 fp16=True)
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    save_total_limit=3,
    lr_scheduler_type="cosine",
    group_by_length=False,        # 将长度相似的样本分在一组，加速训练
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_prompts_func, # 使用我们要定义的格式化函数
    packing=False, # 如果显存够大，设为 True 可以加速训练
)

print("Starting training...")
trainer.train()

print(f"Saving LoRA adapter to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Merging model and saving full weights...")

# 清理显存，防止 OOM
del model
del trainer
torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 加载训练好的 LoRA 适配器
model_to_merge = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

# 合并权重
merged_model = model_to_merge.merge_and_unload()

# 保存完整模型
if not os.path.exists(FINAL_SAVE_DIR):
    os.makedirs(FINAL_SAVE_DIR)

merged_model.save_pretrained(FINAL_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(FINAL_SAVE_DIR)

print(f"Full model saved to: {FINAL_SAVE_DIR}")
print("Training finished!")