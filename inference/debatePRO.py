import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import List, Dict, Tuple
import datetime
import os

# --- 1. 配置模型路径 ---
MODEL_PATH_PRO = "/data/gzb/code/DebateAgents/output/run_20251122_032755/ckpt_merged"
MODEL_PATH_CON = "/data/gzb/modelzoo/Qwen2.5-1.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(model_dir: str, model_name: str = "模型") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载指定路径的模型和分词器。"""
    print(f"正在为 '{model_name}' 从 '{model_dir}' 加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        try:
            generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
            generation_config.temperature = 0.7
            generation_config.top_p = 0.9
            generation_config.do_sample = True
            generation_config.repetition_penalty = 1.15
            model.generation_config = generation_config
        except OSError:
            print(f"警告: 在 '{model_dir}' 中未找到 generation_config.json。")
            model.generation_config = GenerationConfig.from_model_config(model.config)
        
        print(f"'{model_name}' 模型已成功加载到 {DEVICE} 设备。")
        return model, tokenizer
    except Exception as e:
        print(f"加载 '{model_name}' 模型失败，请检查路径 '{model_dir}'。错误: {e}")
        exit()

def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, history: List[Dict[str, str]]) -> str:
    """根据对话历史生成回应。"""
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# --- 关键修复：统一数据结构，修复KeyError ---
def save_debate_to_file(topic: str, pro_path: str, con_path: str, history: List[Dict[str, str]]):
    """将完整的辩论内容保存到文件中。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (' ', '-')]).rstrip().replace(" ", "_")
    filename = f"debate_record_{safe_topic}_{timestamp}.md"
    
    print(f"\n正在将辩论记录保存到文件: {filename}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 辩论记录\n\n")
        f.write(f"**辩题**: {topic}\n\n")
        f.write(f"**正方模型**: `{pro_path}`\n")
        f.write(f"**反方模型**: `{con_path}`\n\n")
        f.write("---\n\n")
        f.write("## 辩论全文\n\n")
        
        for entry in history:
            # 使用 .get() 并提供默认值，使代码更健壮
            speaker = entry.get("speaker_name", "未知角色")
            content = entry.get("content", "")
            f.write(f"### {speaker}:\n")
            f.write(f"{content}\n\n")
            
    print(f"文件 '{filename}' 保存成功。")

def main():
    """主函数：协调整个辩论流程。"""
    is_single_model_mode = (MODEL_PATH_PRO == MODEL_PATH_CON)

    print("--- 准备阶段：加载模型 ---")
    if is_single_model_mode:
        print("检测到正反方路径相同，进入 [单一模型] 模式。")
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH_PRO, "共享模型")
        model_pro, tokenizer_pro, model_con, tokenizer_con = model, tokenizer, model, tokenizer
    else:
        print("检测到正反方路径不同，进入 [双模型] 模式。")
        model_pro, tokenizer_pro = load_model_and_tokenizer(MODEL_PATH_PRO, "正方")
        model_con, tokenizer_con = load_model_and_tokenizer(MODEL_PATH_CON, "反方")
    
    debate_topic = input("\n请输入您想辩论的辩题: ")
    
    print("\n" + "="*20 + "\n第一阶段：双方生成立论\n" + "="*20)
    debate_history, history_pro_private, history_con_private = [], [], []

    # --- 关键修改：更强硬、更沉浸的Prompt，避免模型拒绝 ---
    pro_prompt = f"你正在参加一场关于“{debate_topic}”的辩论赛。你的角色是正方，立场是支持。请直接开始你的立论陈词，阐述你的核心论点和证据。"
    current_pro_history = history_pro_private
    current_pro_history.append({"role": "user", "content": pro_prompt})
    pro_opening_statement = generate_response(model_pro, tokenizer_pro, current_pro_history)
    current_pro_history.append({"role": "assistant", "content": pro_opening_statement})
    debate_history.append({"speaker_name": "正方（立论）", "content": pro_opening_statement})
    print(f"正方: {pro_opening_statement}\n")

    # --- 关键修改：强迫反方进入角色 ---
    con_prompt = f"你正在参加一场关于“{debate_topic}”的辩论赛。你的角色是反方，立场是反对。正方刚刚完成了他的立论。现在请直接开始你的立论陈词，阐述你的核心论点来反驳这个议题。"
    current_con_history = history_con_private
    current_con_history.append({"role": "user", "content": con_prompt})
    con_opening_statement = generate_response(model_con, tokenizer_con, current_con_history)
    current_con_history.append({"role": "assistant", "content": con_opening_statement})
    debate_history.append({"speaker_name": "反方（立论）", "content": con_opening_statement})
    print(f"反方: {con_opening_statement}\n")

    print("\n" + "="*20 + "\n第二阶段：自动进行四轮辩论\n" + "="*20)
    
    for i in range(4):
        round_num = i + 1
        
        # 正方回合
        print(f"\n--- 第 {round_num} 轮: 正方发言 ---")
        latest_con_statement = debate_history[-1]['content']
        pro_turn_prompt = f"这是反方的最新发言：\n> {latest_con_statement}\n\n现在轮到你（正方）发言。请直接针对他的观点进行反驳。"
        current_pro_history.append({"role": "user", "content": pro_turn_prompt})
        pro_response = generate_response(model_pro, tokenizer_pro, current_pro_history)
        current_pro_history.append({"role": "assistant", "content": pro_response})
        debate_history.append({"speaker_name": f"正方 (第{round_num}轮)", "content": pro_response})
        print(pro_response)

        # 反方回合
        print(f"\n--- 第 {round_num} 轮: 反方发言 ---")
        latest_pro_statement = debate_history[-1]['content']
        con_turn_prompt = f"这是正方的最新发言：\n> {latest_pro_statement}\n\n现在轮到你（反方）发言。请直接针对他的观点进行反驳。"
        current_con_history.append({"role": "user", "content": con_turn_prompt})
        con_response = generate_response(model_con, tokenizer_con, current_con_history)
        current_con_history.append({"role": "assistant", "content": con_response})
        debate_history.append({"speaker_name": f"反方 (第{round_num}轮)", "content": con_response})
        print(con_response)

    save_debate_to_file(debate_topic, MODEL_PATH_PRO, MODEL_PATH_CON, debate_history)
    print("\n辩论全部流程结束。")

if __name__ == "__main__":
    main()