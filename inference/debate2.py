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
            generation_config.top_p = 0.95
            generation_config.do_sample = True
            generation_config.repetition_penalty = 1.2 # 进一步加强重复惩罚
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

def save_debate_to_file(topic: str, pro_path: str, con_path: str, history: List[Dict[str, str]]):
    """将完整的辩论内容保存到文件中。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (' ', '-')]).rstrip().replace(" ", "_")
    filename = f"debate_record_{safe_topic}_{timestamp}.md"
    
    print(f"\n正在将辩论记录保存到文件: {filename}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        # ... (此处代码与之前版本相同，无需修改)
        f.write(f"# 辩论记录\n\n")
        f.write(f"**辩题**: {topic}\n\n")
        f.write(f"**正方模型**: `{pro_path}`\n")
        f.write(f"**反方模型**: `{con_path}`\n\n")
        f.write("---\n\n")
        f.write("## 辩论全文\n\n")
        
        for entry in history:
            speaker = entry.get("speaker_name", "未知角色")
            content = entry.get("content", "")
            f.write(f"### {speaker}:\n")
            f.write(f"{content}\n\n")
            
    print(f"文件 '{filename}' 保存成功。")

# --- 核心升级：全新的“铁律”和“沉浸式”Prompt ---
def get_immersive_prompt(topic: str, role: str, your_opening_statement: str, opponent_opening_statement: str, latest_opponent_statement: str) -> str:
    """
    生成一个极具对抗性、现场感和角色感的辩论Prompt。
    """
    stance = "支持" if role == "正方" else "反对"
    opponent_role = "反方" if role == "正方" else "正方"

    prompt = f"""
[赛场情况]
你正站在一场关于“{topic}”的激烈辩论赛的舞台上，评委和观众正注视着你。现在轮到你发言。

[你的身份与铁律]
你的身份是 **{role}**。
你的 **唯一、绝对、不可动摇的立场** 是：**{stance}“{topic}”**。
无论对方说什么，你都必须从你的立场出发进行思考和反驳。绝不允许背叛或混淆你的立场！

[你的武器库]
1.  **你的核心论点**: 这是你的立场根本，要时刻回顾并捍卫。
    > {your_opening_statement}
2.  **对方的核心论点**: 这是对方的立场根本，是你要驳倒的根本目标。
    > {opponent_opening_statement}
3.  **对方的最新发言**: 这是你当前需要立即摧毁的目标。
    > {latest_opponent_statement}

[你的任务]
1.  **保持现场感**: 使用“对方辩友”、“我方认为”、“尊敬的评委”等词汇。
2.  **主动攻击**: 不要只做被动的回应。分析对方的最新发言，指出其中的**逻辑谬误**（如偷换概念、稻草人攻击、错误归因等）、**证据不足**或**观点偏离主题**的地方。
3.  **捍卫自己**: 将对方的攻击引回你的核心论点上进行化解。

现在，请开始你的发言。直接以辩手的口吻说出你的辩词：
"""
    return prompt

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

    # --- 立论阶段：使用更强硬的指令，并严格分离历史记录 ---
    pro_prompt = f"你正在一场辩论赛中。你的角色是正方，你的立场是 **支持**“{debate_topic}”。这是你的立论环节，请以辩手的口吻，清晰、有力地阐述你的核心论点和证据，不要有任何动摇。不少于500字"
    history_pro_private.append({"role": "user", "content": pro_prompt})
    pro_opening_statement = generate_response(model_pro, tokenizer_pro, history_pro_private)
    history_pro_private.append({"role": "assistant", "content": pro_opening_statement})
    debate_history.append({"speaker_name": "正方（立论）", "content": pro_opening_statement})
    print(f"正方: {pro_opening_statement}\n")

    con_prompt = f"你正在一场辩论赛中。你的角色是反方，你的立场是 **反对**“{debate_topic}”。正方刚刚完成了他的立论。现在是你的立论环节，请以辩手的口吻，直接、有力地阐述你的核心论点来反驳这个议题。不少于500字"
    history_con_private.append({"role": "user", "content": con_prompt})
    con_opening_statement = generate_response(model_con, tokenizer_con, history_con_private)
    history_con_private.append({"role": "assistant", "content": con_opening_statement})
    debate_history.append({"speaker_name": "反方（立论）", "content": con_opening_statement})
    print(f"反方: {con_opening_statement}\n")

    print("\n" + "="*20 + "\n第二阶段：自动进行四轮辩论\n" + "="*20)
    
    # 在双模型模式下，我们必须严格使用分离的历史记录（private history）
    # 单一模型模式（如果使用）则继续共享一个历史
    current_pro_history = debate_history if is_single_model_mode else history_pro_private
    current_con_history = debate_history if is_single_model_mode else history_con_private

    for i in range(4):
        round_num = i + 1
        
        # 正方回合
        print(f"\n--- 第 {round_num} 轮: 正方发言 ---")
        latest_con_statement = debate_history[-1]['content']
        pro_turn_prompt = get_immersive_prompt(debate_topic, "正方", pro_opening_statement, con_opening_statement, latest_con_statement)
        current_pro_history.append({"role": "user", "content": pro_turn_prompt})
        pro_response = generate_response(model_pro, tokenizer_pro, current_pro_history)
        current_pro_history.append({"role": "assistant", "content": pro_response})
        debate_history.append({"speaker_name": f"正方 (第{round_num}轮)", "content": pro_response})
        print(pro_response)

        # 反方回合
        print(f"\n--- 第 {round_num} 轮: 反方发言 ---")
        latest_pro_statement = debate_history[-1]['content']
        con_turn_prompt = get_immersive_prompt(debate_topic, "反方", con_opening_statement, pro_opening_statement, latest_pro_statement)
        current_con_history.append({"role": "user", "content": con_turn_prompt})
        con_response = generate_response(model_con, tokenizer_con, current_con_history)
        current_con_history.append({"role": "assistant", "content": con_response})
        debate_history.append({"speaker_name": f"反方 (第{round_num}轮)", "content": con_response})
        print(con_response)

    save_debate_to_file(debate_topic, MODEL_PATH_PRO, MODEL_PATH_CON, debate_history)
    print("\n辩论全部流程结束。")

if __name__ == "__main__":
    main()