import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import List, Dict, Tuple
import datetime
import os

# --- 1. 模型路径配置 ---
MODEL_PATH_PRO = "/data/gzb/code/DebateAgents/output/run_20251122_032755/ckpt_merged"
MODEL_PATH_CON = "/data/gzb/modelzoo/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model(role: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if role in loaded_models:
        return loaded_models[role]
    model_path = MODEL_PATH_PRO if role == "pro" else MODEL_PATH_CON
    print(f"正在为角色 '{role}' 从 '{model_path}' 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        generation_config.temperature = 0.7 # 稍微降低温度，减少发散
        generation_config.top_p = 0.9
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.25
        model.generation_config = generation_config
        loaded_models[role] = (model, tokenizer)
        print(f"'{role}' 模型加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        exit()

def generate_response(role: str, prompt: str) -> str:
    model, tokenizer = get_model(role)
    # 使用简单的text-generation pipeline，而不是复杂的chat template
    # 这让任务更纯粹：就是文本续写
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 从返回的完整文本中，只提取新生成的部分
    return response[len(prompt):].strip()

class DebateState:
    def __init__(self, topic: str):
        self.topic = topic
        self.pro_opening: str = ""
        self.con_opening: str = ""
        self.transcript: List[Dict[str, str]] = []
    def add_speech(self, speaker_name: str, content: str):
        self.transcript.append({"speaker_name": speaker_name, "content": content})
    def get_latest_speech(self) -> str:
        return self.transcript[-1]['content'] if self.transcript else "无"

# --- 核心升级：模板续写 Prompt ---
class PromptDirector:
    @staticmethod
    def get_opening_prompt(topic: str, role: str) -> str:
        stance = "支持" if role == "pro" else "反对"
        return f"辩论赛立论陈词。辩题：【{topic}】。我方是【{role}】，我方立场为【{stance}】。\n我的立论是："

    @staticmethod
    def get_rebuttal_prompt(state: DebateState, role: str) -> str:
        stance = "支持" if role == "pro" else "反对"
        your_opening = state.pro_opening if role == "pro" else state.con_opening
        opponent_speech = state.get_latest_speech()

        return f"""
以下是一场辩论的片段。

辩题：【{state.topic}】

我方是【{role}】，我方立场为【{stance}】。
我方的核心论点是：“{your_opening}”

对方辩友刚才的发言是：“{opponent_speech}”

现在轮到我方发言。我方认为：
"""

def save_debate_to_file(topic: str, history: List[Dict[str, str]]):
    # ... (代码不变)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (' ', '-')]).rstrip().replace(" ", "_")
    filename = f"debate_record_{safe_topic}_{timestamp}.md"
    print(f"\n正在将辩论记录保存到文件: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 辩论记录\n\n**辩题**: {topic}\n\n---\n\n## 辩论全文\n\n")
        for entry in history:
            f.write(f"### {entry.get('speaker_name', '未知')}:\n{entry.get('content', '')}\n\n")
    print(f"文件 '{filename}' 保存成功。")

def main():
    debate_topic = input("\n请输入您想辩论的辩题: ")
    state = DebateState(debate_topic)
    director = PromptDirector()

    print("\n" + "="*20 + "\n第一阶段：双方生成立论\n" + "="*20)
    # 正方立论
    pro_prompt = director.get_opening_prompt(state.topic, "pro")
    state.pro_opening = generate_response("pro", pro_prompt)
    state.add_speech("正方（立论）", state.pro_opening)
    print(f"正方: {state.pro_opening}\n")

    # 反方立论
    con_prompt = director.get_opening_prompt(state.topic, "con")
    state.con_opening = generate_response("con", con_prompt)
    state.add_speech("反方（立论）", state.con_opening)
    print(f"反方: {state.con_opening}\n")

    print("\n" + "="*20 + "\n第二阶段：自动进行四轮辩论\n" + "="*20)
    for i in range(4):
        round_num = i + 1
        
        # 正方回合
        print(f"\n--- 第 {round_num} 轮: 正方发言 ---")
        pro_rebuttal_prompt = director.get_rebuttal_prompt(state, "pro")
        pro_response = generate_response("pro", pro_rebuttal_prompt)
        state.add_speech(f"正方 (第{round_num}轮)", pro_response)
        print(pro_response)

        # 反方回合
        print(f"\n--- 第 {round_num} 轮: 反方发言 ---")
        con_rebuttal_prompt = director.get_rebuttal_prompt(state, "con")
        con_response = generate_response("con", con_rebuttal_prompt)
        state.add_speech(f"反方 (第{round_num}轮)", con_response)
        print(con_response)

    save_debate_to_file(debate_topic, state.transcript)
    print("\n辩论全部流程结束。")

if __name__ == "__main__":
    main()