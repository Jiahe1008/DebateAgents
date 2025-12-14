import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import List, Dict, Tuple
import datetime
import os

# --- 1. 配置模型路径 ---
# 您可以在此处指定正方和反方的模型路径。
# 如果路径相同，将使用“单一模型”模式；如果不同，将使用“双模型”模式。
# MODEL_PATH_PRO = "F:/work/ckpt_merged"  # 正方模型 (SFT)
# MODEL_PATH_CON = "F:/work/ckpt_merged"  # 反方模型 (SFT)
MODEL_PATH_CON = "F:\work\project\svpi\model\Qwen2.5-1.5B"
MODEL_PATH_PRO = "F:\work\project\svpi\model\Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}


# --- 2. 模型加载与生成函数 ---

def load_model_and_tokenizer(model_dir: str, model_name: str = "模型") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """根据路径加载模型和分词器，并进行缓存。"""
    if model_dir in loaded_models:
        print(f"为 '{model_name}' 使用已缓存的模型: '{model_dir}'")
        return loaded_models[model_dir]

    print(f"正在为 '{model_name}' 从 '{model_dir}' 加载新模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, temperature=0.75,
                                                             top_p=0.8, do_sample=True, repetition_penalty=1.15)
        model.generation_config = generation_config

        loaded_models[model_dir] = (model, tokenizer)
        print(f"'{model_name}' 模型 '{model_dir}' 加载并缓存成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载 '{model_name}' 模型失败，路径 '{model_dir}'。错误: {e}")
        exit()


def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    """根据我们优化的SFT Prompt生成回应，并进行健壮的解析。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 采用最终优化的“双重截断”逻辑
    if full_output.startswith(prompt):
        content = full_output[len(prompt):].strip()
    else:
        sft_response_marker = "### Response:"
        last_marker_index = full_output.rfind(sft_response_marker)
        if last_marker_index != -1:
            content = full_output[last_marker_index + len(sft_response_marker):].strip()
        else:
            content = full_output.strip()

    return content if content else "（模型未能生成有效内容）"


# --- 3. 核心：集成了最强Prompt工程的PromptDirector ---

class PromptDirector:
    """终极通用版Prompt指挥官，使用动态“强力纠偏”指令。"""

    @staticmethod
    def _format_sft_prompt(instruction: str, input_data: Dict) -> str:
        input_str = "\n".join([f"- {key}: {value}" for key, value in input_data.items()])
        return f"你必须严格遵守你的角色和立场。\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n"

    @staticmethod
    def get_opening_prompt(topic: str, role: str) -> str:
        if role == 'pro':
            stance_description = f"你的立场是【正方】，你必须论证【{topic}】这个观点是完全正确的。"
        else:
            stance_description = f"你的立场是【反方】，你必须论证【{topic}】这个观点是完全错误的。"
        instruction = f"你是一位辩论赛的一辩。{stance_description} 你的任务是构建一套逻辑严密的立论体系，并发表一段精彩的立论陈词。严禁偏离你的指定立场。"
        return PromptDirector._format_sft_prompt(instruction, {"辩题": topic})

    @staticmethod
    def get_rebuttal_prompt(topic: str, role: str, my_opening: str, opponent_speech: str) -> str:
        if role == 'pro':
            stance_description = f"你的立场是【正方】，必须支持【{topic}】。"
        else:
            stance_description = f"你的立场是【反方】，必须反对【{topic}】。"
        instruction = f"你是一位辩论赛的辩手。{stance_description} 请分析对方的观点，结合我方立论，对其逻辑漏洞或价值缺陷进行有力的反驳。"
        input_data = {
            "辩题": topic,
            "我方核心立论": my_opening,
            "对方最新辩词": opponent_speech
        }
        return PromptDirector._format_sft_prompt(instruction, input_data)

    @staticmethod
    def get_closing_prompt(topic: str, role: str, my_opening: str, history: List[Dict[str, str]]) -> str:
        if role == 'pro':
            stance_description = f"你的立场是【正方】，必须支持【{topic}】。"
        else:
            stance_description = f"你的立场是【反方】，必须反对【{topic}】。"
        instruction = f"你是一位辩论赛的四辩。{stance_description} 请高度概括并收束全场，进行最终总结，为我方锁定胜局。"

        opponent_role_prefix = '反方' if role == 'pro' else '正方'
        opponent_speeches = "\n".join([f"对方发言: {turn['content']}" for turn in history if
                                       turn['speaker_name'].startswith(opponent_role_prefix)])

        input_data = {
            "辩题": topic,
            "我方核心立论": my_opening,
            "对方整场主要观点摘要": opponent_speeches if opponent_speeches else "对方尚未形成有效观点。"
        }
        return PromptDirector._format_sft_prompt(instruction, input_data)


# --- 4. 文件保存功能 ---

def save_debate_to_file(topic: str, pro_path: str, con_path: str, history: List[Dict[str, str]],
                        save_dir: str = "./baseline"):
    """将完整的辩论内容保存到Markdown文件中。"""

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join([c for c in topic if c.isalnum()]).rstrip()
    filename = f"debate_record_{safe_topic}_{timestamp}.md"

    # 完整的文件路径
    filepath = os.path.join(save_dir, filename)

    print(f"\n正在将辩论记录保存到文件: {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(
            f"# 辩论记录\n\n**辩题**: {topic}\n\n**正方模型**: `{pro_path}`\n**反方模型**: `{con_path}`\n\n---\n\n## 辩论全文\n\n")
        for entry in history:
            f.write(f"### {entry['speaker_name']}:\n{entry['content']}\n\n")
    print(f"文件 '{filepath}' 保存成功。")


# --- 5. 主程序：集成了所有优化的辩论流程 ---

def main():
    """主函数：协调整个辩论流程。"""
    director = PromptDirector()
    debate_history: List[Dict[str, str]] = []

    # 准备阶段：加载模型
    print("--- 准备阶段：加载模型 ---")
    model_pro, tokenizer_pro = load_model_and_tokenizer(MODEL_PATH_PRO, "正方")
    model_con, tokenizer_con = load_model_and_tokenizer(MODEL_PATH_CON, "反方")

    debate_topic = input("\n请输入您想辩论的辩题: ")

    # --- 阶段一：双方立论 ---
    print("\n" + "=" * 20 + "\n第一阶段：立论陈词\n" + "=" * 20)

    print("\n[正方] 正在生成立论...")
    pro_prompt = director.get_opening_prompt(debate_topic, "pro")
    pro_opening_statement = generate_response(model_pro, tokenizer_pro, pro_prompt)
    print(f"正方: {pro_opening_statement}\n")
    debate_history.append({"speaker_name": "正方（立论）", "content": pro_opening_statement})

    print("\n[反方] 正在生成立论...")
    con_prompt = director.get_opening_prompt(debate_topic, "con")
    con_opening_statement = generate_response(model_con, tokenizer_con, con_prompt)
    print(f"反方: {con_opening_statement}\n")
    debate_history.append({"speaker_name": "反方（立论）", "content": con_opening_statement})

    # --- 阶段二：自由辩论 (两轮) ---
    print("\n" + "=" * 20 + "\n第二阶段：自由辩论\n" + "=" * 20)
    latest_speech = con_opening_statement

    for i in range(2):
        round_num = i + 1
        print(f"\n--- 第 {round_num} 轮交锋 ---")

        # 正方回合
        pro_prompt = director.get_rebuttal_prompt(debate_topic, "pro", pro_opening_statement, latest_speech)
        pro_response = generate_response(model_pro, tokenizer_pro, pro_prompt)
        print(f"正方: {pro_response}\n")
        debate_history.append({"speaker_name": f"正方 (第{round_num}轮)", "content": pro_response})
        latest_speech = pro_response

        # 反方回合
        con_prompt = director.get_rebuttal_prompt(debate_topic, "con", con_opening_statement, latest_speech)
        con_response = generate_response(model_con, tokenizer_con, con_prompt)
        print(f"反方: {con_response}\n")
        debate_history.append({"speaker_name": f"反方 (第{round_num}轮)", "content": con_response})
        latest_speech = con_response

    # --- 阶段三：总结陈词 ---
    print("\n" + "=" * 20 + "\n第三阶段：总结陈词\n" + "=" * 20)

    print("\n[反方] 正在生成总结陈词...")
    con_closing_prompt = director.get_closing_prompt(debate_topic, "con", con_opening_statement, debate_history)
    con_closing_response = generate_response(model_con, tokenizer_con, con_closing_prompt)
    print(f"反方: {con_closing_response}\n")
    debate_history.append({"speaker_name": "反方（总结）", "content": con_closing_response})

    print("\n[正方] 正在生成总结陈词...")
    pro_closing_prompt = director.get_closing_prompt(debate_topic, "pro", pro_opening_statement, debate_history)
    pro_closing_response = generate_response(model_pro, tokenizer_pro, pro_closing_prompt)
    print(f"正方: {pro_closing_response}\n")
    debate_history.append({"speaker_name": "正方（总结）", "content": pro_closing_response})

    # --- 阶段四：保存记录 ---
    save_debate_to_file(debate_topic, MODEL_PATH_PRO, MODEL_PATH_CON, debate_history)

    print("\n辩论全部流程结束。")


if __name__ == "__main__":
    main()