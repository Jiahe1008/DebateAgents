import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import List, Dict, Tuple
import datetime
import os
import json

# --- 1. 配置区域 ---

# 辩论模型路径
MODEL_PATH_PRO = "/data/gzb/code/DebateAgents/output/qwen-1.5B-finetune/ckpt_merged"
MODEL_PATH_CON = "/data/gzb/code/DebateAgents/output/qwen-1.5B-finetune/ckpt_merged"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

# 评估日志文件名
EVALUATION_LOG_FILE = "evaluation_log.json"

# 评估用的大模型API配置
# 请确保您已设置环境变量 GOOGLE_API_KEY
# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
try:
    from google import genai
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("警告: 未找到 GOOGLE_API_KEY 环境变量，评估功能将不可用。")
        EVALUATION_ENABLED = False
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        EVALUATION_ENABLED = True
except ImportError:
    print("警告: 未安装 google-generativeai 库 (pip install google-generativeai)，评估功能将不可用。")
    EVALUATION_ENABLED = False


# --- 2. 辩论模型加载与生成函数 ---
# (这部分与之前版本相同，保持不变)
def load_model_and_tokenizer(model_dir: str, model_name: str = "模型") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if model_dir in loaded_models:
        print(f"为 '{model_name}' 使用已缓存的模型: '{model_dir}'")
        return loaded_models[model_dir]
    print(f"正在为 '{model_name}' 从 '{model_dir}' 加载新模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, temperature=0.75, top_p=0.8, do_sample=True, repetition_penalty=1.15)
        model.generation_config = generation_config
        loaded_models[model_dir] = (model, tokenizer)
        print(f"'{model_name}' 模型 '{model_dir}' 加载并缓存成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载 '{model_name}' 模型失败，路径 '{model_dir}'。错误: {e}")
        exit()

def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

# --- 3. PromptDirector ---
# (这部分与之前版本相同，保持不变)
class PromptDirector:
    @staticmethod
    def _format_sft_prompt(instruction: str, input_data: Dict) -> str:
        input_str = "\n".join([f"- {key}: {value}" for key, value in input_data.items()])
        return f"你必须严格遵守你的角色和立场。\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n"
    @staticmethod
    def get_opening_prompt(topic: str, role: str) -> str:
        if role == 'pro': stance_description = f"你的立场是【正方】，你必须论证【{topic}】这个观点是完全正确的。"
        else: stance_description = f"你的立场是【反方】，你必须论证【{topic}】这个观点是完全错误的。"
        instruction = f"你是一位辩论赛的一辩。{stance_description} 你的任务是构建一套逻辑严密的立论体系，并发表一段精彩的立论陈词。严禁偏离你的指定立场。"
        return PromptDirector._format_sft_prompt(instruction, {"辩题": topic})
    @staticmethod
    def get_rebuttal_prompt(topic: str, role: str, my_opening: str, opponent_speech: str) -> str:
        if role == 'pro': stance_description = f"你的立场是【正方】，必须支持【{topic}】。"
        else: stance_description = f"你的立场是【反方】，必须反对【{topic}】。"
        instruction = f"你是一位辩论赛的辩手。{stance_description} 请分析对方的观点，结合我方立论，对其逻辑漏洞或价值缺陷进行有力的反驳。"
        input_data = {"辩题": topic, "我方核心立论": my_opening, "对方最新辩词": opponent_speech}
        return PromptDirector._format_sft_prompt(instruction, input_data)
    @staticmethod
    def get_closing_prompt(topic: str, role: str, my_opening: str, history: List[Dict[str, str]]) -> str:
        if role == 'pro': stance_description = f"你的立场是【正方】，必须支持【{topic}】。"
        else: stance_description = f"你的立场是【反方】，必须反对【{topic}】。"
        instruction = f"你是一位辩论赛的四辩。{stance_description} 请高度概括并收束全场，进行最终总结，为我方锁定胜局。"
        opponent_role_prefix = '反方' if role == 'pro' else '正方'
        opponent_speeches = "\n".join([f"对方发言: {turn['content']}" for turn in history if turn['speaker_name'].startswith(opponent_role_prefix)])
        input_data = {"辩题": topic, "我方核心立论": my_opening, "对方整场主要观点摘要": opponent_speeches if opponent_speeches else "对方尚未形成有效观点。"}
        return PromptDirector._format_sft_prompt(instruction, input_data)

# --- 4. 文件保存功能 ---
# (这部分与之前版本相同，保持不变)
def save_debate_to_file(topic: str, pro_path: str, con_path: str, history: List[Dict[str, str]]) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join([c for c in topic if c.isalnum()]).rstrip()
    filename = f"debate_record_{safe_topic}_{timestamp}.md"
    print(f"\n正在将辩论记录保存到文件: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 辩论记录\n\n**辩题**: {topic}\n\n**正方模型**: `{pro_path}`\n**反方模型**: `{con_path}`\n\n---\n\n## 辩论全文\n\n")
        for entry in history:
            f.write(f"### {entry['speaker_name']}:\n{entry['content']}\n\n")
    print(f"文件 '{filename}' 保存成功。")
    return filename

# --- 5. 新增：AI评估模块 ---

def get_evaluation_prompt(debate_transcript: str) -> str:
    """
    生成用于评估辩论质量的、结构化的Prompt。
    """
    # 这是为“AI裁判”精心设计的Prompt
    return f"""
# Role: AI Debate Analyst & Judge

你是一位资深的辩论赛裁判和AI模型行为分析师。你的任务是客观、严谨地评估下面提供的由两个AI模型生成的辩论赛全文，并给出一个结构化的评分。

# Task:
请根据以下【评估维度】，对提供的【辩论记录】进行打分（0-10分），并给出综合评语。请严格按照指定的JSON格式输出。

# 评估维度 (0-10分):
1.  **辩论质量分 (Debate Quality Score)**:
    -   评估语言风格是否流畅、专业、有文采。
    -   评估论据是否充实，是否使用了例子、数据或合理的引证。
    -   高分代表语言表现力强，论证有血有肉。

2.  **逻辑思维分 (Logical Thinking Score)**:
    -   重点评估“自由辩论”和“总结陈词”阶段。
    -   评估双方是否真正针对对方的论点进行反驳，还是在自说自话。
    -   评估反驳是否展现了逻辑推导（例如归谬、演绎、识别对方的逻辑漏洞）。
    -   高分代表交锋质量高，逻辑性强。

3.  **跑题分 (Off-Topic Score)**:
    -   评估整场辩论的焦点是否一致，是否围绕核心辩题展开。
    -   **注意：这是一个“不跑题”的分数。10分代表完全没有跑题，0分代表严重跑题。**
    -   重点关注是否出现了“讨论辩论方法论而非辩题本身”、角色混淆（幻觉）、或无意义的重复。
    -   高分代表AI很好地遵循了指令，保持了对话的连贯性和目的性。

4.  **总的评价分 (Overall Evaluation Score)**:
    -   基于以上三项的综合表现，给出一个总分，并附上一段精准的总体评价。

# 辩论记录:
```markdown
{debate_transcript}
输出格式 (严格的JSON):
请严格按照以下JSON格式输出，不要包含任何额外的解释或Markdown标记。 {{ "debate_quality_score": {{ "score": <0-10的整数>, "reason": "简要说明打分理由，例如：语言流畅有文采，但部分论据稍显单薄。" }}, "logical_thinking_score": {{ "score": <0-10的整数>, "reason": "简要说明打分理由，例如：自由辩论阶段的交锋质量不高，未能直接回应对方核心观点。" }}, "off_topic_score": {{ "score": <0-10的整数>, "reason": "简要说明打分理由，例如：从第二轮开始出现焦点漂移，开始讨论抽象概念，属于中度跑题。" }}, "overall_evaluation": {{ "score": <0-10的整数>, "summary": "一段综合性评语，总结这场辩论的亮点与核心问题。" }} }} """

def append_evaluation_to_log(topic: str, eval_data: dict):
    """将单次评估结果追加到JSON日志文件中。"""
    print(f"正在将评估结果追加到日志文件: {EVALUATION_LOG_FILE}")
    try:
        new_log_entry = {
            "topic": topic,
            "debate_quality_score": eval_data["debate_quality_score"]["score"],
            "logical_thinking_score": eval_data["logical_thinking_score"]["score"],
            "off_topic_score": eval_data["off_topic_score"]["score"],
            "overall_evaluation_score": eval_data["overall_evaluation"]["score"],
            "summary": eval_data["overall_evaluation"]["summary"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        logs = []
        if os.path.exists(EVALUATION_LOG_FILE):
            with open(EVALUATION_LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append(new_log_entry)
        with open(EVALUATION_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        print("评估日志更新成功。")
    except Exception as e:
        print(f"更新评估日志时发生错误: {e}")

def evaluate_debate_with_llm(filename: str, topic: str):
    """读取辩论记录文件，调用大模型API进行评估，打印结果，并记录到日志。"""
    if not EVALUATION_ENABLED:
        print("\n评估功能未启用，跳过评估环节。")
        return

    print("\n" + "="*20 + "\n第五阶段：AI裁判评估\n" + "="*20)
    print("正在调用大模型API进行评估，请稍候...")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            transcript = f.read()

        prompt = get_evaluation_prompt(transcript)
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # 从返回的文本中提取JSON部分
        json_text = response.text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        
        evaluation_json = json.loads(json_text)

        print("\n--- AI裁判评估结果 ---\n")
        
        quality = evaluation_json['debate_quality_score']
        logic = evaluation_json['logical_thinking_score']
        off_topic = evaluation_json['off_topic_score']
        overall = evaluation_json['overall_evaluation']

        print(f"【辩论质量分】: {quality['score']}/10")
        print(f"  理由: {quality['reason']}\n")
        
        print(f"【逻辑思维分】: {logic['score']}/10")
        print(f"  理由: {logic['reason']}\n")

        print(f"【跑 题 分】: {off_topic['score']}/10  (分数越高代表越不跑题)")
        print(f"  理由: {off_topic['reason']}\n")

        print("-------------------------\n")

        print(f"【总 评 价 分】: {overall['score']}/10")
        print(f"  AI裁判总结: {overall['summary']}\n")
        
        # 将评估结果记录到日志文件
        append_evaluation_to_log(topic, evaluation_json)

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        print("请检查API密钥是否有效，网络连接是否正常，或API返回的JSON格式是否正确。")
        
def main():
    """主函数：协调整个辩论流程，并在最后进行评估和日志记录。"""
    director = PromptDirector()
    debate_history: List[Dict[str, str]] = []
    
    print("--- 准备阶段：加载模型 ---")
    model_pro, tokenizer_pro = load_model_and_tokenizer(MODEL_PATH_PRO, "正方")
    model_con, tokenizer_con = load_model_and_tokenizer(MODEL_PATH_CON, "反方")
    
    debate_topic = input("\n请输入您想辩论的辩题: ")
    
    # 阶段一：立论
    print("\n" + "="*20 + "\n第一阶段：立论陈词\n" + "="*20)
    print("\n[正方] 正在生成立论...")
    pro_opening_statement = generate_response(model_pro, tokenizer_pro, director.get_opening_prompt(debate_topic, "pro"))
    print(f"正方: {pro_opening_statement}\n")
    debate_history.append({"speaker_name": "正方（立论）", "content": pro_opening_statement})
    
    print("\n[反方] 正在生成立论...")
    con_opening_statement = generate_response(model_con, tokenizer_con, director.get_opening_prompt(debate_topic, "con"))
    print(f"反方: {con_opening_statement}\n")
    debate_history.append({"speaker_name": "反方（立论）", "content": con_opening_statement})

    # 阶段二：自由辩论
    print("\n" + "="*20 + "\n第二阶段：自由辩论\n" + "="*20)
    latest_speech = con_opening_statement
    for i in range(2):
        round_num = i + 1
        print(f"\n--- 第 {round_num} 轮交锋 ---")
        
        print(f"[正方-第{round_num}轮] 正在生成...")
        pro_response = generate_response(model_pro, tokenizer_pro, director.get_rebuttal_prompt(debate_topic, "pro", pro_opening_statement, latest_speech))
        print(f"正方: {pro_response}\n")
        debate_history.append({"speaker_name": f"正方 (第{round_num}轮)", "content": pro_response})
        latest_speech = pro_response
        
        print(f"[反方-第{round_num}轮] 正在生成...")
        con_response = generate_response(model_con, tokenizer_con, director.get_rebuttal_prompt(debate_topic, "con", con_opening_statement, latest_speech))
        print(f"反方: {con_response}\n")
        debate_history.append({"speaker_name": f"反方 (第{round_num}轮)", "content": con_response})
        latest_speech = con_response

    # 阶段三：总结陈词
    print("\n" + "="*20 + "\n第三阶段：总结陈词\n" + "="*20)
    print("\n[反方] 正在生成总结陈词...")
    con_closing_response = generate_response(model_con, tokenizer_con, director.get_closing_prompt(debate_topic, "con", con_opening_statement, debate_history))
    print(f"反方: {con_closing_response}\n")
    debate_history.append({"speaker_name": "反方（总结）", "content": con_closing_response})
    
    print("\n[正方] 正在生成总结陈词...")
    pro_closing_response = generate_response(model_pro, tokenizer_pro, director.get_closing_prompt(debate_topic, "pro", pro_opening_statement, debate_history))
    print(f"正方: {pro_closing_response}\n")
    debate_history.append({"speaker_name": "正方（总结）", "content": pro_closing_response})

    # 阶段四：保存记录
    saved_filename = save_debate_to_file(debate_topic, MODEL_PATH_PRO, MODEL_PATH_CON, debate_history)
    
    # 阶段五：AI裁判评估
    evaluate_debate_with_llm(saved_filename, debate_topic)
    
    print("\n辩论及评估全部流程结束。")

if __name__ == "__main__":
    main()