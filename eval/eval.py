import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import List, Dict, Tuple
import datetime
import os
import json
import random
from google import genai
from google.genai import types
# --- 1. 配置区域 ---

# 辩论模型路径
MODEL_PATH_PRO = "F:/work/ckpt_merged"
# MODEL_PATH_CON = "F:/work/project/svpi/model/Qwen2.5-1.5B"
MODEL_PATH_CON = "F:/work/ckpt_merged"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

# 评估日志文件名
EVALUATION_LOG_FILE = "baseline_evaluation_log.json"

os.environ['http_proxy'] = "http://127.0.0.1:7897"
os.environ['https_proxy'] = "http://127.0.0.1:7897"
# ==========================================
# 0. 配置区域
# ==========================================
# 建议使用环境变量，或者直接填入 (注意安全)
# os.environ["GOOGLE_API_KEY"] = "AIzaSyDoXp7XJ3MBFc0M3wepFyDMwSgDNhlwpSg"
# os.environ["GOOGLE_API_KEY"] = "AIzaSyB31QHd3RqQ_bgBxPCHkReCTzjEEhVIIjY"
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCeTGu-S91U2vpCZO1ZngSPeXqmT0dJXpE"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC3Mdedtqw2toxvh8-pZVYrnBdsbc8QwqA"
API_KEY = os.getenv("GOOGLE_API_KEY")

# 初始化 Client
client = genai.Client(api_key=API_KEY)
EVALUATION_ENABLED = True


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


def append_evaluation_to_log(eval_data: dict, filename: str):
    """将单次评估结果追加到JSON日志文件中。"""
    print(f"正在将评估结果追加到日志文件: {EVALUATION_LOG_FILE}")
    try:
        new_log_entry = {
            "filename": filename,
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


def evaluate_debate_with_llm(filename: str):
    """读取指定的辩论记录文件，调用大模型API进行评估，并记录到日志。"""
    if not EVALUATION_ENABLED:
        print("\n评估功能未启用，跳过评估环节。")
        return

    print("\n" + "=" * 20 + f"\n正在评估文件: {filename}\n" + "=" * 20)
    print("正在调用大模型API进行评估，请稍候...")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            transcript = f.read()

        prompt = get_evaluation_prompt(transcript)

        # 假设 client 已经全局初始化
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.7,
            },
        )

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

        # 将评估结果和文件名记录到日志文件
        append_evaluation_to_log(evaluation_json, filename)

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        print("请检查API密钥是否有效，网络连接是否正常，或API返回的JSON格式是否正确。")


def main():
    """
    主函数：批量评估指定目录下的所有.md辩论记录文件。
    """
    target_dir = "E:/project/DebateAgents/output/baseline"  # 指定要扫描的目录

    print(f"--- 批量评估模式 ---")
    print(f"将要扫描目录 '{target_dir}' 下的所有 .md 文件并进行评估。")

    if not os.path.isdir(target_dir):
        print(f"错误：目录 '{target_dir}' 不存在，请检查。")
        return

    # 获取目录下所有.md文件
    md_files = [f for f in os.listdir(target_dir) if f.endswith('.md')]

    if not md_files:
        print(f"在目录 '{target_dir}' 中没有找到任何 .md 文件。")
        return

    print(f"找到 {len(md_files)} 个待评估文件。")

    # 依次评估每个文件
    for md_file in md_files:
        full_path = os.path.join(target_dir, md_file)
        try:
            evaluate_debate_with_llm(full_path)
        except Exception as e:
            print(f"处理文件 {full_path} 时发生严重错误: {e}")
            continue # 即使一个文件出错，也继续处理下一个

    print("\n所有文件评估流程结束。")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()