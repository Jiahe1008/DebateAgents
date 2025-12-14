import json
import re
import time
import os
import random
from google import genai
from google.genai import types

os.environ['http_proxy'] = "http://127.0.0.1:7897"
os.environ['https_proxy'] = "http://127.0.0.1:7897"
# ==========================================
# 0. 配置区域
# ==========================================
# 建议使用环境变量，或者直接填入 (注意安全)
os.environ["GOOGLE_API_KEY"] = ""
API_KEY = os.getenv("GOOGLE_API_KEY")

# 初始化 Client
client = genai.Client(api_key=API_KEY)


# 初始化 Client
client = genai.Client(api_key=API_KEY)

# ==========================================
# 1. Prompt 模板：逻辑医生
# ==========================================

CRITIQUE_PROMPT_TEMPLATE = """
# CRITICAL: Your final output MUST be a single, valid JSON object. Do not add any text outside of the JSON structure.

# Role
你是一位逻辑学教授和资深辩论教练，正在为你的新书《辩论中的逻辑艺术》撰写章节。你的写作风格严谨、清晰，善于将复杂的逻辑概念拆解为易于理解的步骤。

# Task
基于给定的【辩题】和【指定谬误】，构造一个高质量的“反面教材”案例，并为其撰写一份详尽的“诊断与修正报告”。

# 输入信息
- **辩题**: {topic}
- **立场**: {stance}  (例如：正方/反方)
- **指定谬误类型**: {fallacy_type}

# 工作流程 (Workflow)
1.  **构思场景 (Scenario Conception)**: 设想一个辩手在讨论 `{topic}` 并持有 `{stance}` 立场时，如何**不经意地**、**听起来有说服力地**犯下 `{fallacy_type}` 错误。关键在于“迷惑性”，要让谬误隐藏在看似合理的论证中。
2.  **撰写谬误文本 (Drafting the Fallacious Argument)**: 撰写一段约 150-200 字的辩论发言片段，作为反面教材。
3.  **撰写诊断报告 (Writing the Diagnosis)**: 切换到逻辑学教授的视角，按照【输出格式】中的 `diagnosis` 结构，一步步拆解、分析并修正这个谬误。

# 内容要求 (Critical)
1.  **谬误文本 (argument_text)**:
    *   **真实感**: 模拟真实的口语辩论风格，有情绪、有立场。
    *   **迷惑性**: 不要过于刻意和夸张，让谬误隐藏在流畅的表达之下。
2.  **诊断报告 (diagnosis)**:
    *   **精确识别**: 准确命名谬误，并引用原文中的关键句。
    *   **深度拆解**: 清晰地解释“A->B”的逻辑链条为何断裂，以及它利用了听众的何种心理偏误。
    *   **有效修正**: 提供具体的、逻辑严谨的修正方案。如果原论点根本无法成立，应直接指出并说明原因。

# JSON 输出格式 (严格遵守)
{{
  "instruction": "你是一位逻辑学教授。请分析以下辩论片段中存在的逻辑谬误，并给出详细的诊断与修正报告。",
  "context": {{
    "topic": "{topic}",
    "stance": "{stance}",
    "fallacy_to_demonstrate": "{fallacy_type}"
  }},
  "input": {{
    "argument_text": "（例如：）对方辩友，我们今天讨论的是是否应该全面禁止含糖饮料。我方认为必须如此！如果我们今天不禁含糖饮料，那么明天青少年就会转而去喝含酒精的饮料，后天他们就会开始尝试软性毒品，最后整个社会将陷入毒品泛滥的深渊！为了避免这可怕的未来，我们必须从源头切断，立即全面禁止含糖饮料！"
  }},
  "output": {{
    "diagnosis": {{
      "1_fallacy_identification": "该片段典型地犯了【滑坡谬误 (Slippery Slope)】的逻辑错误。其核心论证句是：‘如果我们今天不禁含糖饮料，那么明天...最后整个社会将陷入毒品泛滥的深渊！’",
      "2_logical_dissection": "滑坡谬误的问题在于，它预设了一个未经证实的、脆弱的多米诺骨牌效应。发言者将‘不禁含糖饮料’（A）与‘社会毒品泛滥’（Z）之间划上了等号，但未能为 A->B, B->C, ... ->Z 的每一步提供任何证据，仅仅是利用了人们对最终可怕结果（Z）的恐惧。喝可乐和吸毒之间不存在必然的因果联系。",
      "3_correction_and_demonstration": "一个逻辑更严谨的论证，应当聚焦于直接且可验证的后果。修正示范：‘我方认为应限制含糖饮料，并非因为它会导致毒品泛滥，而是基于充分的医学证据。过量摄入糖分与肥胖、糖尿病、心血管疾病等国民健康问题直接相关。因此，我们主张的不是绝对禁止，而是通过加税、限制广告等措施进行有效管制，这才是基于事实、而非恐惧的合理政策。’"
    }}
  }}
}}
"""

# ==========================================
# 2. 谬误库 (Fallacy Bank)
# ==========================================
FALLACIES = [
    # --- 核心/高频谬误 ---
    "滑坡谬误 (Slippery Slope) - 夸大后果，以此推彼",
    "稻草人谬误 (Straw Man) - 歪曲对方观点再攻击",
    "偷换概念 (Equivocation) - 混淆词语定义",
    "人身攻击 (Ad Hominem) - 骂人而不驳理",
    "循环论证 (Circular Reasoning) - 结论即前提",
    "以偏概全 (Hasty Generalization) - 用孤例代表整体",
    "错误归因 (False Cause) - 强行建立因果",
    "非黑即白 (False Dilemma) - 忽视中间选项",
    "诉诸权威 (Appeal to Authority) - 盲信专家而非逻辑",
    "诉诸情感 (Appeal to Emotion) - 用煽情代替论证",

    # --- 进阶/干扰性谬误 ---
    "转移话题 (Red Herring) - 引入无关话题分散注意力",
    "诉诸伪善 (Tu Quoque) - 指责对方也做过同样的事来回避批评",
    "没有真正的苏格兰人 (No True Scotsman) - 临时修改定义以排除反例",
    "诉诸群众 (Ad Populum/Bandwagon) - 认为大家都信就是对的",
    "诉诸无知 (Appeal to Ignorance) - 认为无法证伪即为真",
    "错误类比 (False Analogy) - 将核心属性不同的事物强行对比",
    "起源谬误 (Genetic Fallacy) - 仅仅因为观点的来源（如历史背景）而否定观点",
    "诉诸中庸 (Middle Ground) - 错误地认为折中方案一定是正确的",
    "神枪手谬误 (Texas Sharpshooter) - 先射箭后画靶，只挑选对自己有利的数据",

    # --- 结构/认知谬误 ---
    "合成谬误 (Composition) - 认为部分的性质等同于整体的性质",
    "分解谬误 (Division) - 认为整体的性质等同于部分的性质",
    "诉诸自然 (Appeal to Nature) - 认为自然的即是好的/对的",
    "举证责任倒置 (Burden of Proof Reversal) - 要求对方证伪而非自己证实",
    "赌徒谬误 (Gambler's Fallacy) - 错误地认为独立随机事件受历史结果影响",
    "轶事证据 (Anecdotal Evidence) - 仅凭个人经历反驳客观数据"
]


# ==========================================
# 3. 工具函数
# ==========================================

def call_llm_api(prompt):
    max_retries = 3
    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.8,
                },
            )
            return response.text
        except Exception as e:
            print(f"[API Error] {e}")
            time.sleep(1)
    return "{}"


def save_to_jsonl(data, filename="logic_critique_dataset.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ==========================================
# 4. 主程序
# ==========================================

def run_generation():
    # 读取话题
    topics = []
    try:
        with open("topics.json", "r", encoding="utf-8") as f:
            print("正在读取 topics.json ...")
            topics = json.load(f)
    except:
        topics = ["人工智能", "环保", "教育", "死刑", "生态"]

    print(f"=== 开始生成逻辑批判数据集 (目标: {len(topics) * 3} 条) ===")

    for idx, topic in enumerate(topics):
        # 为每个辩题，随机选 3 个谬误进行生成
        selected_fallacies = random.sample(FALLACIES, 3)

        for fallacy in selected_fallacies:
            stance = random.choice(["正方", "反方"])
            print(f">>> [{idx + 1}/{len(topics)}] 生成: {topic} ({stance}) + {fallacy.split()[0]} ...")

            prompt = CRITIQUE_PROMPT_TEMPLATE.format(
                topic=topic,
                stance=stance,
                fallacy_type=fallacy
            )

            raw_res = call_llm_api(prompt)

            try:
                # 清洗
                text = re.sub(r'```json\s*', '', raw_res)
                text = re.sub(r'```\s*', '', text)
                match = re.search(r'(\{.*\})', text, re.DOTALL)
                clean_text = match.group(1) if match else text.strip()

                data = json.loads(clean_text, strict=False)

                # === 关键修复：匹配新的 JSON 结构 ===

                # 1. 提取输入文本 (Argument Text)
                # Prompt 生成的是 {"input": {"argument_text": "..."}}
                input_text = data["input"]
                if isinstance(input_text, dict) and "argument_text" in input_text:
                    input_text = input_text["argument_text"]

                # 2. 提取诊断内容 (Diagnosis)
                # Prompt 生成的是 {"output": {"diagnosis": {...}}}
                diagnosis_block = data["output"].get("diagnosis", {})

                # 拼接成易读的 Output 格式
                diagnosis_text = (
                    f"【谬误诊断】\n{diagnosis_block.get('1_fallacy_identification', '未识别')}\n\n"
                    f"【逻辑拆解】\n{diagnosis_block.get('2_logical_dissection', '未拆解')}\n\n"
                    f"【修正建议】\n{diagnosis_block.get('3_correction_and_demonstration', '未提供')}"
                )

                final_data = {
                    "instruction": data.get("instruction", "请分析逻辑谬误"),
                    "input": input_text,
                    "output": diagnosis_text,
                    "meta": data.get("context", {})
                }

                save_to_jsonl(final_data)

            except Exception as e:
                print(f"    !!! 解析失败: {e}")
                # print(raw_res[:200]) # 调试用

            time.sleep(1)


if __name__ == "__main__":
    run_generation()