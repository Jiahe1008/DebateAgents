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

# ==========================================
# 1. 定义高约束 Prompt 模板 (新增总结陈词)
# ==========================================

# OPENING_PROMPT_TEMPLATE = """
# # Role
# 你是一位世界级的辩论教练和金牌撰稿人。
#
# # Task
# 请根据提供的【输入信息】，生成一个用于微调模型的数据样本。该样本必须严格遵守指定的 JSON 格式。
#
# # 输入信息
# - **辩题**: {topic}
# - **立场**: {stance}
#
# # 内容要求 (Critical)
# 1. **结构完整**：[问候] -> [核心定义] -> [判准] -> [分论点] -> [总结]。
# 2. **判准明确**：必须明确提出判断标准。
# 3. **语言风格**：气势磅礴，逻辑自洽，使用标准辩论语言。
#
# # JSON输出格式
# {{
#   "type": "opening_statement",
#   "instruction": "你是一位辩论赛的辩手。请根据给定的辩题和你的立场，构建一套逻辑严密的立论体系，明确定义和判准，并发表一段精彩的立论陈词。",
#   "input": {{
#     "topic": "{topic}",
#     "stance": "{stance}"
#   }},
#   "output": {{
#     "opening_statement": "..."
#   }}
# }}
# """
OPENING_PROMPT_TEMPLATE = """
# Role
你是一位**深谋远虑的辩论赛一辩（立论手）**。你不仅文采斐然，更擅长通过精妙的“定义”和“判准”将辩题的讨论范围锁定在对我方有利的领域，让对方无路可走。

# Task
请根据【输入信息】，先进行战略规划（CoT），然后生成一篇逻辑森严、极具统治力的立论陈词。

# 输入信息
- **辩题**: {topic}
- **立场**: {stance}

# 内容要求 (Critical - 必须严格执行)
1. **深度思维链 (CoT) 要求**：
   - **步骤一 [战场切割]**：思考如何定义辩题中的关键词？（例如：将“人性”定义为“社会属性”而非“生物本能”）。**定义必须具有排他性，为我方论证铺路。**
   - **步骤二 [判准确立]**：设立一个对我方最有利的判断标准（例如：“今天我们讨论利弊，不是看对个人的短期利益，而是看对文明的长远存续”）。
   - **步骤三 [论点架构]**：设计2-3个层层递进的论点（逻辑层 -> 现实层 -> 价值层）。

2. **立论陈词 (Opening Statement) 要求**：
   - **开篇立规矩**：开场必须迅速抛出定义和判准，语气要笃定，仿佛这是唯一的真理。
   - **论证有血肉**：论点不能只有干巴巴的逻辑，必须结合具体的**社会现象、数据、历史案例或著名的思想实验**。
   - **修辞与气场**：使用排比、反问、类比等修辞手法。语言要从容大气，不仅要说服评委的脑（逻辑），还要打动评委的心（价值）。
   - **字数控制**：400-500字（立论通常需要铺陈，稍微长一点）。

# JSON输出格式 (Strict JSON Only)
请直接输出合法的 JSON 字符串，**严禁**包含 ```json 或 ``` 等 Markdown 标记。

{{
  "type": "opening_statement",
  "instruction": "你是一位辩论赛的辩手。请根据给定的辩题和你的立场，构建一套逻辑严密的立论体系，通过更有利的定义和判准锁定战场，并发表一段精彩的立论陈词。",
  "input": {{
    "topic": "{topic}",
    "stance": "{stance}"
  }},
  "output": {{
    "cot": "1. [战场切割]: 我将把'{topic}'中的核心词定义为...这样可以排除对方关于...的论述。 \n2. [判准确立]: 今天的判断标准应当是...因为... \n3. [论点规划]: 首先从逻辑上论证...其次从现实层面列举...最后升华到...",
    "opening_statement": "主席，各位好。开宗明义，我们将...定义为...。基于此，判断今天辩题的唯一标准，在于...。\\n\\n我方主张{stance}，理由有三。\\n\\n第一，从逻辑层面看，...。\\n\\n第二，放眼现实，...（此处加入具体案例）。\\n\\n第三，...。\\n\\n综上所述，..."
  }}
}}
"""
REBUTTAL_PROMPT_TEMPLATE = """
# Role
你是一位逻辑如手术刀般精准的**世界级金牌辩手**。你不仅擅长拆解对手的逻辑硬伤，更擅长在逻辑自洽时进行高维度的价值打击。

# Task
请根据输入信息，生成一个用于微调模型的高质量数据样本。

# 输入信息
- **辩题**: {topic}
- **我方立场**: {my_stance}
- **我方核心立论**: {my_opening_summary}
- **对方最新辩词**: {opponent_speech}

# 内容要求 (Critical - 必须严格执行)
1. **深度思维链 (CoT) 要求**：
   请严格按照以下四步进行思考：
   - **1. [靶点锁定]**：原文引用对方最核心的一个观点。
   - **2. [逻辑评估]**：**（关键分支）**
     - *若有硬伤*：指出是滑坡谬误、偷换概念、数据偏差等。
     - *若无硬伤*：指出其底层假设（Premise）的局限性，或其价值取向（Value）的短视。
   - **3. [武器选择]**：选择归谬法、反例法、或者损益比较法（Trade-off）。
   - **4. [回扣立论]**：将反驳落脚点引回我方核心判准。

2. **正式辩词 (Answer) 要求**：
   - **风格**：犀利、紧凑，多用反问句（“难道...”）和排比句。
   - **结构**：驳论（破）+ 立论（立）。先拆掉对方的台，再筑起自己的墙。
   - **字数**：300-400字。

# JSON输出格式 (Strict JSON Only)
请直接输出合法的 JSON 字符串，**严禁**包含 ```json 或 ``` 等 Markdown 标记。

{{
  "type": "rebuttal",
  "instruction": "你是一位辩论赛的辩手。请分析对方的观点，结合我方立论，对其逻辑漏洞或价值缺陷进行有力的反驳。",
  "input": {{
    "topic": "{topic}",
    "my_stance": "{my_stance}",
    "my_opening_statement": "{my_opening_summary}",
    "opponent_speech": "{opponent_speech}"
  }},
  "output": {{
    "cot": "1. [靶点锁定]: 对方核心观点是‘...’\\n2. [逻辑评估]: (情况A/B) 对方逻辑看似自洽，但隐含了一个功利主义的错误假设，即...\\n3. [武器选择]: 我将通过归谬法，指出如果按这个逻辑推演，将导致...\\n4. [回扣立论]: 这反衬出我方坚持的...才是解决问题的根本。",
    "answer": "对方辩友，您刚才口口声声说...，但这背后其实隐藏着一个巨大的逻辑陷阱！\\n\\n您认为...，那按照您的逻辑，岂不是意味着...（归谬）？这显然是荒谬的。\\n\\n事实上，当我们谈论{topic}时，不能只看...，更要看到...。正如我方开篇所言，只有坚持...，才能..."
  }}
}}
"""

# CLOSING_PROMPT_TEMPLATE = """
# # Role
# 你是一位资深的辩论结辩手，擅长价值升华和全场总结。
#
# # Task
# 请根据输入信息，生成一个【总结陈词】的数据样本。
#
# # 输入信息
# - **辩题**: {topic}
# - **我方立场**: {my_stance}
# - **我方核心立论**: {my_opening_summary}
# - **对方刚才的观点**: {opponent_speech}
#
# # 内容要求 (Critical)
# 1. **收束战场**：不要再纠缠细枝末节，要指出对方整场辩论的根本逻辑错误。
# 2. **价值升华**：将辩题提升到哲学、社会或伦理高度。
# 3. **感性号召**：语言要有感染力，金句频出。
#
# # JSON输出格式
# {{
#   "type": "closing_statement",
#   "instruction": "你是一位辩论赛的四辩（结辩）。请根据辩论进程，进行最终的总结陈词，驳斥对方核心逻辑并升华我方价值。",
#   "input": {{
#     "topic": "{topic}",
#     "my_stance": "{my_stance}",
#     "my_opening_statement": "{my_opening_summary}",
#     "opponent_last_argument": "{opponent_speech}"
#   }},
#   "output": {{
#     "cot": "1. [全场归谬]... 2. [价值重申]...",
#     "closing_statement": "..."
#   }}
# }}
# """
CLOSING_PROMPT_TEMPLATE = """
# Role
你是一位**辩论终结者**（四辩）。你不再纠缠于细枝末节的争吵，而是擅长通过重新定义辩题、进行价值排序，将整场辩论收束到我方的逻辑框架中。

# Task
请根据输入信息，生成一个用于微调模型的高质量【总结陈词】数据样本。

# 输入信息
- **辩题**: {topic}
- **我方立场**: {my_stance}
- **我方核心立论**: {my_opening_summary}
- **对方整场主要观点**: {opponent_speech}

# 内容要求 (Critical - 必须严格执行)
1. **深度思维链 (CoT) 要求**：
   - **步骤一 [战场切割]**：识别整场辩论中哪些是对方抛出的“烟雾弹”或次要战场，并在思维中将其剥离（“对方今天一直纠结于...但这是个伪命题”）。
   - **步骤二 [损益比较] (关键)**：使用**“即使……也……” (Even-if)** 的逻辑。承认对方的部分合理性，但论证我方的价值在**长远性、紧迫性或影响范围**上压倒对方。（例如：即使对方讲的效率很重要，但如果没有了公平，效率只是剥削的工具）。
   - **步骤三 [世界观构建]**：描绘“按对方说的做，世界会变成什么样（坏世界）” vs “按我方说的做，世界会变成什么样（好世界）”。

2. **正式辩词 (Answer) 要求**：
   - **高度概括**：不要罗列流水账，要提炼出双方的根本分歧点（“今天双方真正的分歧，其实在于……”）。
   - **感性号召**：结尾必须有金句，要有感染力，甚至带一点悲悯或激昂的情绪。
   - **字数控制**：400字左右（结辩通常比驳论稍长）。

# JSON输出格式 (Strict JSON Only)
请直接输出合法的 JSON 字符串，**严禁**包含 ```json 或 ``` 等 Markdown 标记。

{{
  "type": "closing_statement",
  "instruction": "你是一位辩论赛的四辩（结辩）。请根据辩论进程，进行最终的总结陈词，通过损益比较驳斥对方，并升华我方价值。",
  "input": {{
    "topic": "{topic}",
    "my_stance": "{my_stance}",
    "my_opening_statement": "{my_opening_summary}",
    "opponent_last_argument": "{opponent_speech}"
  }},
  "output": {{
    "cot": "1. [战场切割]: 对方整场都在纠结具体的...，但这偏离了辩题核心... \n2. [损益比较]: 就算承认对方说的...有一定道理，但在我方关注的...面前，那些微不足道。因为... \n3. [世界观对比]: 如果按对方的逻辑，社会将陷入...；而我方主张的...",
    "closing_statement": "主席，各位。今天这场辩论打到最后，我们发现对方辩友始终不敢面对一个核心问题——那就是...。\\n\\n对方一直强调...，好，退一万步讲，即使我们接受您的逻辑，难道我们就应该...吗？不！\\n\\n因为我们看到的不仅仅是...，更是...。如果这个世界真的如对方所愿...，那将是何等的悲哀。因此，我方坚持认为..."
  }}
}}
"""

# ==========================================
# 2. 工具函数
# ==========================================

def extract_json(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def call_llm_api(prompt):
    """
    使用 Google GenAI SDK 调用模型
    """
    max_retries = 3
    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                },
            )
            return response.text
        except Exception as _:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.7,
                    },
                )
                return response.text
            except Exception as _:
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.7,
                        },
                    )
                    return response.text
                except Exception as e:
                    print(f"[API Error] 尝试 {i + 1}/{max_retries}: {e}")
                    time.sleep(5)  # 等待 5 秒重试
    return "{}"


def save_to_jsonl(data, filename="debate_dataset_v2.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ==========================================
# 3. 核心生成逻辑
# ==========================================

def generate_turn(role, topic, stance, context_history, opponent_last_speech=""):
    prompt = ""

    if role == "立论":
        prompt = OPENING_PROMPT_TEMPLATE.format(topic=topic, stance=stance)

    elif role == "反驳":
        my_opening = context_history.get('my_opening', '')[:500]
        prompt = REBUTTAL_PROMPT_TEMPLATE.format(
            topic=topic,
            my_stance=stance,
            my_opening_summary=my_opening,
            opponent_speech=opponent_last_speech
        )

    elif role == "总结":
        my_opening = context_history.get('my_opening', '')[:500]
        prompt = CLOSING_PROMPT_TEMPLATE.format(
            topic=topic,
            my_stance=stance,
            my_opening_summary=my_opening,
            opponent_speech=opponent_last_speech
        )
    print("call gemini...")
    raw_response = call_llm_api(prompt)

    try:
        clean_text = extract_json(raw_response)
        data = json.loads(clean_text, strict=False)
        return data
    except json.JSONDecodeError:
        print(f"!!! JSON 解析失败: {raw_response[:100]}...")
        return None


# ==========================================
# 4. 主程序：双倍轮次 + 总结陈词
# ==========================================

def run_debate_simulation(topic):
    print(f"\n=== 开始辩论生成：{topic} ===")

    # 1. 正方立论
    print(">>> [1/6] 正方立论...")
    pro_open_data = generate_turn("立论", topic, "正方", {})
    if not pro_open_data: return
    pro_open_text = pro_open_data['output']['opening_statement']
    save_to_jsonl(pro_open_data)

    # 2. 反方立论
    print(">>> [2/6] 反方立论...")
    con_open_data = generate_turn("立论", topic, "反方", {})
    if not con_open_data: return
    con_open_text = con_open_data['output']['opening_statement']
    save_to_jsonl(con_open_data)

    # === 自由辩论循环 (2轮) ===

    last_speech = con_open_text  # 初始：正方反驳反方立论

    # 轮次设置：3轮意味着 -> 正方驳, 反方驳, 正方驳, 反方驳
    for round_idx in range(2):
        print(f"\n--- 第 {round_idx + 1} 轮交锋 ---")

        # A. 正方反驳
        print(f">>> 正方反驳 (Round {round_idx + 1})...")
        pro_reb_data = generate_turn("反驳", topic, "正方", {"my_opening": pro_open_text}, last_speech)
        if not pro_reb_data: break
        pro_reb_text = pro_reb_data['output']['answer']
        save_to_jsonl(pro_reb_data)
        last_speech = pro_reb_text  # 更新最新发言

        # B. 反方反驳
        print(f">>> 反方反驳 (Round {round_idx + 1})...")
        con_reb_data = generate_turn("反驳", topic, "反方", {"my_opening": con_open_text}, last_speech)
        if not con_reb_data: break
        con_reb_text = con_reb_data['output']['answer']
        save_to_jsonl(con_reb_data)
        last_speech = con_reb_text  # 更新最新发言

    # === 总结陈词 ===
    print("\n--- 总结陈词阶段 ---")

    # 反方总结 (通常反方先结辩，也可以按规则改)
    print(">>> 反方总结...")
    con_close_data = generate_turn("总结", topic, "反方", {"my_opening": con_open_text}, last_speech)  # 针对正方最后一次反驳
    if con_close_data:
        save_to_jsonl(con_close_data)

    # 正方总结
    print(">>> 正方总结...")
    # 注意：正方总结通常针对反方的总结，或者针对反方最后一次反驳。
    # 这里我们传入反方最后一次反驳的内容作为靶子，或者传入反方的总结(如果你希望正方更有针对性)
    pro_close_data = generate_turn("总结", topic, "正方", {"my_opening": pro_open_text}, last_speech)
    if pro_close_data:
        save_to_jsonl(pro_close_data)

    print(f"=== {topic} 生成结束 ===")


# 执行入口
if __name__ == "__main__":
    # 从JSON文件读取话题列表
    with open('topics.json', 'r', encoding='utf-8') as f:
        topics = json.load(f)  # 直接加载就是话题列表

    for t in topics:
        run_debate_simulation(t)