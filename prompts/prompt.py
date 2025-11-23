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
os.environ["GOOGLE_API_KEY"] = "AIzaSyDoXp7XJ3MBFc0M3wepFyDMwSgDNhlwpSg" 
API_KEY = os.getenv("GOOGLE_API_KEY")

# 初始化 Client
client = genai.Client(api_key=API_KEY)

# ==========================================
# 1. 定义高约束 Prompt 模板 (新增总结陈词)
# ==========================================

OPENING_PROMPT_TEMPLATE = """
# Role
你是一位世界级的辩论教练和金牌撰稿人。

# Task
请根据提供的【输入信息】，生成一个用于微调模型的数据样本。该样本必须严格遵守指定的 JSON 格式。

# 输入信息
- **辩题**: {topic}
- **立场**: {stance}

# 内容要求 (Critical)
1. **结构完整**：[问候] -> [核心定义] -> [判准] -> [分论点] -> [总结]。
2. **判准明确**：必须明确提出判断标准。
3. **语言风格**：气势磅礴，逻辑自洽，使用标准辩论语言。

# JSON输出格式
{{
  "type": "opening_statement",
  "instruction": "你是一位辩论赛的辩手。请根据给定的辩题和你的立场，构建一套逻辑严密的立论体系，明确定义和判准，并发表一段精彩的立论陈词。",
  "input": {{
    "topic": "{topic}",
    "stance": "{stance}"
  }},
  "output": {{
    "opening_statement": "..."
  }}
}}
"""

REBUTTAL_PROMPT_TEMPLATE = """
# Role
你是一位逻辑敏锐的辩论高手，擅长捕捉对手逻辑漏洞。

# Task
请根据输入信息，生成包含“思维链 (CoT)”和“正式反驳”的 JSON 数据样本。

# 输入信息
- **辩题**: {topic}
- **我方立场**: {my_stance}
- **我方核心立论**: {my_opening_summary}
- **对方最新辩词**: {opponent_speech}

# 内容要求 (Critical)
1. **思维链 (CoT)**: 识别对方逻辑谬误（偷换概念、滑坡谬误等），制定反击策略。
2. **正式辩词**: 基于 CoT 展开，字数 300 字左右，犀利且有理有据。不要自说自话，必须回击对方刚才的观点。

# JSON输出格式
{{
  "type": "rebuttal",
  "instruction": "你是一位辩论赛的辩手。请结合我方立论，敏锐地指出对方刚才发言中的逻辑谬误，并进行有力的回击。",
  "input": {{
    "topic": "{topic}",
    "my_stance": "{my_stance}",
    "my_opening_statement": "{my_opening_summary}",
    "opponent_speech": "{opponent_speech}"
  }},
  "output": {{
    "cot": "1. [谬误识别]... 2. [策略]... 3. [回扣]...",
    "answer": "..."
  }}
}}
"""

CLOSING_PROMPT_TEMPLATE = """
# Role
你是一位资深的辩论结辩手，擅长价值升华和全场总结。

# Task
请根据输入信息，生成一个【总结陈词】的数据样本。

# 输入信息
- **辩题**: {topic}
- **我方立场**: {my_stance}
- **我方核心立论**: {my_opening_summary}
- **对方刚才的观点**: {opponent_speech}

# 内容要求 (Critical)
1. **收束战场**：不要再纠缠细枝末节，要指出对方整场辩论的根本逻辑错误。
2. **价值升华**：将辩题提升到哲学、社会或伦理高度。
3. **感性号召**：语言要有感染力，金句频出。

# JSON输出格式
{{
  "type": "closing_statement",
  "instruction": "你是一位辩论赛的四辩（结辩）。请根据辩论进程，进行最终的总结陈词，驳斥对方核心逻辑并升华我方价值。",
  "input": {{
    "topic": "{topic}",
    "my_stance": "{my_stance}",
    "my_opening_statement": "{my_opening_summary}",
    "opponent_last_argument": "{opponent_speech}"
  }},
  "output": {{
    "cot": "1. [全场归谬]... 2. [价值重申]...",
    "closing_statement": "..."
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
            # 注意：Gemini 2.5 目前可能是 preview，如果报错请改回 gemini-1.5-pro
            # response = client.models.generate_content(
            #     model='gemini-2.0-flash', # 建议先用 2.0 flash 测试，正式跑用 pro
            #     contents=prompt,
            #     config=types.GenerateContentConfig(
            #         response_mime_type='application/json',
            #         temperature=0.7
            #     )
            # )
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
            print(f"[API Error] 尝试 {i+1}/{max_retries}: {e}")
            time.sleep(5) # 等待 5 秒重试
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
        data = json.loads(clean_text)
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
    
    last_speech = con_open_text # 初始：正方反驳反方立论
    
    # 轮次设置：2轮意味着 -> 正方驳, 反方驳, 正方驳, 反方驳
    for round_idx in range(2):
        print(f"\n--- 第 {round_idx+1} 轮交锋 ---")
        
        # A. 正方反驳
        print(f">>> 正方反驳 (Round {round_idx+1})...")
        pro_reb_data = generate_turn("反驳", topic, "正方", {"my_opening": pro_open_text}, last_speech)
        if not pro_reb_data: break
        pro_reb_text = pro_reb_data['output']['answer']
        save_to_jsonl(pro_reb_data)
        last_speech = pro_reb_text # 更新最新发言
        
        # B. 反方反驳
        print(f">>> 反方反驳 (Round {round_idx+1})...")
        con_reb_data = generate_turn("反驳", topic, "反方", {"my_opening": con_open_text}, last_speech)
        if not con_reb_data: break
        con_reb_text = con_reb_data['output']['answer']
        save_to_jsonl(con_reb_data)
        last_speech = con_reb_text # 更新最新发言

    # === 总结陈词 ===
    print("\n--- 总结陈词阶段 ---")
    
    # 反方总结 (通常反方先结辩，也可以按规则改)
    print(">>> 反方总结...")
    con_close_data = generate_turn("总结", topic, "反方", {"my_opening": con_open_text}, last_speech) # 针对正方最后一次反驳
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
    topics = [
        "人工智能对人类发展利大于弊",
        "当今社会更需要通才还是专才",
        "顺境还是逆境更有利于人的成长"
    ]
    for t in topics:
        run_debate_simulation(t)
        time.sleep(1) # 避免触发限流