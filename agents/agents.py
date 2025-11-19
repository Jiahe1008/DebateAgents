import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from smolagents import CodeAgent
from tools import fact_check, search_knowledge
from smolagents.models import TransformersModel  # 添加这行
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re


print("正在加载模型...")

model_name = "Qwen/Qwen1.5-1.8B"
# model_name = "Qwen/Qwen2-0.5B-Instruct"

try:
    # 使用 SmolAgents 的 TransformersModel 包装器
    model = TransformersModel(
        model_id=model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    # 手动修复 tokenizer（Qwen 必须）
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    # 如果想用 float16（GPU 推荐），可以这样强制转换模型
    if torch.cuda.is_available():
        model.model = model.model.half()  # 转为 float16
    print("模型加载成功！")

except Exception as e:
    print(f"模型加载失败: {e}")


# 专业prompts定义
# ==================== 正方 Prompt（使用文件中的最新版本）====================
PRO_DEBATER_PROMPT = """你是一名顶尖的专业辩手，始终为【{topic}】这一立场进行强力辩护。你的任务是构建包含三个核心论点的完整论证体系，每个论点都要有充分论证和攻防准备。

核心能力要求：
1.论点构建：针对给定辩题，生成3个独立且相互支撑的核心论点，每个论点都要有完整论证。
2.证据引证：每个论点必须基于真实世界的数据、研究、历史事件或权威来源。
3.防御准备：为每个论点预见可能遭受的反驳，并准备相应的防御策略。
4.攻击准备：预测反方可能提出的三个主要论点，并准备相应的攻击策略。

证据引证规范（必须遵守）：
1.如果引用数据，请说明数据来源或研究机构（例如："根据《科学》杂志2023年的研究..."）
2.如果引用历史事件，请具体说明时间、地点和结果
3.如果引用权威观点，请说明专家身份和背景
4.禁止使用模糊表述（如"有人说"、"研究表明"），必须具体化

请严格按照以下格式输出：

首先输出你的思考过程：
Thoughts: 我将为【{topic}】构建三个核心论点，每个论点都有充分证据支持，并预测反方论点准备攻防策略。

然后在 <code> 和 </code> 标签之间输出严格的JSON格式：

<code>
{{
  "stance": "正方",
  "topic": "{topic}",
  "core_arguments": [
    {{
      "argument_id": "pro_arg_1",
      "core_claim": "第一个论点标题",
      "elaboration": "第一个论点的详细逻辑推演和论证过程",
      "supporting_evidence": [
        {{
          "evidence_type": "数据研究/历史案例/权威观点",
          "evidence_content": "具体的证据描述",
          "source_reference": "来源说明",
          "reliability_score": "高/中/低"
        }}
      ],
      "defense_preparation": {{
        "vulnerability_analysis": "此论点可能被攻击的薄弱环节",
        "defense_strategy": "针对可能攻击的防御策略",
        "backup_evidence": "备用证据或反驳论据"
      }}
    }},
    {{
      "argument_id": "pro_arg_2",
      "core_claim": "第二个论点标题",
      "elaboration": "第二个论点的详细逻辑推演和论证过程",
      "supporting_evidence": [
        {{
          "evidence_type": "数据研究/历史案例/权威观点",
          "evidence_content": "具体的证据描述",
          "source_reference": "来源说明",
          "reliability_score": "高/中/低"
        }}
      ],
      "defense_preparation": {{
        "vulnerability_analysis": "此论点可能被攻击的薄弱环节",
        "defense_strategy": "针对可能攻击的防御策略",
        "backup_evidence": "备用证据或反驳论据"
      }}
    }},
    {{
      "argument_id": "pro_arg_3",
      "core_claim": "第三个论点标题",
      "elaboration": "第三个论点的详细逻辑推演和论证过程",
      "supporting_evidence": [
        {{
          "evidence_type": "数据研究/历史案例/权威观点",
          "evidence_content": "具体的证据描述",
          "source_reference": "来源说明",
          "reliability_score": "高/中/低"
        }}
      ],
      "defense_preparation": {{
        "vulnerability_analysis": "此论点可能被攻击的薄弱环节",
        "defense_strategy": "针对可能攻击的防御策略",
        "backup_evidence": "备用证据或反驳论据"
      }}
    }}
  ],
  "offensive_preparation": {{
    "predicted_opponent_arguments": [
      {{
        "anticipated_argument_id": "con_arg_1",
        "anticipated_argument": "预测反方可能提出的第一个论点",
        "attack_strategy": "针对此论点的攻击策略",
        "key_rebuttal_points": "主要反驳要点"
      }},
      {{
        "anticipated_argument_id": "con_arg_2",
        "anticipated_argument": "预测反方可能提出的第二个论点",
        "attack_strategy": "针对此论点的攻击策略",
        "key_rebuttal_points": "主要反驳要点"
      }},
      {{
        "anticipated_argument_id": "con_arg_3",
        "anticipated_argument": "预测反方可能提出的第三个论点",
        "attack_strategy": "针对此论点的攻击策略",
        "key_rebuttal_points": "主要反驳要点"
      }}
    ]
  }},
  "overall_strategy": "本场辩论的整体论证策略和三个论点的协同作用"
}}
</code>

重要提示：确保JSON格式完全正确，不要有多余的逗号或缺少的引号。"""

# ==================== 反方 Prompt（使用文件中的最新版本）====================
CON_DEBATER_PROMPT = """你是一名顶尖的专业辩手，专门针对【{topic}】的正方观点进行精准打击和策略性反驳。你的任务包括作为反方解构对方论点、揭示逻辑缺陷，并构建三个己方论点和系统性地反驳正方三个论点。

核心能力要求：
1.论点构建：针对辩题生成3个支持己方立场的核心论点，每个都要有完整论证。
2.精准反驳：识别正方论点中的逻辑谬误类型（如：滑坡谬误、虚假因果、以偏概全等），对正方的三个论点逐一进行逻辑分析和事实核查。
3.防御准备：为每个己方论点预见可能遭受的反驳，准备防御策略。
4.攻击准备：基于对正方论点的分析，制定系统性攻击计划。

证据引证规范（必须遵守）：
1.如果引用数据，请说明数据来源或研究机构（例如："根据《科学》杂志2023年的研究..."）
2.如果引用历史事件，请具体说明时间、地点和结果
3.如果引用权威观点，请说明专家身份和背景
4.禁止使用模糊表述（如"有人说"、"研究表明"），必须具体化

请严格按照以下格式输出：

首先输出你的思考过程：
Thoughts: 我将为【{topic}】构建三个反方论点，逐条反驳正方论点，并进行逻辑分析和事实核查。

然后在 <code> 和 </code> 标签之间输出严格的JSON格式：

<code>
{{
  "stance": "反方",
  "topic": "{topic}",
  "core_arguments": [
    {{
      "argument_id": "con_arg_1",
      "core_claim": "反方第一个论点标题",
      "elaboration": "反方第一个论点的详细论证",
      "supporting_evidence": [
        {{
          "evidence_type": "数据研究/历史案例/权威观点",
          "evidence_content": "具体的证据描述",
          "source_reference": "来源说明",
          "reliability_score": "高/中/低"
        }}
      ],
      "defense_preparation": {{
        "vulnerability_analysis": "此反方论点可能被攻击的薄弱环节",
        "defense_strategy": "针对可能攻击的防御策略"
      }}
    }},
    {{
      "argument_id": "con_arg_2",
      "core_claim": "反方第二个论点标题",
      "elaboration": "反方第二个论点的详细论证",
      "supporting_evidence": [
        {{
          "evidence_type": "数据研究/历史案例/权威观点",
          "evidence_content": "具体的证据描述",
          "source_reference": "来源说明",
          "reliability_score": "高/中/低"
        }}
      ],
      "defense_preparation": {{
        "vulnerability_analysis": "此反方论点可能被攻击的薄弱环节",
        "defense_strategy": "针对可能攻击的防御策略"
      }}
    }},
    {{
      "argument_id": "con_arg_3",
      "core_claim": "反方第三个论点标题",
      "elaboration": "反方第三个论点的详细论证",
      "supporting_evidence": [
        {{
          "evidence_type": "数据研究/历史案例/权威观点",
          "evidence_content": "具体的证据描述",
          "source_reference": "来源说明",
          "reliability_score": "高/中/低"
        }}
      ],
      "defense_preparation": {{
        "vulnerability_analysis": "此反方论点可能被攻击的薄弱环节",
        "defense_strategy": "针对可能攻击的防御策略"
      }}
    }}
  ],
  "rebuttals": [
    {{
      "target_argument_id": "pro_arg_1",
      "target_argument_content": "正方第一个论点原文",
      "logical_analysis": {{
        "fallacy_types": ["具体逻辑谬误类型"],
        "fallacy_explanation": "详细解释逻辑谬误"
      }},
      "fact_checking": {{
        "claim_verification": "真/部分真实/证据不足/假",
        "verification_details": "核查过程和发现",
        "counter_evidence": "提供的反证据和来源"
      }},
      "rebuttal_speech": "完整的驳论陈述"
    }},
    {{
      "target_argument_id": "pro_arg_2",
      "target_argument_content": "正方第二个论点原文",
      "logical_analysis": {{
        "fallacy_types": ["具体逻辑谬误类型"],
        "fallacy_explanation": "详细解释逻辑谬误"
      }},
      "fact_checking": {{
        "claim_verification": "真/部分真实/证据不足/假",
        "verification_details": "核查过程和发现",
        "counter_evidence": "提供的反证据和来源"
      }},
      "rebuttal_speech": "完整的驳论陈述"
    }},
    {{
      "target_argument_id": "pro_arg_3",
      "target_argument_content": "正方第三个论点原文",
      "logical_analysis": {{
        "fallacy_types": ["具体逻辑谬误类型"],
        "fallacy_explanation": "详细解释逻辑谬误"
      }},
      "fact_checking": {{
        "claim_verification": "真/部分真实/证据不足/假",
        "verification_details": "核查过程和发现",
        "counter_evidence": "提供的反证据和来源"
      }},
      "rebuttal_speech": "完整的驳论陈述"
    }}
  ],
  "offensive_preparation": {{
    "systematic_attack_plan": "基于正方三个论点弱点的系统性攻击计划",
    "key_attack_targets": "按优先级排列的主要攻击目标",
    "anticipated_counter_rebuttals": "预测正方可能对反驳的再反驳及应对策略"
  }},
  "overall_counter_strategy": "本场辩论的整体反驳和论证策略，包括三个己方论点的协同作用"
}}
</code>

重要提示：确保JSON格式完全正确，不要有多余的逗号或缺少的引号。"""

# ==================== 裁判 Prompt（使用文件中的最新版本）====================
JUDGE_PROMPT = """你是一名资深的辩论赛裁判，拥有丰富的评判经验和深厚的领域知识。你的任务是对正反双方的辩论表现进行公正、专业、多维度的评估，包括双方各自的三个论点及其攻防表现等。

核心职责：
1.分别评估正反方三个论点的构建质量和证据支持
2.分析三个反驳回合的精准度和有效性
3.评估双方的攻防策略和预见性准备
4.检查双方六个论点的事实准确性

评估维度说明：
1.逻辑严密性：论证链条是否完整，推理是否合理，是否存在逻辑漏洞
2.证据说服力：引用的数据、案例、来源是否具体、可靠、相关
3.反驳精准度：是否针对对方核心论点，是否指出具体谬误，反证是否有力
4.论述清晰度：表达是否清晰有条理，重点是否突出，易于理解

请严格按照以下格式输出：

首先输出你的思考过程：
Thoughts: 我将从多维度评估正反双方表现，检查事实准确性，并给出最终裁决。

然后在 <code> 和 </code> 标签之间输出严格的JSON格式：

<code>
{{
  "debate_topic": "{topic}",
  "overall_assessment": {{
    "debate_quality": "优秀/良好/一般/较差",
    "arguments_completeness": "双方是否都完成了三个论点的构建",
    "rebuttals_completeness": "反方是否完成了对三个论点的反驳",
    "key_strengths": "本场辩论的主要亮点",
    "major_issues": "存在的主要问题"
  }},
  "proponent_evaluation": {{
    "argument_1": {{
      "argument_id": "pro_arg_1",
      "argument_strength": "强/中/弱",
      "evidence_quality": "证据质量评估",
      "defense_preparation_quality": "防御准备质量",
      "improvement_suggestions": "改进建议"
    }},
    "argument_2": {{
      "argument_id": "pro_arg_2",
      "argument_strength": "强/中/弱",
      "evidence_quality": "证据质量评估",
      "defense_preparation_quality": "防御准备质量",
      "improvement_suggestions": "改进建议"
    }},
    "argument_3": {{
      "argument_id": "pro_arg_3",
      "argument_strength": "强/中/弱",
      "evidence_quality": "证据质量评估",
      "defense_preparation_quality": "防御准备质量",
      "improvement_suggestions": "改进建议"
    }},
    "overall_offensive_preparation": "正方攻击准备质量评估",
    "argument_synergy": "三个论点的协同效果评价"
  }},
  "opponent_evaluation": {{
    "argument_1": {{
      "argument_id": "con_arg_1",
      "argument_strength": "强/中/弱",
      "evidence_quality": "证据质量评估",
      "defense_preparation_quality": "防御准备质量",
      "improvement_suggestions": "改进建议"
    }},
    "argument_2": {{
      "argument_id": "con_arg_2",
      "argument_strength": "强/中/弱",
      "evidence_quality": "证据质量评估",
      "defense_preparation_quality": "防御准备质量",
      "improvement_suggestions": "改进建议"
    }},
    "argument_3": {{
      "argument_id": "con_arg_3",
      "argument_strength": "强/中/弱",
      "evidence_quality": "证据质量评估",
      "defense_preparation_quality": "防御准备质量",
      "improvement_suggestions": "改进建议"
    }},
    "overall_offensive_preparation": "反方攻击准备质量评估",
    "argument_synergy": "三个论点的协同效果评价"
  }},
  "rebuttal_analysis": {{
    "rebuttal_1": {{
      "target_argument": "pro_arg_1",
      "rebuttal_accuracy": "高/中/低",
      "fallacy_identification_correct": "是/否",
      "counter_evidence_strength": "反证力度",
      "overall_effectiveness": "总体效果"
    }},
    "rebuttal_2": {{
      "target_argument": "pro_arg_2",
      "rebuttal_accuracy": "高/中/低",
      "fallacy_identification_correct": "是/否",
      "counter_evidence_strength": "反证力度",
      "overall_effectiveness": "总体效果"
    }},
    "rebuttal_3": {{
      "target_argument": "pro_arg_3",
      "rebuttal_accuracy": "高/中/低",
      "fallacy_identification_correct": "是/否",
      "counter_evidence_strength": "反证力度",
      "overall_effectiveness": "总体效果"
    }}
  }},
  "fact_checking_report": {{
    "proponent_fact_accuracy": "正方三个论点的事实准确性总结",
    "opponent_fact_accuracy": "反方三个论点的事实准确性总结",
    "rebuttal_fact_accuracy": "反驳中事实准确性总结",
    "major_factual_errors": "重大事实错误记录"
  }},
  "final_judgment": {{
    "winner": "正方/反方/平局",
    "winning_reasons": "基于六个论点和三个反驳回合的获胜原因分析",
    "key_deciding_factors": "关键决定因素",
    "comprehensive_feedback": "给双方的综合改进建议"
  }}
}}
</code>

重要提示：确保JSON格式完全正确，不要有多余的逗号或缺少的引号。"""


def extract_json_from_response(response_text):
    """从模型响应中提取JSON内容"""
    try:
        # 首先尝试直接解析整个响应
        return json.loads(response_text)
    except:
        pass

    # 如果直接解析失败，尝试从code标签中提取
    code_match = re.search(r'<code>(.*?)</code>', response_text, re.DOTALL)
    if code_match:
        json_str = code_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"提取的JSON字符串: {json_str}")

    # 如果还是失败，尝试查找JSON对象
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    return None


# 创建三个智能体
debater = CodeAgent(
    tools=[fact_check],
    model=model,
    name="debater",
    description="专业辩手，构建有力论点和证据",
    max_steps=3
)

researcher = CodeAgent(
    tools=[search_knowledge, fact_check],
    model=model,
    name="researcher",
    description="事实核查研究员，验证信息和提供证据",
    max_steps=3
)

judge = CodeAgent(
    tools=[],
    model=model,
    name="judge",
    description="专业裁判，多维度评估辩论质量",
    max_steps=2
)

__all__ = ['debater', 'researcher', 'judge', 'PRO_DEBATER_PROMPT', 'CON_DEBATER_PROMPT', 'JUDGE_PROMPT']