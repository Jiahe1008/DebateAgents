#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
将新版结构化 seed_debates.jsonl 转换为标准训练格式 (instruction-input-output)
适配 qwen3-max + 三方 Prompt 生成的嵌套 JSON 结构
支持最多4轮辩论（1立论+3交锋），每轮 input 保留完整上下文。
"""

import json
import os


def format_argument_compat(arg: dict) -> str:
    # 新格式：来自新版 data_generator
    if "argument_point" in arg or "evidence_or_example" in arg:
        point = str(arg.get("argument_point", "")).strip()
        evidence = str(arg.get("evidence_or_example", "")).strip()
        impact = str(arg.get("impact_analysis", "")).strip()
        parts = [point]
        if evidence:
            parts.append(f"（{evidence}）")
        if impact:
            parts.append(f"影响：{impact}")
        return " ".join(p for p in parts if p)
    # 老格式：原有结构
    claim = arg.get("core_claim", "") or arg.get("claim", "")
    evidence_texts = []
    for ev in arg.get("supporting_evidence", []):
        content = ev.get("evidence_content", "").strip()
        source = ev.get("source_reference", "").strip()
        if content:
            if source:
                evidence_texts.append(f"（{source}）{content}")
            else:
                evidence_texts.append(content)
    evidence_str = "；".join(evidence_texts) if evidence_texts else ""
    impact = arg.get("impact_analysis", "") or arg.get("impact", "")
    parts = [p for p in [claim, evidence_str, f"影响：{impact}" if impact else ""] if p]
    return " ".join(parts)


def format_opening_for_context(side_data: dict) -> str:
    parts = []
    core_crit = side_data.get("core_criterion", {})
    if core_crit and isinstance(core_crit, dict):
        crit_stmt = core_crit.get("criterion_statement") or core_crit.get("view")
        if crit_stmt:
            parts.append(f"【判准】{crit_stmt}")
    defs = side_data.get("definition_of_terms", {})
    if isinstance(defs, dict) and defs:
        defs_str = "；".join(f"{k}：{v}" for k, v in defs.items())
        parts.append(f"定义：{defs_str}")
    args = side_data.get("core_arguments", [])
    if args:
        arg_texts = [format_argument_compat(arg) for arg in args if isinstance(arg, dict)]
        parts.append("论点：\n" + "\n".join(arg_texts))
    return "\n".join(parts) if parts else "（无内容）"


def format_rebuttal_for_context(rebuttal: dict) -> str:
    texts = []
    defense = rebuttal.get("defense_response", {})
    if isinstance(defense, dict):
        if defense.get("logical_rebuttal"):
            texts.append(str(defense["logical_rebuttal"]))
    cont_reb = rebuttal.get("continued_rebuttal", {})
    if isinstance(cont_reb, dict):
        counter = cont_reb.get("counter_arguments")
        if isinstance(counter, list):
            texts.extend(format_argument_compat(arg) for arg in counter if isinstance(arg, dict))
        elif isinstance(counter, str):
            texts.append(counter)
    deep_arg = rebuttal.get("argument_development", {})
    if isinstance(deep_arg, dict) and deep_arg.get("deepened_arguments"):
        texts.append(str(deep_arg["deepened_arguments"]))
    summary = rebuttal.get("round_summary")
    if summary:
        texts.append(str(summary))
    return "\n".join(texts) if texts else "（无内容）"


def build_debate_context(debate_history: list, up_to_round: int, topic: str) -> str:
    """
    构建从第1轮到 up_to_round 轮（含）的自然语言上下文
    up_to_round: 整数，表示包含到第几轮（用于 input，通常是当前轮-1）
    """
    lines = [f"辩题：{topic}\n"]
    for rnd in debate_history[:up_to_round]:
        rn = rnd["round_number"]
        lines.append(f"--- 第{rn}轮 ---")
        if rn == 1:
            pro_text = format_opening_for_context(rnd["proponent"])
            con_text = format_opening_for_context(rnd["opponent"])
            lines.append(f"【正方立论】\n{pro_text}")
            lines.append(f"【反方立论】\n{con_text}")
        else:
            pro_text = format_rebuttal_for_context(rnd["proponent"])
            con_text = format_rebuttal_for_context(rnd["opponent"])
            lines.append(f"【正方发言】\n{pro_text}")
            lines.append(f"【反方发言】\n{con_text}")
    return "\n\n".join(lines).strip()


def extract_rebuttal_output(rebuttal: dict) -> str:
    """从反驳 JSON 中提取自然语言输出"""
    parts = []
    defense = rebuttal.get("defense_response", {})
    if isinstance(defense, dict) and defense.get("logical_rebuttal"):
        parts.append(str(defense["logical_rebuttal"]))
    cont_reb = rebuttal.get("continued_rebuttal", {})
    if isinstance(cont_reb, dict):
        counter = cont_reb.get("counter_arguments")
        if isinstance(counter, list):
            parts.extend(format_argument_compat(arg) for arg in counter if isinstance(arg, dict))
        elif isinstance(counter, str):
            parts.append(counter)
    deep_arg = rebuttal.get("argument_development", {})
    if isinstance(deep_arg, dict) and deep_arg.get("deepened_arguments"):
        parts.append(str(deep_arg["deepened_arguments"]))
    summary = rebuttal.get("round_summary", "")
    if summary:
        parts.append(str(summary))
    return "；".join(parts).strip()


def convert_seed_to_training(seed_path: str, output_path: str):
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"种子文件不存在: {seed_path}")

    training_samples = []

    with open(seed_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                debate = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_num} 行 JSON 解析失败: {e}")
                continue

            topic = debate.get("topic", "未知辩题")
            debate_history = debate.get("debate_history", [])

            if not debate_history:
                # 老格式：跳过或按需处理（此处暂略）
                continue
            else:
                total_rounds = len(debate_history)

                # === 第1轮：立论 ===
                round1 = debate_history[0]
                pro_stmt = round1.get("proponent", {})
                con_stmt = round1.get("opponent", {})

                def extract_full_opening(side_data: dict) -> str:
                    parts = []
                    # 从新格式提取判准
                    core_crit = side_data.get("core_criterion", {})
                    if core_crit and isinstance(core_crit, dict):
                        crit_stmt = core_crit.get("criterion_statement") or core_crit.get("view")
                        if crit_stmt:
                            parts.append(f"【判准】{crit_stmt}")
                    defs = side_data.get("definition_of_terms", {})
                    if isinstance(defs, dict) and defs:
                        defs_str = "；".join(f"{k}：{v}" for k, v in defs.items())
                        parts.append(f"【定义】{defs_str}")
                    args = side_data.get("core_arguments", [])
                    if args:
                        arg_texts = [format_argument_compat(arg) for arg in args if isinstance(arg, dict)]
                        if arg_texts:
                            parts.append("【论点】\n" + "\n".join(arg_texts))
                    return "\n\n".join(parts).strip()

                pro_output = extract_full_opening(pro_stmt)
                if pro_output:
                    training_samples.append({
                        "instruction": f"作为正方，请就“{topic}”进行立论。",
                        "input": "",
                        "output": pro_output
                    })

                con_output = extract_full_opening(con_stmt)
                if con_output:
                    training_samples.append({
                        "instruction": f"作为反方，请就“{topic}”进行立论。",
                        "input": "",
                        "output": con_output
                    })

                # === 第2至第total_rounds轮：反驳（保留完整上下文）===
                for i in range(1, total_rounds):
                    current_round_num = i + 1
                    rnd = debate_history[i]

                    # 构建完整上下文（第1轮到第i轮，即当前轮之前）
                    full_context = build_debate_context(debate_history, up_to_round=i, topic=topic)

                    # 正方本轮
                    pro_reb = rnd.get("proponent", {})
                    reb_output = extract_rebuttal_output(pro_reb)
                    if reb_output:
                        training_samples.append({
                            "instruction": f"作为正方，请在第{current_round_num}轮就“{topic}”继续辩论。",
                            "input": full_context,
                            "output": reb_output
                        })

                    # 反方本轮
                    con_reb = rnd.get("opponent", {})
                    reb_output = extract_rebuttal_output(con_reb)
                    if reb_output:
                        training_samples.append({
                            "instruction": f"作为反方，请在第{current_round_num}轮就“{topic}”继续辩论。",
                            "input": full_context,
                            "output": reb_output
                        })

            # === 裁判样本 ===
            judgment = debate.get("judgment", {})
            if not isinstance(judgment, dict):
                judgment = {}

            judge_output = ""

            # 优先尝试嵌套路径（来自裁判-prompt.txt 的标准输出结构）
            final_judgment = judgment.get("final_judgment", {})
            if isinstance(final_judgment, dict):
                # 按优先级拼接关键字段
                parts = []
                if final_judgment.get("winner"):
                    parts.append(f"胜方：{final_judgment['winner']}")
                if final_judgment.get("winning_reasons"):
                    parts.append(f"获胜理由：{final_judgment['winning_reasons']}")
                if final_judgment.get("key_deciding_factors"):
                    factors = final_judgment["key_deciding_factors"]
                    if isinstance(factors, list):
                        factors_str = "；".join(str(f) for f in factors)
                    else:
                        factors_str = str(factors)
                    parts.append(f"关键决定因素：{factors_str}")
                if final_judgment.get("comprehensive_feedback"):
                    parts.append(f"综合建议：{final_judgment['comprehensive_feedback']}")
                if parts:
                    judge_output = "\n".join(parts)

            # 如果嵌套结构没取到，再回退到扁平字段（兼容旧格式）
            if not judge_output.strip():
                candidate_fields = [
                    "evaluation_summary", "final_decision", "overall_evaluation",
                    "conclusion", "verdict", "裁判总结", "reasoning", "reason", "analysis"
                ]
                for field in candidate_fields:
                    val = judgment.get(field)
                    if val and str(val).strip():
                        judge_output = str(val).strip()
                        break

            # 只要非空就加入
            if judge_output.strip():
                training_samples.append({
                    "instruction": f"作为资深辩论裁判，请对“{topic}”的辩论进行专业点评。",
                    "input": "",
                    "output": judge_output.strip()
                })

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 转换完成！共 {len(training_samples)} 条训练样本 -> {output_path}")
    print(f"   - 正方样本: {sum(1 for s in training_samples if '正方' in s['instruction'])}")
    print(f"   - 反方样本: {sum(1 for s in training_samples if '反方' in s['instruction'])}")
    print(f"   - 裁判样本: {sum(1 for s in training_samples if '裁判' in s['instruction'])}")


if __name__ == "__main__":
    convert_seed_to_training(
        seed_path="data/seed_debates.jsonl",
        output_path="data/training_data.jsonl"
    )