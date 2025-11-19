#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
辩论数据全链路质量评估脚本
- 基础统计
- 完整性检查
- 多样性检测（辩题/论点）
- 有效性验证（轻量模型模拟）
- 人工抽查
"""

import jsonlines
import os
import re
import random
from collections import Counter
import json

# 可选：安装 sentence-transformers 后启用多样性检测
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False
    print("⚠️ 未安装 sentence-transformers，跳过语义多样性检测")

# 可选：有效性验证（需配置 DashScope API）
USE_VALIDATION = True  # 设为 True 并配置 API 后启用
if USE_VALIDATION:
    from lm_generation.dashscope_api import call_api


def extract_topic_from_instruction(instr: str) -> str:
    """从 instruction 中提取辩题"""
    match = re.search(r"“([^”]+)”|'([^']+)’", instr)
    if match:
        return match.group(1) or match.group(2)
    return instr


def check_completeness(samples):
    """完整性检查"""
    stats = {
        "total": len(samples),
        "empty_output": 0,
        "missing_criteria": 0,
        "role_counts": Counter(),
        "round_distribution": Counter()
    }

    for s in samples:
        instr = s.get("instruction", "")
        output = s.get("output", "").strip()

        # 角色统计
        if "正方" in instr:
            role = "正方"
        elif "反方" in instr:
            role = "反方"
        elif "裁判" in instr:
            role = "裁判"
        else:
            role = "其他"
        stats["role_counts"][role] += 1

        # 轮次统计
        round_match = re.search(r"第(\d+)轮", instr)
        if round_match:
            stats["round_distribution"][int(round_match.group(1))] += 1
        elif "立论" in instr:
            stats["round_distribution"][1] += 1

        # 空输出
        if not output:
            stats["empty_output"] += 1

        # 判准检查（仅对立论）
        if role in ["正方", "反方"] and ("立论" in instr or "第1轮" in instr):
            if not re.search(r"[【\[]?判准[】\]]?", output):
                stats["missing_criteria"] += 1

    return stats


def check_diversity(samples):
    """多样性检测"""
    results = {"topic_duplication": 0, "high_similarity_topics": []}

    # 提取辩题
    topics = []
    topic_to_samples = {}
    for s in samples:
        if "立论" in s.get("instruction", "") or "第1轮" in s.get("instruction", ""):
            topic = extract_topic_from_instruction(s["instruction"])
            if topic:
                topics.append(topic)
                topic_to_samples[topic] = s

    if not topics:
        return results

    # 精确重复
    exact_dups = len(topics) - len(set(topics))
    results["topic_duplication"] = exact_dups

    # 语义相似度（如果可用）
    if HAS_EMBEDDING and len(topics) > 1:
        unique_topics = list(set(topics))
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(unique_topics, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        high_sim_pairs = []
        for i in range(len(unique_topics)):
            for j in range(i + 1, len(unique_topics)):
                if sim_matrix[i][j] > 0.85:
                    high_sim_pairs.append((unique_topics[i], unique_topics[j], sim_matrix[i][j]))
        results["high_similarity_topics"] = high_sim_pairs

    return results


def validate_effectiveness(samples, sample_size=10):
    """有效性验证（轻量模型模拟）"""
    if not USE_VALIDATION:
        return {"skipped": True}

    validated = 0
    valid_count = 0
    examples = []

    # 随机抽样
    sampled = random.sample(samples, min(sample_size, len(samples)))
    for s in sampled:
        if s.get("input") and len(s["input"]) > 50:  # 只验证有上下文的样本
            try:
                simulated = call_api(
                    prompt=f"{s['input']}\n\n{s['instruction']}",
                    max_tokens=300,
                    temperature=0.3,
                    model="deepseek-v3.2-exp"
                )
                # 简单检查：是否非空且长度合理
                if simulated and len(simulated.strip()) > 20:
                    valid_count += 1
                validated += 1
                examples.append({
                    "original_output": s["output"][:100],
                    "simulated_output": simulated[:100] if simulated else "None"
                })
            except Exception as e:
                print(f"验证失败: {e}")

    return {
        "validated_samples": validated,
        "valid_ratio": valid_count / validated if validated > 0 else 0,
        "examples": examples
    }


def generate_human_review_sample(samples, output_dir="review", sample_size=30):
    """生成人工审核样本"""
    os.makedirs(output_dir, exist_ok=True)
    sampled = random.sample(samples, min(sample_size, len(samples)))
    review_path = os.path.join(output_dir, "human_review_sample.jsonl")
    with open(review_path, 'w', encoding='utf-8') as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return review_path


def main(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    with jsonlines.open(file_path) as f:
        samples = list(f)

    print(f"📊 全链路数据质量评估: {file_path}")
    print("=" * 50)

    # 1. 完整性检查
    comp = check_completeness(samples)
    print(f"✅ 总样本数: {comp['total']}")
    print(f"\n🧮 角色分布:")
    for role, count in comp['role_counts'].most_common():
        print(f"  - {role}: {count} ({count/comp['total']*100:.1f}%)")
    
    print(f"\n📈 轮次分布:")
    for rnd in sorted(comp['round_distribution']):
        print(f"  - 第{rnd}轮: {comp['round_distribution'][rnd]}")

    print(f"\n⚠️ 完整性问题:")
    print(f"  - 空输出: {comp['empty_output']}")
    print(f"  - 立论缺失判准: {comp['missing_criteria']}")

    # 2. 多样性检测
    div = check_diversity(samples)
    print(f"\n🔍 多样性分析:")
    print(f"  - 精确重复辩题: {div['topic_duplication']}")
    if div.get("high_similarity_topics"):
        print(f"  - 高相似度辩题对: {len(div['high_similarity_topics'])}")
        for t1, t2, sim in div["high_similarity_topics"][:3]:
            print(f"    • '{t1}' ↔ '{t2}' (相似度: {sim:.2f})")

    # 3. 有效性验证
    val = validate_effectiveness(samples)
    if not val.get("skipped"):
        print(f"\n🧪 有效性验证 (抽样{val['validated_samples']}):")
        print(f"  - 模拟成功率: {val['valid_ratio']:.1%}")
    else:
        print(f"\n🧪 有效性验证: 已跳过 (USE_VALIDATION=False)")

    # 4. 生成人工审核样本
    review_path = generate_human_review_sample(samples)
    print(f"\n👀 人工审核样本已生成: {review_path}")

    print("\n" + "=" * 50)
    print("✅ 评估完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/training_data.jsonl", help="训练数据文件路径")
    args = parser.parse_args()
    main(args.file)