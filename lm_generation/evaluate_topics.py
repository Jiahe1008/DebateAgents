# lm_generation/evaluate_topics.py

import re
import argparse
from pathlib import Path
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False

def standardize_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return ""
    topic = re.sub(r"^[^\w\u4e00-\u9fa5“”‘’]+", "", topic)
    topic = re.sub(r"[^\w\u4e00-\u9fa5“”‘’]+$", "", topic)
    if topic.startswith("与人相处，"):
        core = topic.replace("与人相处，", "").strip()
        if "/" in core:
            p1, p2 = core.split("/", 1)
            return f"与人相处，应该{p1.strip()}还是{p2.strip()}？"
    if not topic.endswith(("？", "?")):
        topic = topic.rstrip("。！!") + "？"
    return topic

def normalize_for_dedup(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fa5]", "", s).lower()

def semantic_dedup(topics: List[str], threshold: float = 0.85) -> List[str]:
    if not HAS_EMBEDDING or len(topics) < 2:
        return topics
    print("🔍 计算语义相似度（首次运行会下载模型）...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    emb = model.encode(topics, show_progress_bar=True)
    sim = cosine_similarity(emb)
    keep = [True] * len(topics)
    for i in range(len(topics)):
        if not keep[i]: continue
        for j in range(i + 1, len(topics)):
            if sim[i][j] > threshold:
                keep[j] = False
    return [topics[i] for i in range(len(topics)) if keep[i]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="debate_topics.txt", help="辩题文件路径")
    args = parser.parse_args()

    file_path = Path(__file__).parent / args.file
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    # 读取原始内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    raw_topics = []
    comments = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comments.append(line)  # 保留注释
        elif stripped:
            raw_topics.append(stripped)

    print(f"📥 读取 {len(raw_topics)} 条辩题")

    # 标准化
    standardized = [standardize_topic(t) for t in raw_topics]
    standardized = [t for t in standardized if t]

    # 精确去重（保持顺序）
    seen = set()
    unique = []
    for t in standardized:
        key = normalize_for_dedup(t)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    print(f"🔁 精确去重后: {len(unique)} 条")

    # 语义去重
    cleaned = semantic_dedup(unique)
    print(f"🧠 语义去重后: {len(cleaned)} 条")

    # 质量提示
    bad_len = [t for t in cleaned if len(t) < 6 or len(t) > 60]
    no_q = [t for t in cleaned if "？" not in t and "?" not in t]
    if bad_len or no_q:
        print("⚠️  警告:")
        if bad_len: print(f"  - 长度异常: {len(bad_len)} 条")
        if no_q: print(f"  - 缺少问号: {len(no_q)} 条")

    # 写回原文件（保留注释 + 新内容）
    with open(file_path, 'w', encoding='utf-8') as f:
        # 先写注释（如果有）
        for comment in comments:
            f.write(comment)
        if comments and cleaned:
            f.write('\n')
        # 再写清洗后的辩题
        for topic in cleaned:
            f.write(topic + '\n')

    print(f"\n✅ 成功清洗并更新: {file_path}")

if __name__ == "__main__":
    main()