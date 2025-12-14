import json
import re
import time
import os
import random
from google import genai
from google.genai import types
from tqdm import tqdm
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
# 定义 10 个领域，每个领域生成 50 个
# ==========================================
CATEGORIES = [
    "科技与人工智能 (例如：AI取代人类、算法偏见、基因编辑)",
    "教育与成长 (例如：内卷、双减、通才专才、挫折教育)",
    "社会与民生 (例如：老龄化、延迟退休、性别议题、贫富差距)",
    "法律与正义 (例如：死刑废除、隐私权、网络暴力实名制)",
    "经济与职场 (例如：996制度、基本收入、消费主义、零工经济)",
    "文化与价值观 (例如：传统文化、饭圈文化、审美多元化)",
    "国际关系与人类命运 (例如：全球化、太空竞赛、难民问题)",
    "环境与可持续发展 (例如：核能、碳中和、动物权利)",
    "哲学与脑洞 (例如：人性本善、特修斯之船、缸中之脑)",
    "娱乐与生活 (例如：短视频、电子游戏、网红经济)"
]


def generate_topics_by_category(category, count=50):
    prompt = f"""
    你是一个专业的辩论赛出题人。
    请生成 {count} 个关于【{category}】的中文辩论题目。

    要求：
    1. 格式必须是标准的辩论赛辩题（通常包含“利大于弊”、“应该/不应该”、“是/不是”）。
    2. 题目要有可辩性，双方都有理可据。
    3. 语言简洁有力。
    4. 不要带编号，直接返回一个 JSON 字符串列表。

    输出示例：
    ["人工智能对人类发展利大于弊", "当今社会更需要通才"]
    """

    retries = 3
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                },
            )
            topics = json.loads(response.text)
            if isinstance(topics, list):
                return topics
        except Exception as e:
            print(f"  - 生成失败，重试中 ({i + 1}/{retries}): {e}")
            time.sleep(1)
    return []


def main():
    all_topics = []
    print(f"=== 开始生成 500 个辩题 (共 {len(CATEGORIES)} 个分类) ===")

    # 检查 tqdm 是否存在，不存在就用普通 range
    iterator = CATEGORIES
    try:
        iterator = tqdm(CATEGORIES, desc="生成进度")
    except ImportError:
        pass

    for category in iterator:
        print(f"\n正在生成领域：{category} ...")
        topics = generate_topics_by_category(category, count=50)

        # 简单的去重
        new_topics = [t for t in topics if t not in all_topics]
        all_topics.extend(new_topics)
        print(f"  + 成功获取 {len(new_topics)} 个辩题")

        # 避免触发速率限制
        time.sleep(1)

    # 最终去重
    all_topics = list(set(all_topics))

    # 保存文件
    output_file = "topics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_topics, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成！共生成 {len(all_topics)} 个辩题 ===")
    print(f"文件已保存至: {output_file}")

    # 预览前 10 个
    print("\n预览前 10 个辩题:")
    for i, t in enumerate(all_topics[:10]):
        print(f"{i + 1}. {t}")


if __name__ == "__main__":
    main()