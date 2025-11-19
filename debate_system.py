# debate_system.py
from agents.agents import run_pro_debater, run_con_debater, run_judge
import json
from typing import List


def run_debate_round(topic: str):
    print(f"\n📢 开始辩题：{topic}\n")

    # 正方发言
    print("➡️ 正方生成中...")
    pro = run_pro_debater(topic)
    if not pro:
        print("❌ 正方生成失败")
        return None

    # 反方发言
    print("⬅️ 反方生成中...")
    con = run_con_debater(topic)
    if not con:
        print("❌ 反方生成失败")
        return None

    # 裁判评判
    print("⚖️ 裁判评判中...")
    # 直接传递结构化对象给裁判（保持一致）
    judge_result = run_judge(topic, pro, con)
    if not judge_result:
        print("❌ 裁判生成失败")
        return None

    # 输出结果
    print("\n" + "="*60)
    print("✅ 辩论完成！")
    # 裁判返回字段兼容：尝试多种 key
    winner = judge_result.get('winner') or judge_result.get('胜方') or judge_result.get('winner', None)
    reason = judge_result.get('reason') or judge_result.get('评判依据') or ''
    print(f"🏆 胜方：{winner or '未知'}")
    print(f"📝 评判依据：{reason}")
    print("="*60)

    return {
        "topic": topic,
        "pro": pro,
        "con": con,
        "judge": judge_result
    }


class DebateSystem:
    """轻量封装，兼容主程序中对 DebateSystem 的使用。"""

    def __init__(self):
        self.records: List[dict] = []

    def run_debate(self, topic: str, rounds: int = 1):
        record = run_debate_round(topic)
        if record:
            self.records.append(record)
            # wrap into expected format with 'rounds' key to match main.py
            return {"topic": topic, "rounds": [record], "final_judgment": record.get('judge')}
        return None

    def save_debate_data(self, path: str = 'results/debate_data.jsonl'):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    topic = "人工智能的发展利大于弊"
    result = run_debate_round(topic)
    if result:
        with open("debate_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print("\n💾 结果已保存到 debate_output.json")