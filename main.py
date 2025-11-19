# main.py

from debate_system import DebateSystem


def main():
    print("🤖 专业辩论系统启动中...")
    print("🎯 使用专业prompts进行高质量辩论生成")

    # 创建辩论系统
    debate_system = DebateSystem()

    # 定义测试辩题
    topics = [
        "人工智能的发展利大于弊",
        "远程办公优于办公室工作",
        "大学教育在数字时代仍然必要"
    ]

    # 运行辩论
    for i, topic in enumerate(topics):
        print(f"\n{'=' * 60}")
        print(f"专业辩论 {i + 1}/{len(topics)}：{topic}")
        print(f"{'=' * 60}")

        try:
            record = debate_system.run_debate(topic, rounds=1)

            if not record or "rounds" not in record or not record["rounds"]:
                print("⚠️ 本轮辩论无有效结果")
                continue

            round_data = record["rounds"][0]

            # 安全提取正方分论点数量
            pro_content = round_data.get("pro", {})
            pro_count = 0
            if isinstance(pro_content, dict):
                pro_points = pro_content.get("分论点") or pro_content.get("sub_points") or []
                if isinstance(pro_points, list):
                    pro_count = len(pro_points)
            print(f"\n📋 正方生成 {pro_count} 个核心论点")

            # 安全提取反方分论点数量
            con_content = round_data.get("con", {})
            con_count = 0
            if isinstance(con_content, dict):
                con_points = con_content.get("分论点") or con_content.get("sub_points") or []
                if isinstance(con_points, list):
                    con_count = len(con_points)
            print(f"🎯 反方生成 {con_count} 个反驳点")

        except Exception as e:
            print(f"❌ 辩论出错：{e}")
            continue

    # 保存数据到 data/ 目录（与 config.py 一致）
    debate_system.save_debate_data(path="results/debate_data.jsonl")
    print("\n🎉 所有专业辩论完成！")
    print("💡 生成的数据已保存至 results/debate_data.jsonl，可直接用于小模型训练")


if __name__ == "__main__":
    main()