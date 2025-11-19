import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# generate_seed_data.py (moved to lm_generation)
from data_generator import SeedDataGenerator


def main():
    generator = SeedDataGenerator()

    # 生成种子数据（更改 num_samples 可调整数量）
    seed_data = generator.generate_seed_dataset(num_samples=61, save_path=os.path.join('data', 'seed_debates.jsonl'))

    print(f"成功生成 {len(seed_data)} 个种子辩论样本")

    # 统计信息
    print("\n=== 数据统计 ===")
    print(f"总样本数: {len(seed_data)}")
    if seed_data:
        print(f"示例主题: {seed_data[0].get('topic', '')}")


if __name__ == "__main__":
    main()
