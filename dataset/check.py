import json

# --- 把您的三个数据集文件路径填在这里 ---
DATA_FILES = [
    "/data/gzb/code/DebateAgents/dataset/training_data_fixed.jsonl",
    "/data/gzb/code/DebateAgents/dataset/debate_dataset_v2.jsonl",
    "/data/gzb/code/DebateAgents/dataset/logic_critique_dataset.jsonl"
]

def check_cot_type(filepath):
    print(f"\n--- 正在检查文件: {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                row_num = i + 1
                try:
                    data = json.loads(line)
                    
                    # 检查 'output' 和 'cot' 是否存在
                    if 'output' in data and isinstance(data['output'], dict) and 'cot' in data['output']:
                        cot_value = data['output']['cot']
                        
                        # 核心检查：cot的值是否不是一个字符串？
                        if not isinstance(cot_value, str):
                            print(f"  [问题行!] 行号: {row_num}")
                            print(f"  > 'output.cot' 的类型是 {type(cot_value)}, 而不是字符串(string)。")
                            print(f"  > 它的值是: {cot_value}")
                            print("-" * 20)
                            
                except json.JSONDecodeError:
                    print(f"  [JSON格式错误!] 行号: {row_num}。这一行本身就不是一个有效的JSON。")
                    print(f"  > 内容: {line.strip()}")
                    print("-" * 20)
                    
        print(f"文件 '{filepath}' 检查完毕。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{filepath}'，请检查路径。")
    except Exception as e:
        print(f"读取文件 '{filepath}' 时发生未知错误: {e}")

if __name__ == "__main__":
    for file in DATA_FILES:
        check_cot_type(file)
