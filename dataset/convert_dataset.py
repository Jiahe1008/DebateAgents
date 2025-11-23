import json
import os

# ======================= 配置区域 =======================
# 1. 在这里填入您所有原始、混乱的数据集文件路径
INPUT_FILES = [
    "/data/gzb/code/DebateAgents/dataset/training_data_fixed.jsonl",
    "/data/gzb/code/DebateAgents/dataset/debate_dataset_v2.jsonl",
    "/data/gzb/code/DebateAgents/dataset/logic_critique_dataset.jsonl" # 确保这个路径正确
]

# 2. 指定您想要输出的、全新的、干净的数据集文件名
OUTPUT_FILE = "/data/gzb/code/DebateAgents/dataset/unified_training_data.jsonl"
# ========================================================


def convert_value_to_string(key_name, value):
    """
    一个智能转换函数，将JSON对象或其它类型的值转换成格式化的字符串。
    """
    if isinstance(value, str):
        return value
    
    if isinstance(value, dict):
        # 对 'input' 和 'output' 字段使用不同的格式化策略
        if key_name == 'input':
            # 将 input 字典转换为 "key: value" 的格式
            parts = [f"{k}: {v}" for k, v in value.items() if v] # 忽略值为空的键
            return "\n".join(parts)
        elif key_name == 'output':
            # 将 output 字典转换为更结构化的 "[KEY]\nvalue" 格式
            parts = []
            if value.get("cot"): parts.append(f"[思维链]\n{value['cot']}")
            if value.get("answer"): parts.append(f"[正式辩词]\n{value['answer']}")
            # 您可以根据需要在这里添加对 opening_statement, closing_statement等的处理
            if value.get("opening_statement"): parts.append(f"[立论]\n{value['opening_statement']}")
            if value.get("closing_statement"): parts.append(f"[结辩]\n{value['closing_statement']}")
            return "\n\n".join(parts)
        else:
            # 对于其他未知的字典，使用通用的JSON字符串格式
            return json.dumps(value, ensure_ascii=False)
            
    # 如果是列表、数字等其他类型，也转换为JSON字符串
    if value is not None:
        return json.dumps(value, ensure_ascii=False)
        
    return "" # 如果值是 None，返回空字符串


def process_files():
    """
    主处理函数，读取所有输入文件，处理后写入到单个输出文件。
    """
    processed_count = 0
    error_count = 0
    
    # 使用 'w' 模式打开输出文件，如果文件已存在则会覆盖
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        print(f"开始处理文件，将结果写入到: {OUTPUT_FILE}")

        for filepath in INPUT_FILES:
            if not os.path.exists(filepath):
                print(f"警告：找不到文件 '{filepath}'，已跳过。")
                continue

            print(f"\n--- 正在处理文件: {filepath} ---")
            with open(filepath, 'r', encoding='utf-8') as infile:
                for i, line in enumerate(infile):
                    line_num = i + 1
                    try:
                        # 1. 加载原始的JSON行
                        original_data = json.loads(line)

                        # 2. 创建一个新的、干净的字典，只包含我们想要的键
                        clean_data = {}

                        # 3. 提取并转换每个字段
                        clean_data['instruction'] = original_data.get('instruction', '')
                        
                        input_val = original_data.get('input')
                        clean_data['input'] = convert_value_to_string('input', input_val)
                        
                        output_val = original_data.get('output')
                        clean_data['output'] = convert_value_to_string('output', output_val)

                        # 4. 将干净的字典写回新的jsonl文件
                        outfile.write(json.dumps(clean_data, ensure_ascii=False) + '\n')
                        processed_count += 1

                    except json.JSONDecodeError:
                        print(f"  错误: 在文件 '{filepath}' 的第 {line_num} 行发现JSON格式错误，已跳过。")
                        error_count += 1
                    except Exception as e:
                        print(f"  错误: 在文件 '{filepath}' 的第 {line_num} 行发生未知错误: {e}，已跳过。")
                        error_count += 1
    
    print("\n==================== 处理完成 ====================")
    print(f"成功处理并转换了 {processed_count} 条数据。")
    if error_count > 0:
        print(f"处理过程中跳过了 {error_count} 个错误行。")
    print(f"所有干净的数据都已保存在: {OUTPUT_FILE}")
    print("================================================")


if __name__ == "__main__":
    process_files()