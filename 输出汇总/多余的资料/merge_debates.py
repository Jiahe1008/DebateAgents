import os
import re
from pathlib import Path

def merge_jsonl_files(output_dir: str = "输出汇总", base_name: str = "seed_debates"):
    """
    合并 output_dir 下所有 seed_debates_i.jsonl 文件为 seed_debates.jsonl
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"目录不存在: {output_path}")

    # 获取所有匹配文件: seed_debates_1.jsonl, seed_debates_2.jsonl, ...
    pattern = re.compile(rf"{base_name}_(\d+)\.jsonl")
    files_with_index = []

    for file in output_path.iterdir():
        if file.is_file() and file.suffix == ".jsonl":
            match = pattern.match(file.name)
            if match:
                index = int(match.group(1))
                files_with_index.append((index, file))

    if not files_with_index:
        raise ValueError(f"未找到 {base_name}_*.jsonl 文件")

    # 按数字索引排序
    files_with_index.sort(key=lambda x: x[0])
    sorted_files = [file for _, file in files_with_index]

    print(f"🔍 找到 {len(sorted_files)} 个文件:")
    for f in sorted_files:
        line_count = sum(1 for _ in open(f, 'r', encoding='utf-8'))
        print(f"  - {f.name} ({line_count} 行)")

    # 合并到新文件
    final_output = output_path / f"{base_name}.jsonl"
    total_lines = 0

    with open(final_output, 'w', encoding='utf-8') as outfile:
        for file_path in sorted_files:
            print(f"📦 正在合并: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # 跳过空行
                        outfile.write(line + '\n')
                        total_lines += 1

    print(f"\n✅ 合并完成!")
    print(f"📁 输出文件: {final_output}")
    print(f"📊 总辩论场次: {total_lines}")
    print(f"💡 提示: 每行 = 1 场完整辩论")

if __name__ == "__main__":
    merge_jsonl_files()