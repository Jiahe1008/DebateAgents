#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
完整的种子数据生成器 - 支持多轮辩论（立论 + 多轮反驳 + 裁判）
适配 qwen3-max 模型的强结构化输出能力
"""
import re
import random
import json
import time
import os
import sys
import traceback
from typing import List, Optional, Dict, Any
from lm_generation.dashscope_api import call_api_with_search
from pathlib import Path


# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import client, MODEL_NAME, DATA_DIR
from lm_generation.dashscope_api import call_api, extract_json


class SeedDataGenerator:
    """种子数据生成器 - 支持多轮辩论生成"""

    def __init__(self, sleep_between_calls: float = 1.5, default_rounds: int = 2):
        self.sleep = sleep_between_calls
        self.default_rounds = default_rounds  # 默认辩论轮数（至少2）
        
        # 定义 Prompt 路径（相对于项目根目录）
        self.prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        self.prompt_files = {
            "pro_opening": "正方陈述-prompt.txt",
            "con_opening": "反方陈述-prompt.txt",
            "pro_rebuttal": "正方反驳-prompt.txt",
            "con_rebuttal": "反方反驳-prompt.txt",
            "judge": "裁判-prompt.txt"
        }
        # 验证文件存在
        for key, filename in self.prompt_files.items():
            path = os.path.join(self.prompt_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺失 Prompt 文件: {path}")

    def _load_prompt(self, key: str) -> str:
        """加载指定角色的 prompt 模板"""
        path = os.path.join(self.prompt_dir, self.prompt_files[key])
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _call_api_json(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> Optional[dict]:
        """调用 API 并返回解析后的 JSON（增强容错）"""
        print(f"  🧠 调用 LLM (temp={temperature})...")
        resp = call_api(
            prompt=prompt,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if not resp:
            print("  ❌ API 返回为空")
            return None
            
        # === 新增：提取 JSON 块（兼容 ```json ... ```）===
        text = resp.strip()
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析全文
            json_str = text

        try:
            parsed = json.loads(json_str, strict=False)
            if isinstance(parsed, dict):
                return parsed
            else:
                print("  ❌ 解析结果不是字典")
                return None
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON 解析失败:\n{text[:500]}...\n错误: {str(e)}")
            return None


    def _format_opening_side(self, side_data: dict) -> str:
        """将立论 JSON 转为自然语言文本"""
        if not side_data:
            return "[内容缺失]"
        parts = []
        if side_data.get("evaluation_criteria"):
            parts.append(f"判准：{side_data['evaluation_criteria']}")
        if side_data.get("definition_of_terms"):
            defs = "；".join(f"{k}={v}" for k, v in side_data["definition_of_terms"].items())
            parts.append(f"定义：{defs}")
        args = side_data.get("core_arguments", [])
        if args:
            arg_texts = []
            for arg in args:
                if isinstance(arg, dict):
                    point = arg.get("argument_point", "").strip()
                    evidence = arg.get("evidence_or_example", "").strip()
                    if point:
                        line = f"- {point}"
                        if evidence:
                            line += f"（{evidence}）"
                        arg_texts.append(line)
            if arg_texts:
                parts.append("论点：\n" + "\n".join(arg_texts))
        return "\n".join(parts) if parts else str(side_data)
    
    def _format_rebuttal_side(self, rebuttal_data: dict) -> str:
        """将反驳 JSON 转为自然语言文本"""
        if not rebuttal_data:
            return "[内容缺失]"
        # 优先使用摘要字段，否则拼接要点
        summary = rebuttal_data.get("rebuttal_summary") or rebuttal_data.get("main_points_summary", "")
        if summary:
            return str(summary)
        points = rebuttal_data.get("rebuttal_points", [])
        if points:
            texts = []
            for p in points:
                if isinstance(p, dict):
                    pt = p.get("point", "")
                    ref = p.get("refuted_content", "")
                    if pt:
                        line = f"- {pt}"
                        if ref:
                            line += f"（针对：{ref}）"
                        texts.append(line)
            return "\n".join(texts) if texts else str(rebuttal_data)
        return str(rebuttal_data)


    def generate_structured_debate(self, topic: str, total_rounds: int = None) -> Optional[dict]:
        """
        生成完整多轮辩论
        total_rounds: 总轮数（至少2）。第1轮=立论，第2~N轮=反驳，最后由裁判评判
        """
        total_rounds = 4  # 🔥 强制设为 4 轮，无视传参

        if total_rounds is None:
            total_rounds = self.default_rounds
        if total_rounds < 2:
            total_rounds = 2

        print(f"\n[{topic}] | 目标轮数: {total_rounds}")
        debate_history = [] 

        # 构建自然语言上下文的函数 
        def build_natural_context(history: list) -> str:
            lines = [f"辩题：{topic}\n"]
            for rnd in history:
                rn = rnd["round_number"]
                lines.append(f"--- 第{rn}轮 ---")
                if rn == 1:
                    pro_text = self._format_opening_side(rnd["proponent"])
                    con_text = self._format_opening_side(rnd["opponent"])
                    lines.append(f"【正方立论】\n{pro_text}")
                    lines.append(f"【反方立论】\n{con_text}")
                else:
                    pro_text = self._format_rebuttal_side(rnd["proponent"])
                    con_text = self._format_rebuttal_side(rnd["opponent"])
                    lines.append(f"【正方反驳】\n{pro_text}")
                    lines.append(f"【反方反驳】\n{con_text}")
            return "\n\n".join(lines).strip()

        # === 第1轮：双方立论 ===
        print("  📢 第1轮：立论阶段")
        pro_opening_prompt = self._load_prompt("pro_opening").replace("{topic}", topic)
        con_opening_prompt = self._load_prompt("con_opening").replace("{topic}", topic)

        pro_stmt = self._call_api_json(pro_opening_prompt, temperature=0.2)
        if not pro_stmt:
            return None
        time.sleep(self.sleep)

        con_stmt = self._call_api_json(con_opening_prompt,temperature=0.2)
        if not con_stmt:
            return None
        time.sleep(self.sleep)

        round1 = {
            "round_number": 1,
            "proponent": pro_stmt,
            "opponent": con_stmt
        }
        debate_history.append(round1)

        pro_reb_prompt_template = self._load_prompt("pro_rebuttal")
        con_reb_prompt_template = self._load_prompt("con_rebuttal")

        # === 第2 到 total_rounds 轮：交替反驳 ===
        for rnd in range(2, total_rounds + 1):
            print(f"  🔁 第{rnd}轮：反驳阶段")
            last_round = debate_history[-1]

            # 获取首轮信息（用于注入判准和定义）
            initial_pro = debate_history[0]["proponent"]
            initial_con = debate_history[0]["opponent"]

            pro_criteria = initial_pro.get("evaluation_criteria", "未明确")
            con_criteria = initial_con.get("evaluation_criteria", "未明确")

            # 提取定义（假设是 dict）
            def format_defs(defs_dict):
                if not isinstance(defs_dict, dict):
                    return "未提供"
                return "；".join(f"{k}：{v}" for k, v in defs_dict.items()) if defs_dict else "未提供"

            pro_defs = format_defs(initial_pro.get("definition_of_terms"))
            con_defs = format_defs(initial_con.get("definition_of_terms"))

            # 构建完整历史文本
            full_history_text = build_natural_context(debate_history)

            # 使用模板副本
            pro_reb_prompt = pro_reb_prompt_template
            con_reb_prompt = con_reb_prompt_template

            # 正方回应反方上一轮
            pro_reb_prompt = pro_reb_prompt.replace("{topic}", topic)
            pro_reb_prompt = pro_reb_prompt.replace("{round_number}", str(rnd))
            pro_reb_prompt = pro_reb_prompt.replace("{proponent_initial_criteria}", pro_criteria)
            pro_reb_prompt = pro_reb_prompt.replace("{proponent_initial_definitions}", pro_defs)
            pro_reb_prompt = pro_reb_prompt.replace("{full_debate_history}", full_history_text)

            pro_resp = self._call_api_json(pro_reb_prompt,temperature=0.3)
            if not pro_resp:
                print("  ⚠️ 正方反驳失败，跳过本轮")
                break
            time.sleep(self.sleep)

            # 反方回应正方本轮
            con_reb_prompt = con_reb_prompt.replace("{topic}", topic)
            con_reb_prompt = con_reb_prompt.replace("{round_number}", str(rnd))
            con_reb_prompt = con_reb_prompt.replace("{opponent_initial_criteria}", con_criteria)
            con_reb_prompt = con_reb_prompt.replace("{opponent_initial_definitions}", con_defs)
            con_reb_prompt = con_reb_prompt.replace("{full_debate_history}", full_history_text)

            con_resp = self._call_api_json(con_reb_prompt,temperature=0.3)
            if not con_resp:
                print("  ⚠️ 反方反驳失败，跳过本轮")
                break
            time.sleep(self.sleep)

            debate_history.append({
                "round_number": rnd,
                "proponent": pro_resp,
                "opponent": con_resp
            })

        # === 裁判评判 ===
        print("  ⚖️ 裁判评判阶段")
        actual_rounds = len(debate_history)

        # 获取首轮信息用于裁判提示
        initial_pro = debate_history[0]["proponent"]
        initial_con = debate_history[0]["opponent"]
        pro_criteria = initial_pro.get("evaluation_criteria", "未明确")
        con_criteria = initial_con.get("evaluation_criteria", "未明确")

        def format_defs(defs_dict):
            if not isinstance(defs_dict, dict):
                return "未提供"
            return "；".join(f"{k}：{v}" for k, v in defs_dict.items()) if defs_dict else "未提供"
        
        pro_defs = format_defs(initial_pro.get("definition_of_terms"))
        con_defs = format_defs(initial_con.get("definition_of_terms"))
        
        full_history_text = build_natural_context(debate_history)

        judge_prompt = self._load_prompt("judge")
        judge_prompt = judge_prompt.replace("{topic}", topic)
        judge_prompt = judge_prompt.replace("{total_rounds}", str(actual_rounds))
        judge_prompt = judge_prompt.replace("{proponent_initial_criteria}", pro_criteria)
        judge_prompt = judge_prompt.replace("{proponent_initial_definitions}", pro_defs)
        judge_prompt = judge_prompt.replace("{opponent_initial_criteria}", con_criteria)
        judge_prompt = judge_prompt.replace("{opponent_initial_definitions}", con_defs)
        judge_prompt = judge_prompt.replace("{full_debate_natural_text}", full_history_text)  

        judgment = self._call_api_json(judge_prompt, temperature=0.1)
        if not judgment:
            return None

        return {
            "topic": topic,
            "total_rounds": actual_rounds,
            "debate_history": debate_history,
            "judgment": judgment,
            "proponent": debate_history[0]["proponent"],
            "opponent": debate_history[0]["opponent"]
        }

    def generate_debate_topics(self, num_topics: int = 10, existing_topics: Optional[List[str]] = None) -> List[str]:
        """从同目录 debate_topics.txt 读取辩题（假设已由 evaluate_topics.py 清洗）"""
        if existing_topics is None:
            existing_topics = []

        topics_file = Path(__file__).parent / "debate_topics.txt"
        if not topics_file.exists():
            raise FileNotFoundError(f"❌ 请先创建并清洗 {topics_file}（运行 evaluate_topics.py）")

        # 仅读取非空、非注释行
        with open(topics_file, 'r', encoding='utf-8') as f:
            topics = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith('#')
            ]

        # 可选：排除与 existing_topics 完全重复的（用于增量生成）
        if existing_topics:
            existing_set = set(existing_topics)
            topics = [t for t in topics if t not in existing_set]

        selected = topics[:num_topics]
        
        print(f"📝 从 {topics_file.name} 加载 {len(selected)} 个辩题（未做清洗，假定已预处理）")
        return selected


    def generate_review_sample(self, debates, output_dir, sample_size=30):
        """生成人工审核样本"""
        os.makedirs(output_dir, exist_ok=True)
        sampled = random.sample(debates, min(sample_size, len(debates)))
        with open(os.path.join(output_dir, "seed_review_sample.jsonl"), 'w', encoding='utf-8') as f:
            for d in sampled:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    def generate_seed_dataset(self, num_samples: int = 20, save_path: Optional[str] = None, total_rounds: int = None) -> List[dict]:
        """主生成流程"""
        print(f"🚀 开始生成 {num_samples} 个标准辩论样本 (多轮模式, rounds={total_rounds or self.default_rounds})...")
        
        topics = self.generate_debate_topics(num_samples)
        print(f"📝 获得 {len(topics)} 个候选话题")

        # 清空保存文件（避免追加旧数据）
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            open(save_path, 'w').close()  # 清空文件，避免上次残留
        
        results = []
        for i, topic in enumerate(topics):
            if len(results) >= num_samples:
                break
            print(f"\n--- [{i+1}/{min(num_samples, len(topics))}] ---")
            structured = self.generate_structured_debate(topic, total_rounds=total_rounds)
            if structured:
                results.append(structured)
                print(f"  ✅ 成功生成样本 {len(results)}")

                if save_path:
                    with open(save_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(structured, ensure_ascii=False) + '\n')
                    print(f"  💾 已追加保存至: {save_path}")

            else:
                print(f"  ❌ 跳过话题: {topic}")
            
            if len(results) >= num_samples:
                break
            time.sleep(self.sleep)
        
        # 保存
        if save_path and results:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\n💾 已保存 {len(results)} 个样本到: {save_path}")
            # 生成审核样本
            review_dir = os.path.join(os.path.dirname(save_path), "..", "review")
            self.generate_review_sample(results, review_dir, sample_size=30)

        return results


if __name__ == "__main__":
    generator = SeedDataGenerator(sleep_between_calls=1.5, default_rounds=2)
    data = generator.generate_seed_dataset(
        num_samples=2,
        save_path=os.path.join(DATA_DIR, "seed_dataset.jsonl"),
        total_rounds=4  # 可在此指定轮数
    )