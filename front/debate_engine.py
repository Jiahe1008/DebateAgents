import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import Dict, Tuple
import time

# --- 模型路径配置 ---
# 您可以在这里指定您的两个模型路径
MODEL_PATH_PRO = "/data/gzb/code/DebateAgents/output/qwen-1.5B-finetune/ckpt_merged"
MODEL_PATH_CON = "/data/gzb/code/DebateAgents/output/qwen-1.5B-finetune/ckpt_merged" # 推荐使用同一个微调模型，以进行最公平的对抗
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model(role: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if role in loaded_models:
        return loaded_models[role]
    model_path = MODEL_PATH_PRO if role == "pro" else MODEL_PATH_CON
    print(f"正在为角色 '{role}' 从 '{model_path}' 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        
        # 优化生成配置，鼓励多样性和自然性
        generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        generation_config.temperature = 0.75      # 稍微增加创造性
        generation_config.top_p = 0.8
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15 # 适度降低重复惩罚
        model.generation_config = generation_config
        
        loaded_models[role] = (model, tokenizer)
        print(f"'{role}' 模型加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise e

def generate_response(role: str, prompt: str) -> str:
    """
    采用双重截断逻辑，确保在任何情况下都能正确提取模型生成的内容。
    """
    try:
        model, tokenizer = get_model(role)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=1024, eos_token_id=tokenizer.eos_token_id)
        
        # 解码得到模型的完整输出
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 在后台打印原始输出，以便调试
        print(f"\n[DEBUG] 角色 '{role}' 的原始输出:\n---\n{full_output}\n---\n")
        if full_output.startswith(prompt):
            content = full_output[len(prompt):].strip()
        else:
            sft_response_marker = "### Response:"
            last_marker_index = full_output.rfind(sft_response_marker) # 使用rfind找到最后一个标记
            if last_marker_index != -1:
                content = full_output[last_marker_index + len(sft_response_marker):].strip()
            else:
                content = full_output.strip()

        # 第二步：对截取后的内容进行健康检查
        if not content:
            return f"（模型输出了空内容，可能是内部错误或内容限制）"
            
        return content

    except Exception as e:
        print(f"错误：在为角色 '{role}' 生成响应时发生异常: {e}")
        return f"（生成过程中发生严重错误: {str(e)}）"

class DebateState:
    def __init__(self, topic: str):
        self.topic = topic
        self.pro_opening: str = ""
        self.con_opening: str = ""
        self.history = [] # 新增：记录完整对话历史，用于总结

class PromptDirector:
    """
    终极通用版Prompt指挥官，使用动态的“强力纠偏”指令，确保模型严格遵守任何辩题的立场。
    """
    @staticmethod
    def _format_sft_prompt(instruction: str, input_data: Dict) -> str:
        input_str = "\n".join([f"- {key}: {value}" for key, value in input_data.items()])
        return f"你必须严格遵守你的角色和立场。\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n"

    @staticmethod
    def get_opening_prompt(topic: str, role: str) -> str:
        if role == 'pro':
            stance_description = f"你的立场是【正方】，你必须论证【{topic}】这个观点是完全正确的。"
        else: # role == 'con'
            stance_description = f"你的立场是【反方】，你必须论证【{topic}】这个观点是完全错误的。"

        instruction = f"你是一位辩论赛的一辩。{stance_description} 你的任务是构建一套逻辑严密的立论体系，并发表一段精彩的立论陈词。严禁偏离你的指定立场。"
        
        input_data = { "辩题": topic }
        return PromptDirector._format_sft_prompt(instruction, input_data)

    @staticmethod
    def get_rebuttal_prompt(state: 'DebateState', role: str, opponent_speech: str) -> str:
        if role == 'pro':
            stance_description = f"你的立场是【正方】，必须支持【{state.topic}】。你的立场是【正方】，必须支持【{state.topic}】"
        else: # role == 'con'
            stance_description = f"你的立场是【反方】，必须反对【{state.topic}】。你的立场是【反方】，必须反对【{state.topic}】"

        instruction = f"你是一位辩论赛的辩手。{stance_description} 请分析对方的观点，结合我方立论，对其逻辑漏洞或价值缺陷进行有力的反驳。"
        
        input_data = {
            "辩题": state.topic,
            "我方核心立论": state.pro_opening if role == "pro" else state.con_opening,
            "对方最新辩词": opponent_speech
        }
        return PromptDirector._format_sft_prompt(instruction, input_data)
        
    @staticmethod
    def get_closing_prompt(state: 'DebateState', role: str) -> str:
        if role == 'pro':
            stance_description = f"你的立场是【正方】，必须支持【{state.topic}】。"
        else: # role == 'con'
            stance_description = f"你的立场是【反方】，必须反对【{state.topic}】。"

        instruction = f"你是一位辩论赛的四辩。{stance_description} 请高度概括并收束全场，进行最终总结，为我方锁定胜局。"
        
        opponent_role_prefix = '反方' if role == 'pro' else '正方'
        opponent_speeches = "\n".join([f"对方发言: {turn['content']}" for turn in state.history if turn['speaker'].startswith(opponent_role_prefix)])
        
        input_data = {
            "辩题": state.topic,
            "我方核心立论": state.pro_opening if role == "pro" else state.con_opening,
            "对方整场主要观点摘要": opponent_speeches if opponent_speeches else "对方尚未形成有效观点。"
        }
        return PromptDirector._format_sft_prompt(instruction, input_data)

def run_debate(debate_topic: str):
    """
    最终优化的辩论流程，增加了对生成结果的健康检查，确保流程的健壮性。
    """
    try:
        state = DebateState(debate_topic)
        director = PromptDirector()
        
        yield {"speaker": "系统", "content": "辩论准备阶段：正在加载模型..."}
        get_model("pro")
        get_model("con")
        yield {"speaker": "系统", "content": f"模型加载完毕！\n辩题：【{debate_topic}】"}
        time.sleep(1)

        # --- 1. 立论阶段 ---
        yield {"speaker": "系统", "content": "="*20 + "\n第一阶段：立论陈词\n" + "="*20}
        
        # 正方立论
        yield {"speaker": "系统", "content": "正方一辩，请立论..."}
        pro_opening_response = generate_response("pro", director.get_opening_prompt(state.topic, "pro"))
        if not pro_opening_response or not pro_opening_response.strip(): # 健康检查
            yield {"speaker": "正方（立论）", "content": "（生成失败，无法进行立论）"}
            yield {"speaker": "系统", "content": "由于正方立论失败，辩论无法继续。"}
            return # 辩论中止
        state.pro_opening = pro_opening_response
        pro_opening_turn = {"speaker": "正方（立论）", "content": state.pro_opening}
        state.history.append(pro_opening_turn)
        yield pro_opening_turn
        
        # 反方立论
        yield {"speaker": "系统", "content": "反方一辩，请立论..."}
        con_opening_response = generate_response("con", director.get_opening_prompt(state.topic, "con"))
        if not con_opening_response or not con_opening_response.strip(): # 健康检查
            yield {"speaker": "反方（立论）", "content": "（生成失败，无法进行立论）"}
            yield {"speaker": "系统", "content": "由于反方立论失败，辩论无法继续。"}
            return # 辩论中止
        state.con_opening = con_opening_response
        con_opening_turn = {"speaker": "反方（立论）", "content": state.con_opening}
        state.history.append(con_opening_turn)
        yield con_opening_turn

        # --- 2. 自由辩论阶段 (两轮) ---
        yield {"speaker": "系统", "content": "="*20 + "\n第二阶段：自由辩论\n" + "="*20}
        latest_speech = state.con_opening

        for i in range(2):
            round_num = i + 1
            yield {"speaker": "系统", "content": f"--- 第 {round_num} 轮交锋 ---"}
            
            # 正方反驳
            pro_response = generate_response("pro", director.get_rebuttal_prompt(state, "pro", latest_speech))
            if not pro_response or not pro_response.strip(): # 健康检查
                yield {"speaker": f"正方 (第{round_num}轮)", "content": "（生成失败，本轮无法发言）"}
                # 注意：即使正方发言失败，辩论依然可以继续，只是latest_speech不会更新
            else:
                pro_rebuttal_turn = {"speaker": f"正方 (第{round_num}轮)", "content": pro_response}
                state.history.append(pro_rebuttal_turn)
                yield pro_rebuttal_turn
                latest_speech = pro_response # 只有在成功生成后才更新

            # 反方反驳
            con_response = generate_response("con", director.get_rebuttal_prompt(state, "con", latest_speech))
            if not con_response or not con_response.strip(): # 健康检查
                yield {"speaker": f"反方 (第{round_num}轮)", "content": "（生成失败，本轮无法发言）"}
            else:
                con_rebuttal_turn = {"speaker": f"反方 (第{round_num}轮)", "content": con_response}
                state.history.append(con_rebuttal_turn)
                yield con_rebuttal_turn
                latest_speech = con_response # 只有在成功生成后才更新

        # --- 3. 总结陈词阶段 ---
        yield {"speaker": "系统", "content": "="*20 + "\n第三阶段：总结陈词\n" + "="*20}
        
        # 反方总结
        yield {"speaker": "系统", "content": "反方四辩，请总结..."}
        con_closing_response = generate_response("con", director.get_closing_prompt(state, "con"))
        if not con_closing_response or not con_closing_response.strip(): # 健康检查
            yield {"speaker": "反方（总结）", "content": "（总结陈词生成失败）"}
        else:
            yield {"speaker": "反方（总结）", "content": con_closing_response}

        # 正方总结
        yield {"speaker": "系统", "content": "正方四辩，请总结..."}
        pro_closing_response = generate_response("pro", director.get_closing_prompt(state, "pro"))
        if not pro_closing_response or not pro_closing_response.strip(): # 健康检查
            yield {"speaker": "正方（总结）", "content": "（总结陈词生成失败）"}
        else:
            yield {"speaker": "正方（总结）", "content": pro_closing_response}
        
        yield {"speaker": "系统", "content": "辩论全部流程结束。"}

    except Exception as e:
        print(f"辩论过程中发生严重错误: {e}")
        yield {"speaker": "系统", "content": f"发生严重错误，辩论中止：{str(e)}"}