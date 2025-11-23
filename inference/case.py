import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from typing import Tuple, Dict

# --- 1. 配置区域 ---

# 请在这里填入您想要测试的、已经合并好的模型路径
# MODEL_PATH = "/data/gzb/code/DebateAgents/output/qwen-1.5B-finetune/ckpt_merged" 
MODEL_PATH = "/data/gzb/modelzoo/Qwen2.5-1.5B/" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局缓存，防止重复加载模型
loaded_model_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

# --- 2. 模型加载与响应生成函数 (来自我们之前的版本) ---

def get_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载模型和分词器到缓存中"""
    if model_path in loaded_model_cache:
        return loaded_model_cache[model_path]
    
    print(f"正在从 '{model_path}' 加载模型，请稍候...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto", 
            trust_remote_code=True
        )
        
        # 配置生成参数
        generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        generation_config.temperature = 0.7
        generation_config.top_p = 0.8
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.1
        model.generation_config = generation_config
        
        loaded_model_cache[model_path] = (model, tokenizer)
        print("模型加载成功！可以开始对话了。")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise e

def generate_ai_response(prompt: str) -> str:
    """调用AI模型生成响应，并使用健壮的逻辑解析输出"""
    try:
        model, tokenizer = get_model(MODEL_PATH)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, max_new_tokens=2048, eos_token_id=tokenizer.eos_token_id)
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 使用我们最终优化的“双重截断”逻辑来确保能拿到干净的回答
        if full_output.startswith(prompt):
            content = full_output[len(prompt):].strip()
        else:
            sft_response_marker = "### Response:"
            last_marker_index = full_output.rfind(sft_response_marker)
            if last_marker_index != -1:
                content = full_output[last_marker_index + len(sft_response_marker):].strip()
            else:
                content = full_output.strip()
                
        return content if content else "（AI没有生成有效内容）"

    except Exception as e:
        print(f"生成响应时发生错误: {e}")
        return f"（生成过程中发生错误: {str(e)}）"

# --- 3. 主程序：简单的命令行聊天循环 ---

def main():
    """主聊天循环函数"""
    # 预先加载模型，避免第一次对话时等待
    try:
        get_model(MODEL_PATH)
    except Exception:
        print("无法启动聊天程序，请检查模型路径。")
        return

    print("\n========================================")
    print("      欢迎使用AI模型简易聊天程序      ")
    print("   输入内容后按回车即可与AI对话   ")
    print("   输入 'quit' 或 'exit' 退出程序   ")
    print("========================================")

    while True:
        try:
            # 获取用户输入
            # user_input = input("\n你: ")
            user_input = "尊敬的评委们: 今天我将代表正方团队发言, 我的观点是“钱是万恶之源”。我的理由如下   首先, 财富不平等加剧社会矛盾. 拥有大量财富的人群往往能够支配更多资源和机会, 这种不公平导致了贫富差距拉大和社会不稳定因素增加.其次,金钱可以扭曲人的行为. 人们往往会为了追求更多的利益而做出违背道德的行为, 如贪污腐败、行贿受贿等现象频发, 影响国家政治生态及公共秩序稳定.此外,过度依赖物质刺激会导致精神空虚. 在现代社会中, 大部分人通过消费来满足自己的欲望, 而且这些物质享受往往是短暂性的, 无法真正带来持久幸福感.最后, 金钱缺乏可持续性. 现代经济体系高度依赖外部资源支持发展, 当自然资源枯竭或环境恶化时, 全球范围内的危机将接踵而来, 导致人类生存环境遭受严重破坏.综上所述,钱确实是一个充满争议的话题, 它既有可能推动社会发展进步, 也可能成为阻碍文明发展的绊脚石. 因此, 我坚信钱是万恶之源, 并期待与各位一同探讨这一问题背后的深刻含义 谢谢大家!"
            # 退出指令
            if user_input.lower() in ["quit", "exit"]:
                print("再见！")
                break

            # 构建符合模型微调格式的Prompt
            instruction = "你是一位辩论赛的辩手。请结合我方立论，敏锐地指出对方刚才发言中的逻辑谬误，并进行有力的回击。如果可以，请给出思考过程。"
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"

            # 生成并打印AI的回答
            print("AI正在思考...", end="", flush=True)
            ai_output = generate_ai_response(prompt)
            print(f"\rAI: {ai_output}") # \r清除“正在思考...”

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n程序发生未知错误: {e}")

if __name__ == "__main__":
    main()