# config.py
import os

# 强制 Python 使用 UTF-8 模式，避免在构建 HTTP 请求时出现 ascii 编码错误
# 这尤其在包含中文 prompt 时能避免 'ascii' codec can't encode characters 错误。
os.environ.setdefault("PYTHONUTF8", "1")

from openai import OpenAI

# 配置API客户端
# set DASHSCOPE_API_KEY = sk-... 
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的API密钥")

# 验证 API Key 是否已设置且为 ASCII（避免把中文占位符传入导致 HTTP header 编码失败）
if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "你的API密钥":
    raise RuntimeError(
        "DASHSCOPE_API_KEY 未设置或仍为占位符。请在命令行设置环境变量：\n"
        "set DASHSCOPE_API_KEY=sk-你的真实密钥\n"
        "然后重新运行脚本。"
    )

if any(ord(ch) > 127 for ch in DASHSCOPE_API_KEY):
    raise RuntimeError(
        "检测到 DASHSCOPE_API_KEY 包含非 ASCII 字符，这会导致构建 HTTP header 时失败。\n"
        "请确保使用 ASCII 格式的密钥，例如从服务端控制台复制的 API key。"
    )

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 模型配置
MODEL_NAME = "qwen3-max"
# 输出目录常量
DATA_DIR = os.getenv("DEBATE_DATA_DIR", "data")

# Ensure data dir exists when imported
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception:
    pass