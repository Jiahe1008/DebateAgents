# lm_generation/dashscope_api.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

import json
import traceback
from config import client, MODEL_NAME, DATA_DIR
import dashscope  
from dashscope import Generation


# 如果想调用其他模型，可以传入 model 参数覆盖默认 MODEL_NAME（qwen3-max）
def call_api(
    prompt: str = None,
    messages: list = None,
    response_format: dict = None,
    max_tokens=10000,
    temperature=0.7,
    enable_thinking=False,
    model=None
) -> str:
    try:
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        
        actual_model = model or MODEL_NAME

        try:
            needs_json_object = isinstance(response_format, dict) and response_format.get('type') == 'json_object'
        except Exception:
            needs_json_object = False

        if needs_json_object:
            contains_json_keyword = any(('json' in (m.get('content') or '').lower()) for m in messages)
            if not contains_json_keyword:
                sys_msg = {"role": "system", "content": "Output MUST be valid JSON. The assistant must return JSON only (包含关键字 JSON)。"}
                messages = [sys_msg] + messages

        create_kwargs = dict(
            model=actual_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )

        if response_format is not None:
            create_kwargs['response_format'] = response_format

        create_kwargs['extra_body'] = {"enable_thinking": enable_thinking}

        completion = client.chat.completions.create(**create_kwargs)

        content = None
        try:
            content = completion.choices[0].message.content
        except Exception:
            try:
                content = completion.choices[0]["message"]["content"]
            except Exception:
                try:
                    content = completion.choices[0]["text"]
                except Exception:
                    try:
                        content = str(completion.choices[0])
                    except Exception:
                        content = None

        try:
            finish_reason = None
            usage_info = None
            try:
                finish_reason = completion.choices[0].finish_reason
            except Exception:
                try:
                    finish_reason = completion.choices[0]["finish_reason"]
                except Exception:
                    finish_reason = None
            try:
                usage_info = completion.usage
            except Exception:
                try:
                    usage_info = completion.get("usage") if isinstance(completion, dict) else None
                except Exception:
                    usage_info = None

            os.makedirs(DATA_DIR, exist_ok=True)
            with open(os.path.join(DATA_DIR, 'debug_output.txt'), 'a', encoding='utf-8') as f:
                f.write(f"\n=== API METADATA ===\nfinish_reason: {finish_reason}\nusage: {usage_info}\nresponse_format: {response_format}\n=== END METADATA ===\n")
        except Exception:
            pass

        if content:
            if isinstance(content, bytes):
                try:
                    content = content.decode("utf-8", errors="ignore")
                except Exception:
                    content = str(content)
            return content
        return ""

    except Exception as e:
        print(f"❌ API调用失败: {e}")
        traceback.print_exc()
        return ""


def extract_json(text: str):
    if not text:
        return None

    text = text.strip()
    text = text.replace('```json', '').replace('```', '')

    try:
        return json.loads(text, strict=False)
    except Exception:
        pass

    for start_tag, end_tag in [("<<<JSON>>>", "<<<END_JSON>>>"), ("<JSON>", "</JSON>")]:
        sidx = text.find(start_tag)
        eidx = text.find(end_tag)
        if sidx != -1 and eidx != -1 and eidx > sidx:
            inner = text[sidx + len(start_tag):eidx].strip()
            try:
                return json.loads(inner)
            except Exception:
                break

    def find_matching_brace(s: str, start_idx: int) -> int:
        i = start_idx
        n = len(s)
        brace_count = 0
        in_string = False
        escape = False
        while i < n:
            ch = s[i]
            if ch == '"' and not escape:
                in_string = not in_string
                i += 1
                continue
            if ch == '\\' and not escape:
                escape = True
                i += 1
                continue
            if not in_string:
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return i
            escape = False
            i += 1
        return -1

    start = text.find('{')
    end = -1
    if start != -1:
        end = find_matching_brace(text, start)

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        def sanitize_json_text(s: str) -> str:
            s = s.replace('，', ',')
            s = s.replace('\u201c', '"').replace('\u201d', '"')
            s = s.replace('“', '"').replace('”', '"')
            s = s.replace("\u2018", "'").replace("\u2019", "'")
            s = s.replace("‘", "'").replace("’", "'")

            out_chars = []
            in_string = False
            escape = False
            for ch in s:
                if ch == '"' and not escape:
                    in_string = not in_string
                    out_chars.append(ch)
                    escape = False
                    continue
                if ch == '\\' and not escape:
                    escape = True
                    out_chars.append(ch)
                    continue
                if (ch == '\n' or ch == '\r') and in_string and not escape:
                    out_chars.append('\\n')
                    escape = False
                    continue
                out_chars.append(ch)
                escape = False

            s2 = ''.join(out_chars)

            import re
            s2 = re.sub(r',\s*}', '}', s2)
            s2 = re.sub(r',\s*\]', ']', s2)

            if s2.count('"') % 2 == 1:
                s2 = s2 + '"'

            if "\\'" in s2 or (s2.count("'") > s2.count('"') * 2):
                s2 = s2.replace("'", '"')

            return s2

        try:
            fixed = sanitize_json_text(candidate)
            return json.loads(fixed, strict=False)
        except Exception:
            try:
                alt = candidate.replace("'", '"')
                alt = sanitize_json_text(alt)
                return json.loads(alt, strict=False)
            except Exception:
                pass

        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(candidate)
            return obj
        except Exception:
            pass

        try:
            if not candidate.strip().endswith('}'):
                cont_prompt = (
                    "上一次模型输出的 JSON 片段可能被截断。\n"
                    "请基于下面的不完整 JSON 继续输出剩余内容，直到产生一个完整的 JSON 对象。\n"
                    "只返回后续的 JSON 片段（不要重复已完整的部分，也不要输出任何解释或额外文字）。\n"
                    "如果确定无法继续，请返回空字符串。\n\n"
                    "已知片段开始：\n" + candidate + "\n"
                )
                cont_resp = call_api(cont_prompt, max_tokens=800, temperature=0.0, response_format={"type": "json_object"})
                if cont_resp:
                    merged = candidate + cont_resp
                    try:
                        merged_fixed = sanitize_json_text(merged)
                        return json.loads(merged_fixed, strict=False)
                    except Exception:
                        try:
                            decoder = json.JSONDecoder()
                            obj, idx = decoder.raw_decode(merged)
                            return obj
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            max_trim = min(1200, len(candidate))
            for trim in range(0, max_trim, 1):
                try_text = candidate[:len(candidate) - trim]
                try_text = sanitize_json_text(try_text)
                try:
                    return json.loads(try_text, strict=False)
                except Exception:
                    continue
        except Exception:
            pass

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, 'debug_output.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception:
        pass
    return None


def call_api_with_search(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """使用 OpenAI 兼容接口调用 DashScope，并启用联网搜索"""
    try:
        messages = [{"role": "user", "content": prompt}]
        
        # 使用 client（来自 config.py）调用，传入 enable_search via extra_body
        completion = client.chat.completions.create(
            model="qwen3-max",  # 或 qwen-max（如果支持），建议用 qwen3-max
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            extra_body={"enable_search": True}  # 👈 关键：启用联网搜索
        )

        if completion.choices:
            content = completion.choices[0].message.content
            return content.strip() if content else ""
        else:
            print("⚠️ No choices returned from API (with search)")
            return ""

    except Exception as e:
        print(f"❌ Exception in call_api_with_search: {e}")
        traceback.print_exc()
        return ""