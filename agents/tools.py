# tools.py
from smolagents import tool 
import json
import time
from typing import List, Dict, Any
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse

def clean_text(text: str) -> str:
    """清理网页文本"""
    if not text:
        return ""
    # 移除多余空白和特殊字符
    text = ' '.join(text.split())
    return text[:1000]  # 限制长度

def fetch_page_content(url: str, timeout: int = 5) -> str:
    """获取网页正文内容（简化版）"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'lxml')
        
        # 移除脚本和样式
        for script in soup(["script", "style"]):
            script.decompose()
            
        # 尝试提取正文（简单策略）
        content = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        if not content:
            content = soup.body
            
        return clean_text(content.get_text() if content else "") if content else ""
    except Exception as e:
        return f"[抓取失败: {str(e)}]"

@tool
def search_knowledge(query: str, max_results: int = 3) -> str:
    """
    联网搜索知识（用于生成辩题或查背景）
    示例: search_knowledge("2024年热门社会争议话题")
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region='cn-zh',
                safesearch='off',
                timelimit='y',  # 过去一年
                max_results=max_results
            )
            summaries = []
            for r in results:
                title = clean_text(r.get('title', ''))
                body = clean_text(r.get('body', ''))
                link = r.get('href', '')
                summaries.append(f"【{title}】{body}\n来源: {link}")
            return "\n\n".join(summaries) if summaries else f"未找到关于'{query}'的相关信息"
    except Exception as e:
        return f"搜索失败: {str(e)}"

@tool
def fact_check(claim: str) -> str:
    """
    事实核查（用于辩论中验证数据）
    示例: fact_check("2023年全球AI投资总额为930亿美元")
    """
    query = f"核实 {claim}"
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region='cn-zh',
                safesearch='off',
                timelimit='y',
                max_results=2
            )
            if not results:
                return json.dumps({
                    "status": "无可靠来源",
                    "claim": claim,
                    "verification": "未能找到权威来源验证该声明",
                    "suggestion": "建议谨慎使用此数据"
                }, ensure_ascii=False)
                
            # 获取第一个结果的详情
            first = results[0]
            content = fetch_page_content(first['href'], timeout=4)
            
            return json.dumps({
                "status": "已核查",
                "claim": claim,
                "source_title": first['title'],
                "source_url": first['href'],
                "snippet": first['body'],
                "full_content_preview": content[:500] + ("..." if len(content) > 500 else ""),
                "suggestion": "请结合上下文判断信息可靠性"
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "核查失败",
            "claim": claim,
            "error": str(e),
            "suggestion": "建议手动验证或换用更明确的表述"
        }, ensure_ascii=False)