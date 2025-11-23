from flask import Flask, render_template, request, Response, stream_with_context
import json
from debate_engine import run_debate # 导入我们改造后的辩论引擎

app = Flask(__name__)

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/start-debate', methods=['POST'])
def start_debate():
    """开始辩论并以流式响应返回结果"""
    topic = request.form['topic']
    if not topic:
        return "错误：辩题不能为空！", 400

    def event_stream():
        """
        这个内部函数会调用辩论引擎，并将每一句输出
        都格式化为 Server-Sent Event (SSE) 推送出去。
        """
        for speech in run_debate(topic):
            # 格式化为SSE事件流
            sse_data = f"data: {json.dumps(speech)}\n\n"
            yield sse_data
            
    # 使用流式响应，内容类型为 text/event-stream
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
