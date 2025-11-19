# 大模型高质量数据生成文档说明
## 一、大模型数据采集全流程
1. 概览：使用大模型qwen3-max；生成辩题100条；原始seed_debates.jsonl文件比training_data.jsonl提取文件信息更多，如果想获取更详细的输入语料可更改lm_generation/convert_seed_to_training.py文件

2. 使用流程：【全部在根目录下运行】

- 设置 API key（Windows cmd）:
```
set DASHSCOPE_API_KEY=sk-你的真实密钥
```
- 安装依赖（建议在虚拟环境中，且不保证version全都正确且模块一定必要）:
```
pip install -r requirements.txt
```
- 清洗手动输入的题目
```
python lm_generation/evaluate_topics.py
```
- 生成种子(其实是所有)数据（生成几条在generate_debate_topics中定义，正反方反驳轮数在main的参数里）:
```
python lm_generation\generate_seed_data.py
```
- 转为训练格式
```
python lm_generation/convert_seed_to_training.py
```
- 数据评估
```
python lm_generation/evaluate_debate_data.py --file data/training_data.jsonl
```

## 二、数据质量评估项详细
- 总样本数
统计输入文件中共有多少条辩论数据样本，是所有评估的基础数量指标。
- 角色分布
分析每条样本对应的角色类型（正方、反方、裁判或其他），检查三类角色是否均衡覆盖，避免某一方数据缺失。
- 轮次分布
统计各轮次（如第1轮立论、第2轮驳论等）的数据量，确保多轮辩论结构完整，尤其关注第1轮是否充足。
- 空输出问题
检查是否存在模型输出为空字符串的样本，这类数据无效，会影响训练效果。
- 立论缺失判准
针对正方或反方的立论（第1轮）样本，检查其输出中是否包含“判准”关键词（如【判准】或[判准]），这是辩论逻辑的核心要素，缺失说明内容不完整。
- 精确重复辩题
统计完全相同的辩题出现次数，重复辩题会降低数据多样性，影响模型泛化能力。
- 高相似度辩题对
利用语义向量计算辩题之间的相似度，识别虽文字不同但含义高度接近的辩题（相似度 > 0.85），防止表面多样实则冗余。
- 有效性验证（模拟成功率）
随机抽取部分样本，用本地轻量模型（如 Qwen/Qwen1.5-1.8B）重新生成响应，检验是否能产出合理长度的非空内容，以此间接评估原始数据的可学习性和合理性。
- 人工审核样本生成
自动抽取30条代表性样本保存到 review/human_review_sample.jsonl，供人工抽查内容质量、逻辑连贯性与格式规范性。

## 三、当前文件结构
- data 大模型即时生成数据
- 输出汇总 存放大模型高质量数据的汇总备份
- prompts 提示词
- lm_generation 大模型数据生成逻辑
- review 人工审核大模型样本
- config.py api调用配置文件
- training 为小模型训练预留，可改
- agents 为构建小模型智能体预留，引入smolagents框架，可改
- debate_system.py 为小模型训练流程预留，可改
- main.py 为启动小模型预留，可改
- tools.py 为小模型调用工具预留，可改