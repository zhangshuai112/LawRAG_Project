# 导入pandas库，用于数据处理和保存CSV文件
import pandas as pd
# 导入ragas库的evaluate函数，用于执行RAG评估
from ragas import evaluate
# 导入ragas评估指标（这里使用已实例化的指标对象，兼容当前 evaluate 接口）
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from openai import OpenAI
# 导入datasets库的Dataset类，用于构建RAGAS所需的数据格式
from datasets import Dataset
# 导入json库，用于加载JSON格式的评估数据集
import json
import os,sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base.config import Config

# 1. 加载生成的数据集
# 使用with语句打开JSON文件，确保文件正确关闭，指定编码为utf-8
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_evaluate_data.json"), "r", encoding="utf-8") as f:
    # 将JSON文件内容加载到data变量中，data为包含多个数据条目的列表
    data = json.load(f)

# 2. 转换为RAGAS格式
# 创建字典eval_data，将JSON数据转换为RAGAS要求的字段格式
eval_data = {
    # 提取每个数据条目的question字段，组成问题列表
    "question": [item["question"] for item in data][0:3],
    # 提取每个数据条目的answer字段，组成答案列表
    "answer": [item["answer"] for item in data][0:3],
    # 提取每个数据条目的context字段，组成上下文列表（每个context为列表）
    "contexts": [item["context"] for item in data][0:3],
    # 提取每个数据条目的ground_truth字段，组成真实答案列表
    "ground_truth": [item["ground_truth"] for item in data][0:3]
}
# 使用Dataset.from_dict将字典转换为RAGAS所需的Dataset对象
dataset = Dataset.from_dict(eval_data)

# 3. 配置RAGAS评估环境
# collections 指标只支持 ragas 的 modern InstructorLLM，需要用 llm_factory + OpenAI client
openai_client = OpenAI(api_key=Config().DASHSCOPE_API_KEY, base_url=Config().DASHSCOPE_BASE_URL)
ragas_llm = llm_factory(Config().LLM_MODEL, client=openai_client)
# 4. 执行评估
# 调用evaluate函数，传入数据集、评估指标、LLM模型和嵌入模型
result = evaluate(
    # 传入转换好的Dataset对象
    dataset=dataset,
    # 指定使用的评估指标列表
    metrics=[
        faithfulness,  # 忠实度：答案是否基于上下文
        context_precision,  # 上下文精确率：检索上下文中相关信息比例
        context_recall,  # 上下文召回率：上下文是否包含所有必要信息
    ],
    llm=ragas_llm,
)

# 5. 输出并保存结果
# 打印评估结果标题
print("RAGAS评估结果：")
# 打印评估结果，包含各指标的分数
print(result)
# 将评估结果转换为pandas DataFrame，便于保存
result_df = pd.DataFrame([result])
# 将DataFrame保存为CSV文件，文件名为ragas_evaluation_results.csv，不保存索引
result_df.to_csv("ragas_evaluation_results.csv", index=False)
