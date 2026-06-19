#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 19:02
# @Site    : 
# @File    : strategy_selector.py
# @Software: PyCharm
import os,sys

from langchain_core.prompts import PromptTemplate

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_path)
from base import logger
from base.config import Config
from openai import OpenAI
from langchain_openai import ChatOpenAI
import ollama


class StrategySelection:
    def __init__(self):

        # self.client = OpenAI(api_key=Config().DASHSCOPE_API_KEY,
        #                      base_url=Config().DASHSCOPE_BASE_URL,)
        self.client = ChatOpenAI(model=Config().LLM_MODEL,
                                 api_key=Config().DASHSCOPE_API_KEY,
                                 base_url=Config().DASHSCOPE_BASE_URL, temperature=0.1)
        self.strategy_prompt = self._get_strategy_prompt()

    def call_dashscope(self,prompt):
        # completion = self.client.invoke( messages=[{'role':'system','content':'你是一个有用的助手。'},
        #                                               {'role':'user','content':prompt}],
        #                                 temperature=0.1)
        result = self.client.invoke(input=[('system', '你是一个有用的助手。'),
                                                  ('user', prompt)])
        # return completion.choices[0].message.content
        return result.content

    def stream_call_dashscope(self, prompt):
        # completion = self.client.invoke( messages=[{'role':'system','content':'你是一个有用的助手。'},
        #                                               {'role':'user','content':prompt}],
        #                                 temperature=0.1)
        result = self.client.stream(input=[('system', '你是一个有用的助手。'),
                                           ('user', prompt)])
        # return completion.choices[0].message.content
        return result

    def call_ollama(self,prompt):
        completion = ollama.chat(
            model="deepseek_law-8b",
            messages=[{'role':'system','content':'你是一个有用的助手。'}
                        ,{"role": "user", "content": prompt}],

        )
        return completion["message"]["content"]
    def _get_strategy_prompt(self):
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )
    def select_strategy(self,query):
        prompt = self.strategy_prompt.format(query=query)
        try:
            strategy =self.call_dashscope(prompt).strip()
            logger.info(f'策略选择为--》{strategy}')
        except Exception as e:
            logger.error(f'策略选择错误原因：{e}')
            strategy ='直接检索'
        return strategy
if __name__ == '__main__':
    cs = StrategySelection()
    # a=cs.select_strategy('你是个谁')
    a = cs.call_dashscope('你好')
    print(a)
