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
    return PromptTemplate(
        template="""
            你是法律 RAG 系统中的“检索策略选择器”。请根据用户查询，从以下四种策略中选择一个最适合的检索策略。

            你只能返回以下四个策略名称之一：
            直接检索
            假设问题检索
            子查询检索
            回溯问题检索

            不要输出解释、标点、编号或其他内容。

            策略说明：

            1. 直接检索
            适用场景：
            - 用户问题表达清晰，法律主体、行为、法规名称、法条编号或关键词明确。
            - 问题只需要查找某个具体规定、定义、处罚标准、适用条件或程序要求。
            - 问题可以直接用原 query 检索到相关法律依据。

            典型示例：
            - 民法典关于合同解除的规定是什么？
            - 反洗钱法中拒绝配合调查有什么处罚？
            - 中华人民共和国食品安全法对食品生产经营者有哪些要求？
            - 婚姻登记条例规定的离婚登记流程是什么？

            2. 假设问题检索
            适用场景：
            - 用户问题比较口语化、抽象或缺少明确法律关键词。
            - 直接检索可能难以命中法律依据，需要先生成一段可能的法律表述再检索。
            - 问题更像生活咨询，需要转成法律规范语言。

            典型示例：
            - 老板一直拖着不发工资怎么办？
            - 网购东西坏了商家不处理怎么办？
            - 别人拿我的照片做广告算违法吗？
            - 小区物业乱收费该怎么办？

            3. 子查询检索
            适用场景：
            - 用户问题包含多个法律问题、多个主体、多个行为或多个判断维度。
            - 需要分别检索不同方面的法律依据，再合并回答。
            - 问题中出现“同时”“分别”“比较”“有哪些责任”“如何处理多个事项”等信号。

            典型示例：
            - 公司拖欠工资并且没有签劳动合同，员工可以主张哪些权利？
            - 交通事故中一方逃逸，保险公司和肇事者分别承担什么责任？
            - 平台泄露个人信息并造成损失，平台和直接侵权人分别有什么责任？
            - 离婚时房产、子女抚养和债务应该怎么处理？

            4. 回溯问题检索
            适用场景：
            - 用户问题包含复杂案情、长背景或大量细节，需要抽象成更基础的法律问题。
            - 问题核心不是某个具体条文，而是需要先识别上位法律关系或法律领域。
            - 需要从个案描述回溯到一般法律规则。

            典型示例：
            - 我朋友借了我的钱一直不还，但没有写借条，只有微信聊天记录和转账记录，我能不能起诉？
            - 员工在试用期被公司以不符合录用条件辞退，但公司没有说明具体标准，这种情况是否合法？
            - 商家在直播间承诺假一赔十，但买到假货后拒绝赔偿，消费者可以依据什么维权？
            - 邻居装修导致我家墙体开裂，对方不承认责任，我应该依据什么法律处理？

            选择规则：
            - 如果问题明确、短、包含法规名/法条/处罚/流程/定义，优先选择“直接检索”。
            - 如果问题包含多个独立诉求或多个责任主体，选择“子查询检索”。
            - 如果问题很口语化但目标单一，选择“假设问题检索”。
            - 如果问题案情复杂、细节多，需要抽象为上位法律关系，选择“回溯问题检索”。
            - 不要为了增强而增强；能直接检索的问题不要选择其他策略。

            用户查询：
            {query}

            策略名称：
            """,
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
