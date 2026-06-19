#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 19:02
# @Site    : 
# @File    : main.py
# @Software: PyCharm
from openai import OpenAI




import os,sys
current_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(current_path))
dir_path =  os.path.dirname(current_path)
core_path = os.path.join(dir_path,'core')
vector_path = os.path.join(core_path,'vector_test')
preprocess_path = os.path.join(core_path,'document_processor')
sys.path.insert(0,project_path)
sys.path.insert(0,core_path)
sys.path.insert(0,vector_path)
sys.path.insert(0,preprocess_path)
from base import config,logger
from config import Config
conf = Config()
from logger import logger
from core.vector_store import VectorStore
from core.document_processor import process_documents
# from core.strategy_selector import StrategySelection
from core.rag_system import RAGSystem
# llm_stream=StrategySelection().stream_call_dashscope

def main(query_mode=True,directory_path='./data_dir'):
    try:
        client = OpenAI(api_key=conf.DASHSCOPE_API_KEY,base_url=conf.DASHSCOPE_BASE_URL)
    except Exception as e:
        logger.error(f'大模型客服端初始化失败，失败原因为：{e}')
        return


    def call_dashscope(prompt):
        if not client:
            logger.error('大模型客服端初始化失败')
            return None
        try:
            completion = client.chat.completions.create( model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手."},
                    {"role": "user", "content": prompt},
                ])
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logger.error("LLM API 调用返回无效响应或空消息")
                return "错误: LLM返回无效响应"
        except Exception as e:
            logger.error(f"LLM API (call_dashscope) 调用失败: {e}")
            return f"错误: 调用LLM失败 - {e}"
    try:
        vector_store = VectorStore()
    except Exception as e:
        logger.error(f'向量数据库初始化失败，失败原因为：{e}')
        return


    #todo:数据存储模式
    if not query_mode:
        logger.info('开始向量数据库存储数据')
        total_chunks_added = 0
        for source_dir in conf.VALID_SOURCES:
            path = os.path.join(directory_path,f'{source_dir}_data')
            if os.path.exists(path):
                chunks = process_documents(path)
                try:
                    if  chunks:
                        vector_store.add_documents(chunks)
                        logger.info(f'向量数据库成功添加数据{len(chunks)}个')
                        total_chunks_added +=len(chunks)
                    else:
                        logger.error(f'向量数据库添加数据失败，目录{path}数据为空')
                except Exception as e:
                    logger.error(f'向量数据库添加数据失败，失败原因为：{e}')
            else:
                logger.warning(f'数据源目录不存在：{path}')
        logger.info(f'向量数据库添加数据完成，共添加{total_chunks_added}个数据')
    else:
        logger.info('开始向量数据库查询数据')
        try:
            rag = RAGSystem(vector_store,call_dashscope)
            # rag=RAGSystem(vector_store,llm_stream)
            logger.info('RAGSystem初始化成功')
        except Exception as e:
            logger.error(f'RAGSystem初始化失败，失败原因为：{e}')
            return
        source_filter = conf.VALID_SOURCES

        while True:
            query = input("\n请输入您的问题：")
            if query.lower() == "exit":
                logger.info("用户退出查询模式")
                print("再见！")
                break
            input_source = input('请输入查询类型：')
            if input_source in source_filter:
                logger.info(f'用户查询类型为：{input_source}')
            else:
                logger.warning(f'用户输入的查询类型{input_source}无效，将默认不过滤')
                print(f'用户输入的查询类型{input_source}无效，将默认不过滤')
            try:
                print('开始查询')
                answer=rag.generate_answer(query,source_filter=input_source)
                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {str(e)}")
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员。\n")
if __name__ == '__main__':
    main(query_mode=True)



