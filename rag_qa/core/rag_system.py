#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 19:02
# @Site    : 
# @File    : rag_system.py
# @Software: PyCharm
import os,sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_path)
from base import logger
from  base.config import Config
conf = Config()
from prompts import RAGPrompts
from query_classifier import QueryClassifier
from strategy_selector import StrategySelection
from vector_store import VectorStore
import time
class RAGSystem:
    def __init__(self,vector_store,llm):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts().rag_prompt()
        path = os.path.join(project_path,'rag_qa','models','bert_query_classifier')
        self.query_class = QueryClassifier(path)
        self.retrieval_selection = StrategySelection()
        self.max_prompt_length = 10000

    def _invoke_llm_text(self, prompt: str) -> str:
        """内部单次调用 LLM，始终返回完整字符串（检索增强等场景使用）。"""
        result = self.llm(prompt)
        if isinstance(result, str):
            return result
        parts = []
        for chunk in result:
            if hasattr(chunk, "content"):
                parts.append(chunk.content or "")
            else:
                parts.append(str(chunk))
        return "".join(parts)

    def _stream_llm_tokens(self, prompt: str):
        """流式产出 LLM 文本片段，兼容 stream / 非 stream 两种 callable。"""
        result = self.llm(prompt)
        if isinstance(result, str):
            yield result
            return
        for chunk in result:
            if hasattr(chunk, "content"):
                text = chunk.content
                if text:
                    yield text
            elif chunk:
                yield str(chunk)

    def _retrieve_with_backtracking(self, query, source_filter=None):
        back_prompt =RAGPrompts().backtracking_prompt().format(query= query)
        try:
            query = self._invoke_llm_text(back_prompt)
            logger.info(f"开始使用回溯问题检索，问题：{query}")
            sub_rankers=self.vector_store.hybrid_search_with_rerank(query = query,source_filter = source_filter)
            return sub_rankers
        except Exception as e:
            logger.info(f"回溯问题检索失败，失败原因：{e}")
            return  []

    def _retrieve_with_subqueries(self,query,source_filter=None):
        #todo:子查询检索实现思路：先进行子查询模板的拼接，送入大模型并使用列表生成式转变为子查询列表
        #todo:遍历每一个子查询，把子查询送入到大模型中得到多个子块，把多个子块加入到一个子块列表中，遍历这个子块列表
        #todo:利用字典键不能重复的特性，进行子块的去重，并最后取值得到最后的子块列表返回
        try:
            sub_prompt = RAGPrompts.subquery_prompt().format(query = query)
            docs_prompt = self._invoke_llm_text(sub_prompt)
            docs_prompt_list = [i.strip()  for i in docs_prompt.split("\n") if i.strip()]
            logger.info(f'开始使用子块查询,子问题：{docs_prompt_list}')
            add_docs = []
            for i in docs_prompt_list:
                chunks = self.vector_store.hybrid_search_with_rerank(query=i, source_filter=source_filter)
                add_docs.extend(chunks)
            unique_docs_dict = {doc.page_content:doc for doc in add_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f'去重之后的检索文档数量：{len(unique_docs)}')
            return unique_docs
        except Exception as e:
            logger.info(f"子块查询失败，失败原因：{e}")
            return []

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query,source_filter = None):
        hyde_prompt = RAGPrompts().hyde_prompt().format(query=query)
        try:
            hyde_query = self._invoke_llm_text(hyde_prompt)
            logger.info(f"开始使用假设问题检索，问题：{hyde_query}")
            hyde_content=self.vector_store.hybrid_search_with_rerank(query=hyde_query,source_filter=source_filter)
            return hyde_content
        except Exception as e:
            logger.info(f"假设问题检索失败，失败原因：{e}")
            return []







    def retrieval_and_merge(self,query,strategy= None,source_filter=None):
        if not strategy:
            strategy = self.retrieval_selection.select_strategy(query)

        if strategy== "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query,source_filter)
        elif strategy== "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query,source_filter)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query,source_filter)
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(query=query,source_filter=source_filter)
        logger.info(f'{strategy}选择了{len(ranked_sub_chunks)}个文档，最后保存了{conf.CANDIDATE_M}个文档作为上下文')
        return ranked_sub_chunks[:conf.CANDIDATE_M]



    def generate_answer(self,query,source_filter=None, history=None):
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")
        # 验证历史格式
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
        elif history:
            history = history[-5:]  # 限制最多5轮
            for h in history:
                if not (isinstance(h, dict) and "question" in h and "answer" in h):
                    logger.warning(f"无效的历史条目: {h}，忽略历史")
                    history = []
                    break

        # 构造历史上下文
        history_context = ""
        if history:
            history_context = "\n".join(
                [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
            )
            logger.info(f"使用对话历史: {history_context[:100]}...")



        strategy = self.query_class.predict(query)
        #todo:先使用BERT模型进行策略选择
        if strategy == "通用知识":
            logger.info("查询为通用知识，直接调用 LLM")
            prompt = self.rag_prompt.format(
                context='', question=query,
                phone=Config().CUSTOMER_SERVICE_PHONE, history=history_context
            )
            try:
                collected = []
                for token in self._stream_llm_tokens(prompt):
                    collected.append(token)
                    yield token
                logger.info(f"成功调用大模型，\n问题：{query}\n答案：{''.join(collected)}")
            except Exception as e:
                logger.info(f'调用大模型失败，失败原因为:{e}')
                fallback = "信息不足，无法回答，请联系人工客服，电话：{}".format(Config().CUSTOMER_SERVICE_PHONE)
                yield fallback
            logger.info(f'运行模型完成，总耗时：{time.time() - start_time:.2f}秒')
            return

        # 专业咨询：RAG 检索 + 生成
        logger.info('检测出是专业咨询，开始调用 RAG 系统')
        retrieval_strategy = self.retrieval_selection.select_strategy(query)
        chunks = self.retrieval_and_merge(query, retrieval_strategy, source_filter)
        if chunks:
            contents = '\n\n'.join([chunk.page_content for chunk in chunks])
            logger.info(f'构建上下文完成，共{len(chunks)}个文档')
        else:
            contents = ""
            logger.info('没有检索到相关文档')

        prompt = self.rag_prompt.format(
            context=contents, question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE, history=history_context
        )

        if len(prompt) > self.max_prompt_length:
            logger.warning(f"提示长度 {len(prompt)} 超过 {self.max_prompt_length}，进行截断")
            prompt = prompt[:self.max_prompt_length]
            logger.info(f"截断后提示长度: {len(prompt)}")

        processing_time = time.time() - start_time
        logger.info(f"检索完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
        llm_start = time.time()

        try:
            collected = []
            for token in self._stream_llm_tokens(prompt):
                collected.append(token)
                yield token
            logger.info(f"LLM 查询处理完成 (耗时: {time.time() - llm_start:.2f}s, 查询: '{query}')")
            logger.info(f"答案：{''.join(collected)}")
        except Exception as e:
            logger.info(f'调用大模型失败，失败原因：{e}')
            logger.error(f"详细错误信息: {str(e)}")
            yield "抱歉，处理您的问题时出错。请联系人工客服，电话：{}".format(conf.CUSTOMER_SERVICE_PHONE)



if __name__ == '__main__':
    vector_store = VectorStore()
    llm = StrategySelection().stream_call_dashscope
    rag = RAGSystem(vector_store, llm)
    query = '在中国拒绝、阻碍反洗钱监督管理、调查，或者故意提供虚假材料有什么惩罚？'
    print(f"问题：{query}\n回答：", end="", flush=True)
    for token in rag.generate_answer(query):
        print(token, end="", flush=True)
    print()
