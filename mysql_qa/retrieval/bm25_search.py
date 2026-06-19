import os
import sys
from rank_bm25 import BM25Okapi
import numpy as np

current_path = os.path.abspath(__file__)
dirdir_path = os.path.dirname(os.path.dirname(current_path))
project_path = os.path.dirname(dirdir_path)
path = os.path.join(dirdir_path, 'utils')
sys.path.insert(0, path)
path = os.path.join(dirdir_path, 'db')
sys.path.insert(0, path)
path = os.path.join(dirdir_path, 'cache')
sys.path.insert(0, path)
path = os.path.join(project_path, 'base')
sys.path.insert(0, path)
from preprocess import preprocess_text
from mysql_client import Mysql_client
from redis_client import RedisClient
from logger import logger

##bm25类：_load_data-》加载存储的问题列表的方法


class BM25_search:
    def __init__(self,mysql_client,redis_client):
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.questions =None
        self.bm25=None
        self.logger = logger
        self.original_questions = None
        self._load_data()
    def _load_data(self):
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        self.original_questions = self.redis_client.get_data(original_key)
        self.questions = self.redis_client.get_data(tokenized_key)
        if not self.original_questions or not self.questions:
            original_questions = self.mysql_client.fetch_questions()
            if not original_questions:
                self.logger.error("没有获取到数据")
                return
            self.questions = [preprocess_text(q[0]) for q in original_questions]
            self.original_questions = [q[0] for q in original_questions]
            #存储到redis库中方便下次调用
            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, self.questions)
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25 模型初始化完成")
    def _softmax(self,x):
        exp_num = np.exp(x-np.max(x))
        return exp_num/np.sum(exp_num)
    def search(self,query,threshold=0.85):
        if not query:
            self.logger.error("请输入查询内容")
            return None,False
        if self.redis_client.get_answer(query):
            self.logger.info('在redis库中找到答案')
            return self.redis_client.get_answer(query),False
        try:
            query = preprocess_text(query)
            scores = self.bm25.get_scores(query)
            scores = self._softmax(scores)
            best_score_idx = scores.argmax()
            best_score = scores[best_score_idx]
            if best_score >= threshold:
                original_question = self.original_questions[best_score_idx]
                answer = self.mysql_client.fetch_answers(original_question)
                if answer:
                    self.redis_client.set_ex(query, answer)
                    self.logger.info(f"在mysql中找到答案，并缓存到redis中，最高分数为：{best_score}")
                    return answer, False
                else:
                    self.logger.info(f"在mysql中答案为空，最高分数为：{best_score}")
                    return None,True
            self.logger.info(f"在mysql中未找到答案，最高分数为：{best_score}")
            return None,True
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return None,True
if __name__ == '__main__':
    mysql_client = Mysql_client()
    redis_client = RedisClient()
    bm25_search = BM25_search(mysql_client,redis_client)
    print(bm25_search.search("两个人开发项目,我push到github上后别人直接pull可以看到我的代码吗为什么我要pull request 他才能看到我提交的代码"))