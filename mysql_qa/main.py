from database.mysql_client import Mysql_client
from retrieval.bm25_search import BM25_search
from cache.redis_client import RedisClient
from base.logger import logger
import time

class MySQL_QA_System:
    def __init__(self):
        self.mysql_client = Mysql_client()
        self.redis_client = RedisClient()
        self.bm25_search = BM25_search(self.mysql_client,self.redis_client)
        self.logger = logger
    def search(self,query):
        start_time = time.time()
        self.logger.info("开始在sql中搜索")
        answer,_ = self.bm25_search.search(query)
        if answer:
            self.logger.info(f"在sql中找到答案，答案为：{answer}")
        else:
            self.logger.info("在sql中未找到答案，转到RAG检索")
            answer = "sql中没答案"

        self.logger.info(f"搜索总共耗时：{time.time()-start_time:.2f}")
        return answer
def main():
    my_sql_qa = MySQL_QA_System()
    try:
        print("欢迎使用mysql_qa_system")
        print("输入‘exit退出系统’")
        while True:
            query = input("\n输入问题").strip()
            if query.lower() == "exit":
                print("退出系统, 再见")
                logger.info("退出系统")
                break
            answer = my_sql_qa.search(query)
            print(f"\n答案{answer}")
    except Exception as e:
        logger.error(f"系统错误：{e}")
        print("退出系统")
main()