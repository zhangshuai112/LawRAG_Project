import os
import sys
import pymysql
import pandas as pd

current_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.insert(0,project_path)

from base import Config, logger

class Mysql_client:
    def __init__(self):
        self.logger = logger
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                db=Config().MYSQL_DATABASE
            )
            self.cursor = self.connection.cursor()
            self.logger.info('连接数据库成功')
        except pymysql.MySQLError as e:
            self.logger.error('连接数据库失败')
            self.logger.error(e)
            raise
    def create_table(self):
        create_table_query ='''
        CREATE TABLE IF NOT EXISTS jpkb (
        id int  AUTO_INCREMENT primary key,
        subject_name VARCHAR(20),
        question VARCHAR(1000),
        answer VARCHAR(1000)) 
    '''
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info('创建表成功')
        except pymysql.MySQLError as e:
            self.logger.error('创建表失败')
            self.logger.error(e)
            raise
    def insert_data(self,file_path):
        ##data.iterrows() 是 pandas 中用于遍历 DataFrame 行的方法。它返回一个迭代器，每次迭代生成一个 (index, row) 元组
        try:
            data = pd.read_csv(file_path)
            for _,row in data.iterrows():
                insert_query = "insert into jpkb (subject_name, question, answer) values (%s, %s, %s)"
                self.cursor.execute(insert_query,(row['学科名称'],row["问题"],row['答案']))
            self.connection.commit()
            self.logger.info('插入数据成功')
        except pymysql.MySQLError as e:
            self.logger.error('插入数据失败')
            self.logger.error(e)
            self.connection.rollback()
            raise

    def fetch_questions(self):
        try:
            selection_query = "select question from jpkb"
            self.cursor.execute(selection_query)
            questions = self.cursor.fetchall()
            self.logger.info("获取所有问题成功")
            return questions
        except pymysql.MySQLError as e:
            self.logger.error("获取所有问题失败")
            self.logger.error(e)
            return  None


    def fetch_answers(self, question):
        try:
            self.cursor.execute('select answer from jpkb where question=%s',(question,))
            answers = self.cursor.fetchone()
            logger.info("获取答案成功")
            return answers[0] if answers else None
        except pymysql.MySQLError as e:
            self.logger.error("获取答案失败")
            self.logger.error(e)
            return  None

    def close(self):
        try:
            self.cursor.close()
            self.connection.close()
            logger.info("关闭数据库连接成功")
        except pymysql.MySQLError as e:
            self.logger.error("关闭数据库连接失败")
            self.logger.error(e)





if __name__ == '__main__':
    mysql_client = Mysql_client()
    # mysql_client.create_table()
    # mysql_client.insert_data('../data/JP学科知识问答.csv')
    # questions = mysql_client.fetch_questions()
    # print(questions)
    # print(mysql_client.fetch_answers(question="关联子查询的执行顺序是什么"))
    mysql_client.close()