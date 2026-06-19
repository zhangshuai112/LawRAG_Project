import os
import sys
import jieba

current_path = os.path.abspath(__file__)
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.insert(0, project_path)
from base import logger

#小写+分词列表

def preprocess_text(text):
    try:
        result = jieba.lcut(text.lower())
        logger.info(f"分词成功")
        return result
    except Exception as e:
        logger.error(f"分词失败: {e}")
        result = []


if __name__ == '__main__':
    print(preprocess_text("我今天要学习"))
