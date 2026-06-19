import logging
import os
from config import Config

#设置log保存路径
base_path = Config().LOG_FILE
current_path = os.path.abspath(__file__)
f_path = os.path.dirname(current_path)
project_path = os.path.dirname(f_path)
log_path = os.path.join(project_path, base_path)
def setup_logging(log_file=log_path):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 创建日志记录器
    logger = logging.getLogger('LawRAG')
    # 创建日志记录器的记录水平
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 创建文件处理器，控制台处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 设置format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 放入到日志记录器中
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    # 返回logger
    return logger

logger = setup_logging()
if __name__ == '__main__':
    logger = setup_logging()
    logger.info('good')