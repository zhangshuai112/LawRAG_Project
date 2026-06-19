import configparser
import os

current_path = os.path.abspath(__file__)
base_path = os.path.dirname(current_path)
project_path = os.path.dirname(base_path)
config_file = os.path.join(project_path, 'config.ini')


class Config:
    def __init__(self, config_path=config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')

        self.PROJECT_ROOT = project_path
        self.RAG_QA_ROOT = os.path.join(project_path, 'rag_qa')
        self.DATA_DIR = os.path.join(self.RAG_QA_ROOT, 'data_dir')
        self.SAMPLES_DIR = os.path.join(self.RAG_QA_ROOT, 'samples')

        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')

        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.get('redis', 'port', fallback='6379')
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        self.REDIS_DB = self.config.get('redis', 'db', fallback='0')

        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')

        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='law_rag')
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='lawrag_final')

        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen3.6-plus')
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key')
        self.DASHSCOPE_BASE_URL = self.config.get(
            'llm', 'dashscope_base_url',
            fallback='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        self.VALID_SOURCES = eval(
            self.config.get(
                'app', 'valid_sources',
                fallback='["Administrative_regulations", "Constitution", "Judicial_Interpretation", "Law", "Local_regulations", "Supervisory_Law"]'
            )
        )
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        self.EMBEDDING_MODEL = self.config.get('model', 'embedding_model', fallback='D:/nlp_model/bge-m3')
        self.RERANKER_MODEL = self.config.get('model', 'reranker_model', fallback='D:/nlp_model/bge-reranker-large')
        self.BERT_MODEL = self.config.get('model', 'bert_model', fallback='D:/nlp_model/bert-base-chinese')
        self.DOCUMENT_SEGMENTATION_MODEL = self.config.get(
            'model', 'document_segmentation_model',
            fallback='D:/nlp_model/nlp_bert_document-segmentation_chinese-base'
        )


if __name__ == '__main__':
    conf = Config()
    print(conf.DASHSCOPE_BASE_URL)
    print(conf.DASHSCOPE_API_KEY)
    print(conf.LLM_MODEL)
    print(conf.PROJECT_ROOT)
