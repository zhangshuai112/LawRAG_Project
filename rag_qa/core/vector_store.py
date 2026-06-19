#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/3 18:32
# @Site    :
# @File    : vector_test.py
# @Software: PyCharm
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 19:02
# @Site    :
# @File    : vector_store.py
# @Software: PyCharm
import hashlib
import os,sys,torch
from typing import List

# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel


project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_path)
from base import logger,config
from logger import logger
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from config import Config
from document_processor import process_documents
conf = Config()

# BGE-M3 对部分极短文本无法产出 lexical 权重；Milvus 稀疏字段不能为空，用占位符保证可入库
_EMPTY_SPARSE_PLACEHOLDER = {"0": 1e-6}


def _normalize_sparse_vector(sparse_vector):
    """将 lexical_weights 转为 Milvus 可接受的 dict；空稀疏时使用占位符。"""
    if not sparse_vector:
        return _EMPTY_SPARSE_PLACEHOLDER.copy()
    return dict(sparse_vector)


class VectorStore:
    #todo 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        #初始化重排序模型
        self.reranker = CrossEncoder(model_name_or_path= conf.RERANKER_MODEL, device='cuda',local_files_only=True)
        #初始化BGE_M3嵌入模型
        self.embedding_model = BGEM3FlagModel(
            model_name_or_path=conf.EMBEDDING_MODEL,
            use_fp16=False,
            device='cuda'
        )
        # self.embedding_model = BGEM3EmbeddingFunction(use_fp16=False, device="cuda",model_name_or_path=conf.EMBEDDING_MODEL,return_dense= True, return_sparse=True)
        #规定稠密向量维度
        self.dense_dim = 1024
        #milvus客户端
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    # 定义私有方法，创建或加载 Milvus 集合
    def _create_or_load_collection(self):
        # 检查指定集合是否已存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型sparse
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX (inverted) 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2}#为每个向量忽略其值最小的 20% 的非零维度
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name)


    def add_documents(self, documents):
        # 过滤掉 page_content 为 None 的文档
        valid_docs = [doc for doc in documents if doc.page_content is not None]
        if not valid_docs:
            logger.warning("没有有效的文档内容")
            return

        texts = [doc.page_content for doc in valid_docs]
        embeddings = self.embedding_model.encode(texts,batch_size=8,return_dense=True,
        return_sparse=True)
        # sparse_matrix = csr_matrix(embeddings["sparse"].tocsr())
        # sparse_vector = embeddings['lexical_weights']

        # 现在 embeddings 是一个字典，通过键名来访问数据
        dense_vectors = embeddings['dense_vecs']
        sparse_vectors = embeddings['lexical_weights']

        data = []
        empty_sparse_count = 0
        for i, doc in enumerate(valid_docs):  # ✅ 使用过滤后的列表
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            dense_vec = dense_vectors[i]
            raw_sparse = sparse_vectors[i]

            if not raw_sparse:
                empty_sparse_count += 1
                logger.debug(
                    f"稀疏向量为空，仅写入稠密向量: '{doc.page_content[:50]}...'"
                )
            sparse_vector = _normalize_sparse_vector(raw_sparse)

            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(
                f"已插入或更新 {len(data)} 个文档"
                + (f"，其中 {empty_sparse_count} 个稀疏向量为空（已用占位符保留稠密向量）"
                   if empty_sparse_count else "")
            )


    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None) -> List[Document]:
        # 使用 BGEM3FlagModel 生成查询的嵌入
        query_embeddings = self.embedding_model.encode(
            [query],
            return_dense=True,
            return_sparse=True
        )
        # query_embeddings 是字典，包含 'dense_vecs' 和 'lexical_weights' 两个键
        # 每个键对应的值是一个列表（长度为1）
        dense_query_vector = query_embeddings['dense_vecs'][0]
        sparse_query_vector = query_embeddings['lexical_weights'][0]  # 直接取列表第一个元素

        # 初始化过滤表达式
        filter_expr = f"source == '{source_filter}'" if source_filter else None

        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "nprobe": 10},
            limit=k,
            expr=filter_expr,
        )

        # 根据稀疏向量是否为空决定检索方式
        if sparse_query_vector:
            sparse_request = AnnSearchRequest(
                data=[sparse_query_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=k,
                expr=filter_expr,
            )
            ranker = WeightedRanker(1.0, 0.7)  # 稠密权重1.0，稀疏权重0.7
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=ranker,
                limit=k,
                output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
            )[0]
        else:
            logger.warning(f"查询 '{query}' 稀疏向量为空，向量路退化为仅稠密检索")
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [dense_query_vector],
                "anns_field": "dense_vector",
                "search_params": {"metric_type": "IP", "nprobe": 10},
                "limit": k,
                "output_fields": ["text", "parent_id", "parent_content", "source", "timestamp"],
            }
            if filter_expr:
                search_kwargs["filter"] = filter_expr
            results = self.client.search(**search_kwargs)[0]

        # 后续处理：先对子块重排，再聚合父块（保留重排顺序）
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        logger.info(f"向量召回子块数: {len(sub_chunks)}")

        if len(sub_chunks) > 1:
            pairs = [[query, doc.page_content] for doc in sub_chunks]
            scores = self.reranker.predict(pairs)
            sub_chunks = [doc for _, doc in sorted(zip(scores, sub_chunks), reverse=True)]

        parent_docs = self._get_unique_parent_docs(sub_chunks)
        logger.info(f"去重后父块数: {len(parent_docs)}，返回 Top-{conf.CANDIDATE_M}")
        return parent_docs[:conf.CANDIDATE_M]

    # 定义私有方法，从子块中提取去重的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
        # 返回去重后的父文档列表
        return unique_docs

    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )


if __name__ == '__main__':
    vector_store = VectorStore()
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Administrative_regulations_data')))
    print (vector_store.hybrid_search_with_rerank("云南省实施《中华人民共和国各级人民代表大会常务委员会监督法》办法是什么"))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Constitution_data')))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Judicial_Interpretation_data')))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Law_data')))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Local_regulations_data')))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Supervisory_Law_data')))