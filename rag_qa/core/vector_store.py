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
from typing import Dict, List, Optional, Tuple

import jieba
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
# from rank_bm25 import BM25Okapi
import bm25s
from bm25s.tokenization import Tokenizer

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_path)
from base import logger,config
from logger import logger
from langchain_core.documents import Document
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from config import Config
from document_processor import process_documents
conf = Config()

# Milvus query 单次拉取上限（用于构建 BM25 内存索引）
_MILVUS_QUERY_BATCH = 16384
_BM25S_INDEX_DIR = os.path.join(project_path, "rag_qa", "indexes", "bm25s", conf.MILVUS_COLLECTION_NAME)
_BM25S_DELTA_INDEX_DIR = os.path.join(project_path, "rag_qa", "indexes", "bm25s", f"{conf.MILVUS_COLLECTION_NAME}_delta")


def _tokenize(text: str) -> List[str]:
    """中文分词：小写 + jieba，供 BM25 语料与 query 使用。"""
    return jieba.lcut((text or "").lower())


def _rrf_fuse(
    ranked_id_lists: List[List[str]],
    k: int = 60,
    top_k: int = 10,
) -> List[str]:
    """
    Reciprocal Rank Fusion：按 chunk_id 排名融合多路召回，不依赖原始分数尺度。
    RRF_score(d) = sum_i 1 / (k + rank_i(d))，rank 从 0 开始。
    """
    scores: Dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]


# class ChunkBM25Index:
#     """子块级 BM25 内存索引：语料为 Milvus 中所有子块 text 的分词列表。"""
#
#     def __init__(self) -> None:
#         self.chunk_ids: List[str] = []
#         self.tokenized_corpus: List[List[str]] = []
#         self._bm25: Optional[BM25Okapi] = None
#
#     def build(self, chunk_ids: List[str], texts: List[str]) -> None:
#         self.chunk_ids = list(chunk_ids)
#         self.tokenized_corpus = [_tokenize(t) for t in texts]
#         self._bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
#
#     def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
#         if not self._bm25 or not self.chunk_ids:
#             return []
#         query_tokens = _tokenize(query)
#         if not query_tokens:
#             return []
#         scores = self._bm25.get_scores(query_tokens)
#         ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
#         results: List[Tuple[str, float]] = []
#         for idx, score in ranked[:top_k]:
#             if score <= 0:
#                 break
#             results.append((self.chunk_ids[idx], float(score)))
#         return results

class ChunkBM25Index:
    def __init__(self, index_dir: str = _BM25S_INDEX_DIR):
        self.index_dir = index_dir
        self.chunk_ids: List[str] = []
        self.retriever: Optional[bm25s.BM25] = None
        self.corpus_by_id: Dict[str, dict] = {}
        self.tokenizer = Tokenizer(
            lower=True,
            splitter=lambda text: jieba.lcut(text or ""),
            stopwords=[],
            stemmer=None,
        )

    def _has_saved_index(self) -> bool:
        return os.path.exists(os.path.join(self.index_dir, "params.index.json"))

    def _get_chunk_id(self, item) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("id")
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, int) and 0 <= item < len(self.chunk_ids):
            return self.chunk_ids[item]
        return str(item) if item is not None else None

    def load(self) -> bool:
        if not self._has_saved_index():
            return False
        try:
            self.retriever = bm25s.BM25.load(
                self.index_dir,
                mmap=True,
                load_corpus=True,
            )
            self.tokenizer.load_vocab(self.index_dir)
            try:
                self.tokenizer.load_stopwords(self.index_dir)
            except FileNotFoundError:
                pass

            corpus = getattr(self.retriever, "corpus", None) or []
            self.chunk_ids = []
            self.corpus_by_id = {}
            for item in corpus:
                chunk_id = self._get_chunk_id(item)
                if not chunk_id:
                    continue
                self.chunk_ids.append(chunk_id)
                if isinstance(item, dict):
                    self.corpus_by_id[chunk_id] = item
            logger.info(f"BM25S index loaded with mmap: {self.index_dir}, chunks={len(self.chunk_ids)}")
            return True
        except Exception as e:
            logger.warning(f"BM25S index load failed, rebuilding from Milvus: {e}")
            self.retriever = None
            self.chunk_ids = []
            self.corpus_by_id = {}
            return False

    def build_from_rows(self, rows: List[dict], persist: bool = True) -> None:
        rows = [dict(row) for row in rows if row.get("id") and row.get("text")]
        if not rows:
            self.retriever = None
            self.chunk_ids = []
            self.corpus_by_id = {}
            return

        texts = [row["text"] for row in rows]
        self.chunk_ids = [row["id"] for row in rows]
        self.corpus_by_id = {row["id"]: row for row in rows}
        self.tokenizer.reset_vocab()
        corpus_tokens = self.tokenizer.tokenize(
            texts,
            update_vocab=True,
            show_progress=False,
            return_as="tuple",
        )
        self.retriever = bm25s.BM25(method="lucene", corpus=rows)
        self.retriever.index(corpus_tokens, show_progress=False)

        if persist:
            os.makedirs(self.index_dir, exist_ok=True)
            self.retriever.save(self.index_dir, corpus=rows, show_progress=False)
            self.tokenizer.save_vocab(self.index_dir)
            self.tokenizer.save_stopwords(self.index_dir)
            logger.info(f"BM25S index saved: {self.index_dir}")

    def build(self, chunk_ids, texts):
        rows = [{"id": cid, "text": text} for cid, text in zip(chunk_ids, texts)]
        self.build_from_rows(rows)

    def get_meta(self, chunk_id: str) -> Optional[dict]:
        return self.corpus_by_id.get(chunk_id)

    def search(self, query, top_k=10):
        if not self.retriever:
            return []
        query_tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False,
            show_progress=False,
            return_as="tuple",
        )
        if not query_tokens or not query_tokens[0]:
            return []
        results, scores = self.retriever.retrieve(query_tokens, k=top_k, show_progress=False)
        hits = []
        for item, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            chunk_id = self._get_chunk_id(item)
            if not chunk_id:
                continue
            if isinstance(item, dict):
                self.corpus_by_id[chunk_id] = item
            hits.append((chunk_id, float(score)))
        return hits


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
        # RRF 平滑参数
        self.rrf_k = 60
        # BM25 内存索引 + chunk_id -> 元数据（RRF 后还原 Document）
        self.bm25_index = ChunkBM25Index()
        self.bm25_delta_index = ChunkBM25Index(_BM25S_DELTA_INDEX_DIR)
        self._chunk_meta: Dict[str, dict] = {}
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()
        # 从 Milvus 加载子块语料，构建 BM25 索引
        self._load_corpus_from_milvus()

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
        skipped_count = 0
        for i, doc in enumerate(valid_docs):  # ✅ 使用过滤后的列表
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            # row = sparse_matrix.getrow(i)
            # indices = row.indices
            # values = row.data
            # sparse_vector = {idx: value for idx, value in zip(indices, values)}
            # 获取稠密向量（注意字段名是 dense_vecs）
            dense_vec = dense_vectors[i]
            sparse_vector = sparse_vectors[i]

            if not sparse_vector:
                print(f"警告：文档内容 '{doc.page_content[:50]}...' 的稀疏向量为空，已跳过")
                skipped_count += 1
                continue

            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": dense_vec,#embeddings["dense"][i]
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata["parent_id"],
                "parent_content": doc.metadata["parent_content"],
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f"已插入或更新 {len(data)} 个文档，跳过 {skipped_count} 个空稀疏向量文档")
            # 同步更新 BM25 语料与元数据映射
            for item in data:
                self._chunk_meta[item["id"]] = item
            self._rebuild_bm25_delta(data)

    def _load_corpus_from_milvus(self) -> None:
        """从 Milvus 拉取子块，构建 BM25 索引与 chunk_id 元数据映射。"""
        if not self.client.has_collection(self.collection_name):
            return
        main_loaded = self.bm25_index.load()
        delta_loaded = self.bm25_delta_index.load()
        if main_loaded:
            self._chunk_meta = dict(self.bm25_index.corpus_by_id)
            self._chunk_meta.update(self.bm25_delta_index.corpus_by_id)
            return
        rows = []
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=_MILVUS_QUERY_BATCH,
            limit=-1,
            filter="",
            output_fields=["id", "text", "parent_id", "parent_content", "source", "timestamp"],
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()
        if not rows:
            logger.warning("Milvus 中无子块数据，BM25 索引为空")
            return
        self.bm25_index.build_from_rows(rows)
        self._chunk_meta = {row["id"]: row for row in rows}
        if delta_loaded:
            self._chunk_meta.update(self.bm25_delta_index.corpus_by_id)
        logger.info(f"BM25 索引已加载，子块数: {len(rows)}")

    def _rebuild_bm25(self) -> None:
        """根据当前 _chunk_meta 全量重建 BM25 索引。"""
        self.bm25_index.build_from_rows(list(self._chunk_meta.values()))

    def _rebuild_bm25_delta(self, rows: List[dict]) -> None:
        """Only rebuild the small delta index for newly inserted chunks."""
        delta_rows = dict(self.bm25_delta_index.corpus_by_id)
        for row in rows:
            delta_rows[row["id"]] = row
        self.bm25_delta_index.build_from_rows(list(delta_rows.values()))

    def _search_bm25_ids(self, query: str, k: int, source_filter: Optional[str] = None) -> List[str]:
        """BM25 关键词检索，返回 Top-K chunk_id 列表。"""
        search_k = max(k * 3, k)
        main_hits = self.bm25_index.search(query, top_k=search_k)
        delta_hits = self.bm25_delta_index.search(query, top_k=search_k)
        candidate_ids = _rrf_fuse(
            [[chunk_id for chunk_id, _ in main_hits], [chunk_id for chunk_id, _ in delta_hits]],
            k=self.rrf_k,
            top_k=search_k,
        )
        ranked_ids: List[str] = []
        for chunk_id in candidate_ids:
            meta = self.bm25_delta_index.get_meta(chunk_id) or self.bm25_index.get_meta(chunk_id)
            if meta:
                self._chunk_meta[chunk_id] = meta
            if source_filter:
                meta = self._chunk_meta.get(chunk_id, {})
                if meta.get("source") != source_filter:
                    continue
            ranked_ids.append(chunk_id)
            if len(ranked_ids) >= k:
                break
        return ranked_ids

    def _search_hybrid_ids(self, query: str, k: int, source_filter: Optional[str] = None) -> List[str]:
        """
        BGE-M3 稀疏 + 稠密 Milvus 混合检索（WeightedRanker），
        返回按混合分排序的 chunk_id 列表。
        """
        query_embeddings = self.embedding_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
        )
        dense_query_vector = query_embeddings["dense_vecs"][0]
        sparse_query_vector = query_embeddings["lexical_weights"][0]
        filter_expr = f"source == '{source_filter}'" if source_filter else None

        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "nprobe": 10},
            limit=k,
            expr=filter_expr,
        )

        if sparse_query_vector:
            sparse_request = AnnSearchRequest(
                data=[sparse_query_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=k,
                expr=filter_expr,
            )
            ranker = WeightedRanker(1.0, 0.7)
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=ranker,
                limit=k,
                output_fields=["id","text", "parent_id", "parent_content", "source", "timestamp"],
            )[0]
        else:
            logger.warning(f"查询 '{query}' 稀疏向量为空，向量路退化为仅稠密检索")
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [dense_query_vector],
                "anns_field": "dense_vector",
                "search_params": {"metric_type": "IP", "nprobe": 10},
                "limit": k,
                "output_fields": ["id", "text", "parent_id", "parent_content", "source", "timestamp"],
            }
            if filter_expr:
                search_kwargs["filter"] = filter_expr
            results = self.client.search(**search_kwargs)[0]

        ranked_ids = []
        for hit in results:
            entity = hit.get("entity", hit)
            chunk_id = entity.get("id") or hit.get("id")
            if not chunk_id:
                continue
            self._chunk_meta[chunk_id] = {
                "id": chunk_id,
                "text": entity.get("text"),
                "parent_id": entity.get("parent_id"),
                "parent_content": entity.get("parent_content"),
                "source": entity.get("source"),
                "timestamp": entity.get("timestamp"),
            }
            ranked_ids.append(chunk_id)
        return ranked_ids



    def _ids_to_documents(self, chunk_ids: List[str]) -> List[Document]:
        """根据 RRF 融合后的 chunk_id 列表还原子块 Document。"""
        docs: List[Document] = []
        for cid in chunk_ids:
            meta = self._chunk_meta.get(cid)
            if not meta:
                continue
            docs.append(
                Document(
                    page_content=meta.get("text", ""),
                    metadata={
                        "id": cid,
                        "parent_id": meta.get("parent_id"),
                        "parent_content": meta.get("parent_content"),
                        "source": meta.get("source"),
                        "timestamp": meta.get("timestamp"),
                    },
                )
            )
        return docs

    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None) -> List[Document]:
        """
        多路检索 + RRF 融合 + bge-reranker 精排：

            BM25 Top-K ─────────────┐
                                    ├→ RRF → 父块去重 → bge-reranker → Top-M
            BGE 稀疏+稠密 hybrid Top-K ─┘
        """
        if not query or not str(query).strip():
            logger.error("query 为空")
            return []

        # 1. BM25 关键词路 + BGE 稀疏/稠密混合向量路，各自召回 Top-K chunk_id
        bm25_ids = self._search_bm25_ids(query, k=k, source_filter=source_filter)
        hybrid_ids = self._search_hybrid_ids(query, k=k, source_filter=source_filter)
        logger.info(f"BM25 召回 {len(bm25_ids)} 条，混合向量召回 {len(hybrid_ids)} 条")

        # 2. RRF 融合两路 chunk_id 排名（非 Document 列表拼接）
        fused_ids = _rrf_fuse([bm25_ids, hybrid_ids], k=self.rrf_k, top_k=k)
        logger.info(f"RRF 融合后保留 {len(fused_ids)} 条子块")

        # 3. 还原子块 Document，并按 parent_id 去重为父块
        sub_chunks = self._ids_to_documents(fused_ids)
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        logger.info(f"父块去重后 {len(parent_docs)} 条")

        # 4. bge-reranker 对父块精排，返回 Top-M
        if len(parent_docs) < 5:
            return parent_docs[:conf.CANDIDATE_M]

        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs)
        ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        return ranked_parent_docs[:conf.CANDIDATE_M]

    # 定义私有方法，从子块中提取去重的父文档
    def _get_unique_parent_docs(self, sub_chunks):
        # 按 parent_id 去重（无 parent_id 时回退 parent_content）
        seen = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_id = chunk.metadata.get("parent_id")
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            dedup_key = parent_id or parent_content
            if dedup_key and dedup_key not in seen:
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                seen.add(dedup_key)
        return unique_docs

    # 定义私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "id": hit.get("id"),
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
            },
        )


if __name__ == '__main__':
    vector_store = VectorStore()
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Administrative_regulations_data')))
    # print (vector_store.hybrid_search_with_rerank("关于虚拟财产继承有哪些规定？"))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Constitution_data')))
    # vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Judicial_Interpretation_data')))
    vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Law_data')))
    vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Local_regulations_data')))
    vector_store.add_documents(process_documents(os.path.join(conf.DATA_DIR, 'Supervisory_Law_data')))
