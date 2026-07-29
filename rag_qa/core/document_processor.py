#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 19:09
# @Site    : 
# @File    : document_processor.py
# @Software: PyCharm
import os
from xml.dom.minidom import Document

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter
from datetime import datetime
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))

rag_path = os.path.dirname(dir_path)
project_path=os.path.dirname(rag_path)
sys.path.insert(0,rag_path)
sys.path.insert(0,project_path)
base_path = os.path.join(project_path,'base')
sys.path.insert(0,base_path)
from edu_text_spliter import AliTextSplitter, ChineseRecursiveTextSplitter
from edu_document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base import logger, Config

conf = Config()
# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path)->list[Document]:
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 获取支持的文件扩展名集合
    supported_extensions = document_loaders.keys()
    # 从目录名提取学科类别（如 "ai_data" -> "ai"）
    source = os.path.basename(directory_path).replace("_data", "")

    # 遍历指定目录及其子目录
    for root, _, files in os.walk(directory_path):
        # 遍历当前目录下的所有文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件扩展名并转换为小写
            file_extension = os.path.splitext(file_path)[1].lower()
            # 检查文件类型是否在支持的扩展名列表中
            if file_extension in supported_extensions:
                # 使用 try-except 捕获加载过程中的异常
                try:
                    # 根据文件扩展名获取对应的加载器类
                    loader_class = document_loaders[file_extension]
                    # 实例化加载器对象，传入文件路径
                    if file_extension == ".txt":
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    # 调用加载器加载文档内容，返回文档列表
                    loaded_docs = loader.load()
                    # 遍历加载的每个文档
                    for doc in loaded_docs:
                        # 为文档添加学科类别元数据
                        doc.metadata["source"] = source
                        # 为文档添加文件路径元数据
                        doc.metadata["file_path"] = file_path
                        # 为文档添加当前时间戳元数据
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    # 将加载的文档添加到总列表中
                    documents.extend(loaded_docs)
                    # 记录成功加载文件的日志
                    logger.info(f"成功加载文件: {file_path}")
                # 捕获加载过程中可能出现的异常
                except Exception as e:
                    # 记录加载失败的日志，包含错误信息
                    logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            # 如果文件类型不在支持列表中
            else:
                # 记录警告日志，提示不支持的文件类型
                logger.warning(f"不支持的文件类型: {file_path}")
    # 返回加载的所有文档列表
    return documents



def  process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP):
    #加载文档
    documents = load_documents_from_directory(directory_path)
    logger.info(f'加载的文档总数为：{len(documents)}')
    #todo:初始化父块分割器和子块分割器
    parent_splitter = AliTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
    parent_splitter_pdf = AliTextSplitter(pdf=True, chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)
    #todo:初始化markdown文本分割器
    child_markdown_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)
    parent_markdown_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
    #todo:开始文本分割,splitext
    child_list =[]
    for i,doc in enumerate(documents):
        #选择使用的分割器
        if os.path.splitext(doc.metadata.get('file_path',''))[1].lower()=='.md':
            logger.info(f'正在处理文件：{doc.metadata.get("file_path")}')
            parent_true_splitter =parent_markdown_splitter
            child_true_splitter =child_markdown_splitter
        elif os.path.splitext(doc.metadata.get('file_path',''))[1].lower()=='.pdf':
            parent_true_splitter =parent_splitter_pdf
            child_true_splitter =child_splitter
        else:
            parent_true_splitter =parent_splitter
            child_true_splitter = child_splitter
        #开始分割
        parents = parent_true_splitter.split_documents([doc])

        #遍历每一个父块，将一个父块分割成多个子块
        for j,parent in enumerate(parents):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f"doc_{i}_parent_{j}"
            # 将父块 ID 添加到元数据
            parent.metadata["parent_id"] = parent_id
            # 将父块内容存储到元数据
            parent.metadata["parent_content"] = parent.page_content
            # 开始子块分割
            children = child_true_splitter.split_documents([parent])
            for h,child in enumerate(children):
                # 为子块添加父块 ID 到元数据
                child.metadata["parent_id"] = parent_id
                # 为子块添加父块内容到元数据
                child.metadata["parent_content"] = parent.page_content
                # 为子块生成唯一 ID，格式为 "parent_id_child_h"
                child.metadata["id"] = f"{parent_id}_child_{h}"
                # 将子块添加到子块列表中
                child_list.append(child)
    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_list)}")
    # 返回所有子块列表
    return child_list
if __name__ == '__main__':
   print(process_documents(os.path.join(Config().DATA_DIR, 'Law_data')))