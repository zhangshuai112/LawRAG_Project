#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/11/10 17:05
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm
import os,sys
current_path = os.path.abspath(__file__)
core_path = os.path.join(os.path.dirname(current_path),'core')
sys.path.insert(0,core_path)

sys.path.insert(0,os.path.join(os.path.dirname(current_path),'edu_text_spliter'))
sys.path.insert(0,os.path.join(os.path.dirname(current_path),'edu_document_loaders'))
# print(sys.path)
from .core.vector_store import VectorStore
from .core.rag_system import RAGSystem