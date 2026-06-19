#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/10/29 19:02
# @Site    : 
# @File    : query_classifier.py
# @Software: PyCharm
# 导入标准库
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import os,sys
# 导入 PyTorch
import torch
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,project_path)
# 导入日志
from base import logger
from base.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = Config()

class QueryClassifier:
    def __init__(self, model_save_path=None):
        self.model_save_path = model_save_path or os.path.join(
            conf.RAG_QA_ROOT, 'models', 'bert_query_classifier'
        )
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained(conf.BERT_MODEL)
        self.device = device
        # 记录设备信息
        logger.info(f"使用设备: {self.device}")
        self.label_map = {'通用知识':0 ,'专业咨询' :1 }
        self.load_model()
    def load_model(self):
        #todo:检查模型是否存在，存在就直接按照路径加载，不存在就按照预训练模型路径加载，用日志记录是否成功并最后送到设备里面
        if os.path.exists(self.model_save_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_save_path)
            self.model.to(device)
            logger.info('模型加载成功')
        else:
            self.model = BertForSequenceClassification.from_pretrained(conf.BERT_MODEL, num_labels=2)
            self.model.to(device)
            logger.info('模型初始化成功')
    def save_model(self):
        self.model.save_pretrained(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        logger.info(f'模型保存成功,保存到{self.model_save_path}')

    def create_datasets(self,text, label):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self,texts, labels):
                super().__init__()
                self.texts = texts
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self,idx):
                item= { k:value[idx]   for k,value in self.texts.items()}
                item['labels'] =torch.tensor(self.labels[idx])
                return item

        return Dataset(text, label)

    def preprocess_data(self,texts, labels):
        encoding = self.tokenizer(texts, padding='max_length', truncation=True, max_length=128,return_tensors="pt")
        labels_num = [self.label_map[i]  for i in labels]
        return encoding,labels_num


    def compute_metrics(self,eval_pred):
        texts,labels =eval_pred
        predictions = np.argmax(texts, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    def evaluate_model(self,data_path):
        with open(data_path, 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        texts = [i['query'] for i in data]
        labels = [i['label'] for i in data]
        encoding,labels_num = self.preprocess_data(texts,labels)
        # print(f'encoding-->{encoding}')
        dataset = self.create_datasets(encoding,labels_num)
        train = Trainer(model=self.model)
        pre_labels = np.argmax(train.predict(dataset).predictions,axis=-1)
        # print(f'label_num->{labels_num}')
        logger.info('分类报告：')
        logger.info(classification_report(labels_num, pre_labels,target_names=["通用知识", "专业咨询"]))
        logger.info('混淆矩阵：')
        logger.info(confusion_matrix(labels_num, pre_labels))



    def train_model(self,data_path='../data/law_bert_data.json'):
        if not os.path.exists(data_path):
            logger.info("数据加载失败,模型退出训练")
            raise FileNotFoundError(f"数据集文件 {data_path} 不存在")
        # with open(data_path,'r',encoding='utf-8-sig') as f:
        #     data=[json.loads(docs.strip()) for docs in f.readlines() if docs.strip()]
        with open(data_path, 'r', encoding='utf-8-sig') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]

        texts = [i['query'] for i in data]
        labels = [i['label'] for i in data]
        x_train,x_test,y_train,y_test = train_test_split(texts,labels,test_size=0.2,random_state=42)

        x_train_p,y_train_p = self.preprocess_data(x_train,y_train)
        x_test_p,y_test_p = self.preprocess_data(x_test,y_test)

        #todo:构建datasets
        train_dataset = self.create_datasets( x_train_p,y_train_p )
        val_dataset = self.create_datasets( x_test_p,y_test_p )


        # print(f'train_dataset->{train_dataset}')
        #todo:开始训练
        training_args = TrainingArguments(
            output_dir="./bert_results",#输出目录：模型检查点、训练日志等所有文件将保存到此文件夹
            num_train_epochs=3,#训练轮数：整个训练数据集将被完整遍历 3 次
            per_device_train_batch_size=8,#每设备训练批次大小：每个 GPU（或 CPU）上每次前向/后向传播使用的样本数。总批次大小 = 设备数 × 这个值
            per_device_eval_batch_size=8,#每设备评估批次大小
            warmup_steps=50,#预热步数
            weight_decay=0.01,#权重衰减
            logging_dir="./bert_logs",#日志目录
            logging_steps=10,#日志记录间隔：每 10 个训练步记录一次损失值等指标到 logging_dir。
            eval_strategy="epoch",#评估策略：每个 epoch 结束后在验证集上评估一次。也可设为 "steps" 按步数评估。
            save_strategy="epoch",#保存策略：每个 epoch 结束后保存一次模型检查点。与 eval_strategy 保持一致有助于同步
            load_best_model_at_end=True,# 结束时加载最优模型：训练结束后，自动加载在验证集上指标最好的那个检查点，而不是最后一个 epoch 的模型。
            save_total_limit=1,  # 只保存一个检查点，即最优的模型
            metric_for_best_model="eval_loss",#最优模型判断指标：根据验证集上的 eval_loss
            fp16=True, # 启用 16 位浮点数计算，可以加速训练并减少显存占用
            learning_rate=3e-5,
        )

        # 初始化 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        # 训练模型
        logger.info("开始训练 BERT 模型...")
        trainer.train()
        self.save_model()
        self.evaluate_model(data_path='../data/model_test_data')
    def predict(self,query):
        if self.model==None:
            logger.info("模型未训练")
        # train = Trainer(self.model)
        encoding = self.tokenizer(query, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # print(f'outputs->{outputs}')
            # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"
if __name__ == '__main__':
    # QueryClassifier().train_model(data_path='../data/law_bert_data.json')
    #
    # QueryClassifier().train_model()
    print(QueryClassifier().predict('我如果开车创到人，会被怎么处罚？'))