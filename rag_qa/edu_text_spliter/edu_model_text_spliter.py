from langchain_text_splitters import CharacterTextSplitter
import re
import os
import sys
from typing import List
from modelscope.pipelines import pipeline
from modelscope.pipelines.nlp.document_segmentation_pipeline import DocumentSegmentationPipeline

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_path)
from base.config import Config

conf = Config()

# modelscope>=1.35 会把 trust_remote_code 传入 preprocessor，但该类不接受该参数
_orig_doc_seg_init = DocumentSegmentationPipeline.__init__


def _compat_doc_seg_init(self, model, preprocessor=None, config_file=None,
                         device='gpu', auto_collate=True, **kwargs):
    kwargs.pop('trust_remote_code', None)
    return _orig_doc_seg_init(
        self, model, preprocessor=preprocessor, config_file=config_file,
        device=device, auto_collate=auto_collate, **kwargs
    )


if not getattr(DocumentSegmentationPipeline.__init__, '_compat_patched', False):
    _compat_doc_seg_init._compat_patched = True
    DocumentSegmentationPipeline.__init__ = _compat_doc_seg_init

_DOC_SEG_PIPELINE = None
_DOC_SEG_MODEL_PATH = None


def _get_doc_segmentation_pipeline(model_path: str, device: str = "cpu"):
    global _DOC_SEG_PIPELINE, _DOC_SEG_MODEL_PATH
    if _DOC_SEG_PIPELINE is None or _DOC_SEG_MODEL_PATH != model_path:
        _DOC_SEG_PIPELINE = pipeline(
            task="document-segmentation",
            model=model_path,
            device=device,
        )
        _DOC_SEG_MODEL_PATH = model_path
    return _DOC_SEG_PIPELINE


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            #作用：将连续 3 个或以上的换行符（\n）替换为 1 个换行符。
            text = re.sub(r"\n{3,}", r"\n", text)
            #将所有空白字符（包括空格、制表符、换行符等）替换为 单个空格。
            text = re.sub(r'\s', " ", text)
            #删除 两个连续的换行符（即空行）。
            text = re.sub(r"\n\n", "", text)
        p = _get_doc_segmentation_pipeline(
            conf.DOCUMENT_SEGMENTATION_MODEL,
            device="cuda",
        )
        result = p(documents=text)
        #将模型输出的分段文本按 \n\t 分割，过滤空字符串。
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
if __name__ == '__main__':
    model_split = AliTextSplitter()
    result = model_split.split_text(text='上一节将成本列为模型选型的关键维度之一，但Agent 场景下的成本远比简单的 token 定价复杂——多轮推理、工具调用和上下文累积会使成本呈非线性增长。系统性的成本分析是评估体系不可或缺的一环，也是生产部署的必要前提。成本的构成要素。Agent 系统的成本可分解为三个层次：模型推理成本是最直接的部分，由输入token和输出token的消耗决定。但Agent场景下有两个常被忽视的放大因素。一是上下文累积效应：Agent每轮调用LLM时，都会把之前所有的对话历史和工具返回结果一起发送（这样模型才能理解上下文）。如果没有利用好KVCache（即缓存已处理过的上下文，避免重复计算），成本增长会非常快——第1轮发送1000token，第2轮发送2000token，第3轮发送3000token，总量是1000+2000+3000=6000而非3×1000=3000，轮次越多差距越大。二是思考token成本：支持思考的模型会生成大量思考token，这些token虽然不展示给用户，但同样计入费用。工具调用成本包括外部API费用（搜索引擎按次计费、数据库查询消耗计算资源）、代码执行的沙盒资源，以及一个容易被忽视的间接成本：工具返回结果注入上下文后产生的token费用。一次网页搜索返回的内容可能就占用2000-5000 个 token，而且在后续每轮推理中都会作为输入被反复计费。')
    print(result)