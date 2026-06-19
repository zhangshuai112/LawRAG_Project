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
    result = model_split.split_text(text='移动端语音唤醒模型，检测关键词为“小云小云”。模型主体为4层FSMN结构，使用CTC训练准则，参数量750K，适用于移动端设备运行。模型输入为Fbank特征，输出为基于char建模的中文全集token预测，测试工具根据每一帧的预测数据进行后处理得到输入音频的实时检测结果。模型训练采用“basetrain + finetune”的模式，basetrain过程使用大量内部移动端数据，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。后续用户可在basetrain模型基础上，使用其他关键词数据进行微调，得到新的语音唤醒模型，但暂时未开放模型finetune功能。')
    print(result)