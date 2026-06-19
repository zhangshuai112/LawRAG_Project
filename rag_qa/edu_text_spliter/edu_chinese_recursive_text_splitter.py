import re  # 导入正则表达式模块，用于按分隔符拆分文本、匹配分隔符
from typing import List, Optional, Any  # 类型注解：列表、可选类型、任意类型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 继承 LangChain 的递归字符文本分割器
import logging  # 日志模块（本文件已配置 logger，但当前逻辑未使用）

logger = logging.getLogger(__name__)  # 以当前模块名为 logger 名称，便于按模块过滤日志


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """
    使用正则分隔符从文本末尾方向拆分（由调用方传入已处理好的 separator 模式）。
    keep_separator=True 时，分隔符会保留在相邻片段的拼接结果中。
    """
    if separator:  # 若提供了非空分隔符模式
        if keep_separator:  # 需要保留分隔符在切分结果里
            # 用捕获组包住分隔符，re.split 会把分隔符也作为独立元素放进列表
            _splits = re.split(f"({separator})", text)
            # 奇偶位配对：正文 + 分隔符 拼成一段，例如 ["段落1", "。", "段落2"] -> ["段落1。", "段落2"]
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:  # 若列表长度为奇数，说明末尾还有一段没有配对的分隔后文本
                splits += _splits[-1:]  # 把最后一段单独追加进去
            # splits = [_splits[0]] + splits  # 历史写法，已注释
        else:  # 不保留分隔符，直接按模式切开
            splits = re.split(separator, text)
    else:  # 分隔符为空字符串时，退化为按单个字符拆分
        splits = list(text)
    return [s for s in splits if s != ""]  # 过滤掉空字符串片段后返回


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """面向中文的递归文本分割器：优先按段落、换行、中文句号等层级切分，再合并到 chunk_size。"""

    def __init__(
            self,
            separators: Optional[List[str]] = None,  # 自定义分隔符列表；None 时使用下方默认中文友好规则
            keep_separator: bool = True,  # 切分时是否把分隔符保留在文本块中
            is_separator_regex: bool = True,  # True：分隔符按正则解释；False：按普通字符串并自动转义
            **kwargs: Any,  # 其余参数传给父类，如 chunk_size、chunk_overlap、length_function 等
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)  # 初始化父类（块大小、重叠等配置在父类中）
        self._separators = separators or [  # 未传入则使用默认分隔符优先级（从粗到细）
            r"\n\n",  # 双换行：段落级
            r"\n",  # 单换行：行级
            r"。|！|？",  # 中文句号、叹号、问号
            r"\.\s|\!\s|\?\s",  # 英文句号/叹号/问号后常跟空格
            r"；|;\s",  # 中文分号或英文分号加可选空格
            r"，|,\s"  # 中文逗号或英文逗号加可选空格（最细一级）
        ]
        self._is_separator_regex = is_separator_regex  # 保存是否按正则处理分隔符

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分入口：为当前 text 选出最合适的分隔符，切分后再合并或继续细分。"""
        final_chunks = []  # 存放最终返回的文本块列表
        separator = separators[-1]  # 默认用最细的分隔符（列表最后一个）
        new_separators = []  # 若当前分隔符能匹配，则更细一级的分隔符列表（用于递归）
        for i, _s in enumerate(separators):  # 从粗到细遍历分隔符
            _separator = _s if self._is_separator_regex else re.escape(_s)  # 非正则模式时对特殊字符转义
            if _s == "":  # 空分隔符表示“按字符切”，直接选用并结束查找
                separator = _s
                break
            if re.search(_separator, text):  # 当前文本中是否存在该级分隔符
                separator = _s  # 选用这一级作为本次切分符
                new_separators = separators[i + 1:]  # 更长的片段将用后续更细的分隔符递归处理
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)  # 得到实际用于 split 的模式
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)  # 按选定分隔符切成多段

        _good_splits = []  # 暂存长度未超 chunk_size 的片段，等待合并
        _separator = "" if self._keep_separator else separator  # 合并时用的连接符：保留分隔符则不再额外插入
        for s in splits:  # 遍历每一段切分结果
            if self._length_function(s) < self._chunk_size:  # 片段长度小于块上限，先缓存
                _good_splits.append(s)
            else:  # 片段仍然过长，需要处理
                if _good_splits:  # 先把之前缓存的短片段合并成块并写入结果
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:  # 没有更细的分隔符了，只能原样保留（可能仍超长）
                    final_chunks.append(s)
                else:  # 用更细的分隔符对该长片段递归切分
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:  # 循环结束后若还有未合并的短片段，再合并一次
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        # 去掉块首尾空白，把连续多个换行压成一个换行，并丢弃空块
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


if __name__ == "__main__":  # 直接运行本文件时执行下方测试代码
    text_splitter = ChineseRecursiveTextSplitter(  # 创建中文递归分割器实例
        keep_separator=True,  # 切分后保留句号、换行等分隔符
        is_separator_regex=True,  # 默认分隔符按正则使用
        chunk_size=150,  # 每个块的目标最大长度（由 length_function 计量，一般为字符数）
        chunk_overlap=10  # 相邻块之间的重叠长度，便于 RAG 检索时保留上下文
    )
    ls = [  # 待切分的示例文本列表
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
    ]
    # text = """"""  # 可在此替换为自定义测试文本
    for inum, text in enumerate(ls):  # 遍历每条示例文本
        print(inum)  # 打印当前文本在列表中的序号

        chunks = text_splitter.split_text(text)  # 调用 LangChain 对外 API，内部会走到 _split_text
        for i, chunk in enumerate(chunks):  # 逐块打印切分结果
            print(f"第{i}块是:\n{chunk}")
