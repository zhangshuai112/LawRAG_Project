import os
from typing import Iterator
from edu_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRIMGLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, img_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            img_path: The path to the img to load.
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """

        line = self.img2text()
        yield Document(page_content=line, metadata={"source": self.img_path})

    def img2text(self):
        resp = ""
        ocr = get_ocr()
        #result：检测框 + 文本 + 置信度，elapse_list：各阶段耗时
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            resp += "\n".join(ocr_result)
        return resp


if __name__ == '__main__':
    _samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'samples')
    img_loader = OCRIMGLoader(img_path=os.path.join(_samples_dir, 'ocr_04.png'))
    doc = img_loader.load()
    print(doc)