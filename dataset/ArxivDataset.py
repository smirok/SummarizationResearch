import json
import os

from AbstractDataset import AbstractDataset
from datasets import Dataset


class ArxivDataset(AbstractDataset):
    def __init__(self, path=None, multi_articles=False):
        super().__init__(path, multi_articles)

        self.train = None  # TODO: dataset size is about 14 gb

        with open(os.path.join(path, "arxiv-dataset", "val.txt")) as file:
            lines = file.readlines()
            articles = list(map(lambda elem: self.DOC_SEP.join(json.loads(elem)['article_text']), lines))
            summaries = list(map(lambda elem: '\n'.join(json.loads(elem)['abstract_text']), lines))
            self.val = Dataset.from_dict({'articles': articles, 'summaries': summaries})

        with open(os.path.join(path, "arxiv-dataset", "test.txt")) as file:
            lines = file.readlines()
            articles = list(map(lambda elem: self.DOC_SEP.join(json.loads(elem)['article_text']), lines))
            summaries = list(map(lambda elem: '\n'.join(json.loads(elem)['abstract_text']), lines))
            self.test = Dataset.from_dict({'articles': articles, 'summaries': summaries})
