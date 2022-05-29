import copy
import os

from dataset import AbstractDataset
from datasets import Dataset


class Duc2004Dataset(AbstractDataset):
    def __init__(
            self,
            path=None,
            multi_articles=False
    ):
        super().__init__(path, multi_articles)

        article_path = os.path.join(path, "DUC2004_Summarization_Documents", "duc2004_testdata", "tasks1and2",
                                    "duc2004_tasks1and2_docs", "docs")
        reference_path = os.path.join(path, "reference")

        article_dirs = sorted([os.path.join(article_path, filename) for filename in os.listdir(article_path)])
        reference_dirs = list(filter(lambda filename: filename.find("reference1") != -1, sorted(
            [os.path.join(reference_path, filename) for filename in os.listdir(reference_path)])))

        articles = [self.__load_article__(article_dir) for article_dir in article_dirs]
        summaries = []
        for reference in reference_dirs:
            with open(reference) as file:
                summaries.append(file.read())

        self.train = Dataset.from_dict({'articles': articles, 'summaries': summaries})
        self.test = copy.deepcopy(self.train)

    def __load_article__(self, path):
        article_parts = []
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r") as file:
                article_parts.append(file.read())

        return self.DOC_SEP.join(article_parts)
