import copy
import os

import numpy as np

from dataset import AbstractDataset
from datasets import Dataset

from model.util.barycenter import BarycenterModel


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

        article_barycenter_tuples_list = [self.__load_article__(article_dir) for article_dir in article_dirs]
        articles, barycenters = map(list, zip(*article_barycenter_tuples_list))
        summaries = []
        for reference in reference_dirs:
            with open(reference) as file:
                summaries.append(file.read().replace('\n', ''))

        self.train = Dataset.from_dict({'articles': articles, 'summaries': summaries, 'barycenters': barycenters})
        self.test = copy.deepcopy(self.train)

    def __load_article__(self, path):
        article_parts = []
        article_barycenters = []
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r") as file:
                text = file.read()
                article_parts.append(text.replace('\n', ''))
                article_barycenters.append(BarycenterModel.calculate_text_barycenter(text))

        return self.DOC_SEP.join(article_parts), BarycenterModel.calculate_vectors_barycenter(
            np.mean(np.array(article_barycenters)))
