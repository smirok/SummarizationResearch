import gzip
import json
import os

from datasets import Dataset

from dataset import AbstractDataset
from model.util.barycenter import BarycenterModel


class WcepDataset(AbstractDataset):
    def __init__(
            self,
            path=None,
            multi_articles=False
    ):
        super().__init__(path, multi_articles)

        self.train = self.__wrap_data__(
            list(WcepDataset.__load_jsonl_gz__(os.path.join(path, "train.jsonl.gz"))))
        self.val = self.__wrap_data__(list(WcepDataset.__load_jsonl_gz__(os.path.join(path, "val.jsonl.gz"))))
        self.test = self.__wrap_data__(list(WcepDataset.__load_jsonl_gz__(os.path.join(path, "test.jsonl.gz"))))

    @staticmethod
    def __load_jsonl_gz__(path):
        with gzip.open(path) as file:
            for line in file:
                yield json.loads(line)

    def __wrap_data__(self, data):
        summaries = list(map(lambda x: x['summary'], data))

        articles = list(
            map(lambda x: self.DOC_SEP.join(list(map(lambda y: y['text'], x))),
                list(map(lambda x: x['articles'], data))))
        barycenters = list(
            map(lambda x: BarycenterModel.calculate_texts_barycenter(list(map(lambda y: y['text'], x))),
                list(map(lambda x: x['articles'], data))))

        return Dataset.from_dict({'articles': articles, 'summaries': summaries, 'barycenters': barycenters})
