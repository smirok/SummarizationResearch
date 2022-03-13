import gzip
import json
import os
from datasets import Dataset


class AbstractDataset:
    def __init__(self, path: str):
        self.path = path
        self.train = None
        self.val = None
        self.test = None


class WcepDataset(AbstractDataset):
    def __init__(self, path: str):
        super().__init__(path)

        self.train = WcepDataset.__wrap_data__(
            list(WcepDataset.__load_jsonl_gz__(os.path.join(path, "train.jsonl.gz"))))
        self.val = WcepDataset.__wrap_data__(list(WcepDataset.__load_jsonl_gz__(os.path.join(path, "val.jsonl.gz"))))
        self.test = WcepDataset.__wrap_data__(list(WcepDataset.__load_jsonl_gz__(os.path.join(path, "test.jsonl.gz"))))

    @staticmethod
    def __load_jsonl_gz__(path):
        with gzip.open(path) as file:
            for line in file:
                yield json.loads(line)

    @staticmethod
    def __wrap_data__(data):
        summaries = list(map(lambda x: x['summary'], data))
        articles = list(
            map(lambda x: '\n\n'.join(list(map(lambda y: y['text'], x))), list(map(lambda x: x['articles'], data))))
        return Dataset.from_dict({'articles': articles, 'summaries': summaries})
