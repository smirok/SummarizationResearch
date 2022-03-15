import gzip
import json
import os
from datasets import Dataset, load_dataset
import copy


class AbstractDataset:
    def __init__(self, path=None):
        self.path = path
        self.train = None
        self.val = None
        self.test = None


class WcepDataset(AbstractDataset):
    def __init__(self, path=None):
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


class Duc2004Dataset(AbstractDataset):
    def __init__(self, path=None):
        super().__init__(path)

        article_path = os.path.join(path, "DUC2004_Summarization_Documents", "duc2004_testdata", "tasks1and2",
                                    "duc2004_tasks1and2_docs", "docs")
        reference_path = os.path.join(path, "reference")

        article_dirs = sorted([os.path.join(article_path, filename) for filename in os.listdir(article_path)])
        reference_dirs = list(filter(lambda filename: filename.find("reference1") != -1, sorted(
            [os.path.join(reference_path, filename) for filename in os.listdir(reference_path)])))

        articles = [Duc2004Dataset.__load_article__(article_dir) for article_dir in article_dirs]
        summaries = []
        for reference in reference_dirs:
            with open(reference) as file:
                summaries.append(file.read())

        self.train = Dataset.from_dict({'articles': articles, 'summaries': summaries})
        self.test = copy.deepcopy(self.train)

    @staticmethod
    def __load_article__(path):
        article_parts = []
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r") as file:
                article_parts.append(file.read())

        return "\n".join(article_parts)


class MultiNewsDataset(AbstractDataset):
    def __init__(self, path=None):
        super().__init__(path)

        dataset = load_dataset('multi_news')
        dataset = dataset.rename_column('document', 'articles')
        dataset = dataset.rename_column('summary', 'summaries')

        self.train = dataset['train']
        self.val = dataset['validation']
        self.test = dataset['test']


class ArxivDataset(AbstractDataset):
    def __init__(self, path=None):
        super().__init__(path)

        self.train = None  # TODO: dataset size is about 14 gb

        with open(os.path.join(path, "arxiv-dataset", "val.txt")) as file:
            lines = file.readlines()
            articles = list(map(lambda elem: '\n'.join(json.loads(elem)['article_text']), lines))
            summaries = list(map(lambda elem: '\n'.join(json.loads(elem)['abstract_text']), lines))
            self.val = Dataset.from_dict({'articles': articles, 'summaries': summaries})

        with open(os.path.join(path, "arxiv-dataset", "test.txt")) as file:
            lines = file.readlines()
            articles = list(map(lambda elem: '\n'.join(json.loads(elem)['article_text']), lines))
            summaries = list(map(lambda elem: '\n'.join(json.loads(elem)['abstract_text']), lines))
            self.test = Dataset.from_dict({'articles': articles, 'summaries': summaries})
