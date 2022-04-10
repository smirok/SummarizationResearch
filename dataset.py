import gzip
import json
import os
from datasets import Dataset, load_dataset
import copy


def delete_last_sep(example):
    example['document'] = example['document'][:-5]
    return example


#def preprocess(example):
#    example['articles'] = ' '.join(example['articles'].replace('\n', ' ').split())
#    return example


def process_separator(example, doc_sep):
    example['articles'] = example['articles'].replace(doc_sep, ' ')
    return example


class AbstractDataset:
    def __init__(self, path=None, multi_articles=False):
        self.DOC_SEP = "|||||"
        if not multi_articles:
            self.DOC_SEP = "\n"

        self.path = path
        self.multi_articles = multi_articles
        self.train = None
        self.val = None
        self.test = None


class WcepDataset(AbstractDataset):
    def __init__(self, path=None, multi_articles=False):
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

        return Dataset.from_dict({'articles': articles, 'summaries': summaries})


class Duc2004Dataset(AbstractDataset):
    def __init__(self, path=None, multi_articles=False):
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


class MultiNewsDataset(AbstractDataset):
    def __init__(self, path=None, multi_articles=True):
        super().__init__(path, multi_articles)

        dataset = load_dataset('multi_news', ignore_verifications=True, download_mode='force_redownload')
        dataset = dataset.rename_column('document', 'articles').rename_column('summary', 'summaries').map(
            delete_last_sep
        )

        if not multi_articles:
            dataset = dataset.map(lambda elem: process_separator(elem, self.DOC_SEP))

        self.train = dataset['train']
        self.val = dataset['validation']
        self.test = dataset['test']


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
