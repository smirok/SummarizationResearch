from AbstractDataset import AbstractDataset
from datasets import load_dataset

from util import delete_last_sep, process_separator


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