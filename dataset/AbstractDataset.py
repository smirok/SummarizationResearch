from abc import ABC


class AbstractDataset(ABC):
    def __init__(self, path=None, multi_articles=False):
        self.DOC_SEP = "|||||"
        if not multi_articles:
            self.DOC_SEP = "\n"

        self.path = path
        self.multi_articles = multi_articles
        self.train = None
        self.val = None
        self.test = None
