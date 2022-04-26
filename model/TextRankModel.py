from datasets import Dataset
from model import AbstractModel
from summa.summarizer import summarize
from transformers import AutoTokenizer


class TextRankModel(AbstractModel):
    def __init__(self, max_target_length, max_source_length):
        super().__init__()
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        self.test_dataset = None

    def train(self, train_dataset, val_dataset=None):
        pass

    def predict(self, test_dataset):
        model_inputs = self.tokenizer(
            test_dataset['articles'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True
        )
        articles = self.tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)

        model_inputs = self.tokenizer(test_dataset['summaries'])
        summaries = self.tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)

        self.test_dataset = Dataset.from_dict({'articles': articles, 'summaries': summaries})

        if self.max_source_length is not None:
            return [summarize(article, words=self.max_target_length) for article in
                    self.test_dataset['articles']]
        else:
            return [summarize(article, words=self.max_target_length) for article in self.test_dataset['articles']]
