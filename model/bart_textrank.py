from datasets import Dataset

from model import AbstractModel
from model import BartModel
from model import TextRankModel


class BartTextRankModel(AbstractModel):
    def __init__(
            self,
            max_target_length=64,
            max_source_length=1024,
            save_path="./bart-textrank-model/",
            model_checkpoint="facebook/bart-base",
            tokenizer_checkpoint="facebook/bart-base",
            epochs=3,
            batch_size=16
    ):
        super().__init__(
            max_target_length=max_target_length,
            max_source_length=max_source_length,
            save_path=save_path
        )

        self.epochs = epochs
        self.batch_size = batch_size

        self.bart = BartModel(
            max_target_length=max_target_length,
            max_source_length=max_source_length,
            model_checkpoint=model_checkpoint,
            tokenizer_checkpoint=tokenizer_checkpoint,
            epochs=epochs,
            batch_size=batch_size
        )
        self.textrank = TextRankModel(max_target_length, max_source_length)

    def train(self, train_dataset, val_dataset=None):
        self.bart.train(train_dataset, val_dataset)

    def predict(self, test_dataset):
        test = test_dataset.map(lambda dataset: self.bart.__preprocess_function__(dataset, self.bart.tokenizer,
                                                                                  self.bart.max_source_length),
                                batched=True)

        bart_predictions = [self.bart.tokenizer.decode(prediction, skip_special_tokens=True) for prediction in
                            self.bart.predict(test).predictions]
        textrank_predictions = self.textrank.predict(test_dataset)

        test = Dataset.from_dict(
            {'articles': [bart_predictions[i] + "  " + textrank_predictions[i] for i in range(len(test_dataset))],
             'summaries': test['summaries']})

        return self.bart.predict(test)
