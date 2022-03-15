from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
from summa.summarizer import summarize

import numpy as np
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


class AbstractModel:
    def __init__(self):
        pass

    def train(self, train_dataset, val_dataset=None):
        raise NotImplementedError()

    def predict(self, test_dataset):
        raise NotImplementedError()


class BartModel(AbstractModel):
    def __init__(self, model_checkpoint="facebook/bart-base"):
        super().__init__()
        self.MAX_SOURCE_LENGTH = 256
        self.MAX_TARGET_LENGTH = 32

        self.batch_size = 16
        self.num_train_epochs = 4
        self.rouge_score = load_metric("rouge")
        self.trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    def train(self, train_dataset, val_dataset=None):
        train = train_dataset.map(lambda dataset: self.__preprocess_function__(dataset, self.tokenizer,
                                                                               self.MAX_SOURCE_LENGTH,
                                                                               self.MAX_TARGET_LENGTH), batched=True)
        if val_dataset is not None:
            val = val_dataset.map(lambda dataset: self.__preprocess_function__(dataset, self.tokenizer,
                                                                               self.MAX_SOURCE_LENGTH,
                                                                               self.MAX_TARGET_LENGTH), batched=True)
        else:
            val = None

        logging_steps = len(train) // self.batch_size

        args = Seq2SeqTrainingArguments(
            output_dir=f"multidoc-bart",
            evaluation_strategy="epoch",
            learning_rate=5.6e-6,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=self.num_train_epochs,
            predict_with_generate=True,
            logging_steps=logging_steps,
            push_to_hub=False,
            disable_tqdm=True
        )

        self.trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=train,
            eval_dataset=val,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def predict(self, test_dataset):
        test = test_dataset.map(lambda dataset: self.__preprocess_function__(dataset, self.tokenizer,
                                                                             self.MAX_SOURCE_LENGTH,
                                                                             self.MAX_TARGET_LENGTH), batched=True)

        return self.trainer.predict(test)

    @staticmethod
    def __preprocess_function__(dataset, tokenizer, max_source_length, max_target_length):
        model_inputs = tokenizer(dataset['articles'], max_length=max_source_length, padding='max_length',
                                 truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(dataset['summaries'], max_length=max_target_length, padding='max_length',
                               truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    @staticmethod
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence

        decoded_preds, decoded_labels = BartModel.postprocess_text(decoded_preds, decoded_labels)

        # Compute ROUGE scores
        result = self.rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}


class TextRank(AbstractModel):
    def __init__(self, n_words=64, max_source_length=None):
        super().__init__()
        self.n_words = n_words
        self.max_source_length = max_source_length

    def train(self, train_dataset, val_dataset=None):
        pass

    def predict(self, test_dataset):
        if self.max_source_length is not None:
            return [summarize(article[:self.max_source_length], words=self.n_words) for article in
                    test_dataset['articles']]
        else:
            return [summarize(article, words=self.n_words) for article in test_dataset['articles']]


class BartTextRank(AbstractModel):
    def __init__(self, model_checkpoint="facebook/bart-base", n_words=64):
        super().__init__()

        self.bart = BartModel(model_checkpoint)
        self.text_rank = TextRank(n_words)

    def train(self, train_dataset, val_dataset=None):
        self.bart.train(train_dataset, val_dataset)

    def predict(self, test_dataset):
        test = test_dataset.map(lambda dataset: self.bart.__preprocess_function__(dataset, self.bart.tokenizer,
                                                                                  self.bart.MAX_SOURCE_LENGTH,
                                                                                  self.bart.MAX_TARGET_LENGTH),
                                batched=True)

        bart_predictions = [self.bart.tokenizer.decode(prediction) for prediction in
                            self.bart.predict(test).predictions]
        text_rank_predictions = self.text_rank.predict(test_dataset)

        test['articles'] = [bart_predictions[i] + " " + text_rank_predictions[i] for i in range(len(test_dataset))]

        return self.bart.predict(test)
