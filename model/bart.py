import gensim
import numpy as np

from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, IntervalStrategy

from model import AbstractModel, BarycenterModel
from util import postprocess_text


class BartModel(AbstractModel):
    def __init__(
            self,
            max_target_length=64,
            max_source_length=1024,
            save_path="./bart-model/",
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
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.rouge_score = load_metric("rouge")
        self.trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.args = Seq2SeqTrainingArguments(
            output_dir=self.save_path,
            evaluation_strategy=IntervalStrategy.EPOCH,
            learning_rate=5.6e-6,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            logging_strategy=IntervalStrategy.EPOCH,
            push_to_hub=False,
        )

    def train(self, train_dataset, val_dataset=None):
        self.train_dataset = train_dataset.map(lambda dataset: self.__preprocess_function__(dataset, self.tokenizer,
                                                                                            self.max_source_length),
                                               batched=True)
        if val_dataset is not None:
            self.val_dataset = val_dataset.map(lambda dataset: self.__preprocess_function__(dataset, self.tokenizer,
                                                                                            self.max_source_length),
                                               batched=True)
        else:
            self.val_dataset = None

        self.trainer = Seq2SeqTrainer(
            self.model,
            self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

    def predict(self, test_dataset):
        self.test_dataset = test_dataset.map(lambda dataset: self.__preprocess_function__(dataset, self.tokenizer,
                                                                                          self.max_source_length),
                                             batched=True)

        if self.trainer is None:
            self.trainer = Seq2SeqTrainer(
                self.model,
                self.args,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )

        return self.trainer.predict(test_dataset, max_length=self.max_target_length)

    @staticmethod
    def __preprocess_function__(dataset, tokenizer, max_source_length):
        model_inputs = tokenizer(
            dataset['articles'],
            max_length=max_source_length,
            padding='max_length',
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(dataset['summaries'])

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        rouge_result = self.rouge_score.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        rouge_result = {key: round(value.mid.fmeasure * 100, 4) for key, value in rouge_result.items()}

        barycenter_result = np.array(BarycenterModel.calculate_texts_barycenter(decoded_preds))
        barycenter_labels = np.array(self.val_dataset['barycenters'])
        barycenter_result = {'barycenters': np.mean(np.linalg.norm(barycenter_result - barycenter_labels, axis=1))}

        return rouge_result.update(barycenter_result)
