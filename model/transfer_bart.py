import numpy as np

from datasets import load_metric
from transformers import BartForSequenceClassification, TrainingArguments, BartTokenizer, DataCollatorWithPadding, \
    IntervalStrategy, Seq2SeqTrainer, Seq2SeqTrainingArguments, BartForConditionalGeneration, \
    DataCollatorForSeq2Seq

from rank_text import rank_text
from model import AbstractModel, TransferTrainer, BarycenterModel
from util import postprocess_text


class TransferBartModel(AbstractModel):
    def __init__(
            self,
            max_target_length=64,
            max_source_length=1024,
            save_path="./transfer-bart-model/",
            model_checkpoint="facebook/bart-base",
            tokenizer_checkpoint="facebook/bart-base",
            epochs=3,
            batch_size=16,
            max_target_sentences=10
    ):
        super().__init__(
            max_target_length=max_target_length,
            max_source_length=max_source_length,
            save_path=save_path
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_target_sentences = max_target_sentences

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.rouge_score = load_metric("rouge")
        self.trainer = None
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_checkpoint)
        self.model = BartForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=self.max_target_sentences,
            output_attentions=False,
            output_hidden_states=False
        )
        self.sum_model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

        self.args = TrainingArguments(
            output_dir=f"multidoc-bart",
            evaluation_strategy=IntervalStrategy.EPOCH,
            learning_rate=5.6e-6,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=self.epochs,
            # predict_with_generate=True,
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

        self.trainer = TransferTrainer(
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
        self.test_dataset = test_dataset.map(
            lambda dataset: self.__preprocess_normal_function__(dataset, self.tokenizer,
                                                                self.max_source_length),
            batched=True)

        self.predict_args = Seq2SeqTrainingArguments(
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
        self.sum_model.model = self.model.model
        self.data_collator_predict = DataCollatorForSeq2Seq(self.tokenizer, model=self.sum_model)

        self.trainer_for_predict = Seq2SeqTrainer(
            self.sum_model,
            self.predict_args,
            data_collator=self.data_collator_predict,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        return self.trainer_for_predict.predict(self.test_dataset, max_length=self.max_target_length)

    def __preprocess_function__(self, dataset, tokenizer, max_source_length):
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

        dataset_size = len(labels["input_ids"])

        class_scores = list(map(lambda text: rank_text(text=text, ratio=1.0),
                                tokenizer.batch_decode(labels["input_ids"], skip_special_tokens=True)))

        results = np.zeros((dataset_size, self.max_target_sentences))
        for j in range(dataset_size):
            for i in range(min(self.max_target_sentences, len(class_scores[j]))):
                results[j][i] = class_scores[j][i]

        results = results / np.max(results)

        model_inputs["labels"] = results
        return model_inputs

    @staticmethod
    def __preprocess_normal_function__(dataset, tokenizer, max_source_length):
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
