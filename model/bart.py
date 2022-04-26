import numpy as np

from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

from model import AbstractModel
from util import postprocess_text


class BartModel(AbstractModel):
    def __init__(self, max_target_length, max_source_length, model_checkpoint="facebook/bart-base",
                 tokenizer_checkpoint=None):
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

        self.batch_size = 32
        self.num_train_epochs = 3
        self.rouge_score = load_metric("rouge")
        self.trainer = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint if (tokenizer_checkpoint is None) else tokenizer_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        self.args = Seq2SeqTrainingArguments(
            output_dir=f"multidoc-bart",
            evaluation_strategy="epoch",
            learning_rate=5.6e-6,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=self.num_train_epochs,
            predict_with_generate=True,
            logging_strategy="epoch",
            # logging_steps=logging_steps,
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

        logging_steps = len(self.train_dataset) // self.batch_size

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

        return self.trainer.predict(self.test_dataset, max_length=self.max_target_length)

    @staticmethod
    def __preprocess_function__(dataset, tokenizer, max_source_length):
        model_inputs = tokenizer(
            dataset['articles'],
            max_length=max_source_length,
            padding='max_length',
            truncation=True
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(dataset['summaries'])

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

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

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # Compute ROUGE scores
        result = self.rouge_score.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
