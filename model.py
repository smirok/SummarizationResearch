import copy

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import LEDForConditionalGeneration
from datasets import load_metric
from summa.summarizer import summarize
from datasets import Dataset

import numpy as np
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize


def compute_metrics(preds, labels):
    rouge_score = load_metric("rouge")
    preds, labels = BartModel.postprocess_text(preds, labels)

    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=preds, references=labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


class AbstractModel:
    def __init__(self):
        pass

    def train(self, train_dataset, val_dataset=None):
        raise NotImplementedError()

    def predict(self, test_dataset):
        raise NotImplementedError()


class BartModel(AbstractModel):
    def __init__(self, max_target_length, max_source_length, model_checkpoint="facebook/bart-base"):
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}


class TextRank(AbstractModel):
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


class BartTextRank(AbstractModel):
    def __init__(self, max_target_length, max_source_length, model_checkpoint="facebook/bart-base"):
        super().__init__()

        self.bart = BartModel(max_target_length, max_source_length, model_checkpoint)
        self.text_rank = TextRank(max_target_length, max_source_length)

    def train(self, train_dataset, val_dataset=None):
        self.bart.train(train_dataset, val_dataset)

    def predict(self, test_dataset):
        test = test_dataset.map(lambda dataset: self.bart.__preprocess_function__(dataset, self.bart.tokenizer,
                                                                                  self.bart.max_source_length),
                                batched=True)

        bart_predictions = [self.bart.tokenizer.decode(prediction, skip_special_tokens=True) for prediction in
                            self.bart.predict(test).predictions]
        text_rank_predictions = self.text_rank.predict(test_dataset)

        test = Dataset.from_dict(
            {'articles': [bart_predictions[i] + "  " + text_rank_predictions[i] for i in range(len(test_dataset))],
             'summaries': test['summaries']})

        return self.bart.predict(test)


class Primer(AbstractModel):
    DOCSEP_TOKEN = "|||||"

    def __init__(self, max_target_length, max_source_length):
        super().__init__()
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

        self.tokenizer = AutoTokenizer.from_pretrained('allenai/PRIMERA')
        self.model = LEDForConditionalGeneration.from_pretrained('allenai/PRIMERA')
        self.model.gradient_checkpointing_enable()
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.DOCSEP_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
        self.test_dataset = None

    def train(self, train_dataset, val_dataset=None):
        pass

    def predict(self, test_dataset):
        self.test_dataset = copy.deepcopy(test_dataset)
        return test_dataset.map(self.batch_process, batched=True, batch_size=40)['summaries']

    def process_document(self, documents):
        input_ids_all = []
        for data in documents:

            if Primer.DOCSEP_TOKEN in data:
                all_docs = data.split(Primer.DOCSEP_TOKEN)  # [:-1]
            else:
                all_docs = [data]

            for i, doc in enumerate(all_docs):
                doc = doc.replace("\n", " ")
                doc = " ".join(doc.split())
                all_docs[i] = doc

            #### concat with global attention on doc-sep
            input_ids = []
            for doc in all_docs:
                input_ids.extend(
                    self.tokenizer.encode(
                        doc,
                        truncation=True,
                        max_length=self.max_source_length // len(all_docs),
                    )[1:-1]
                )
                input_ids.append(self.DOCSEP_TOKEN_ID)

            input_ids = (
                    [self.tokenizer.bos_token_id]
                    + input_ids
                    + [self.tokenizer.eos_token_id]
            )

            input_ids_all.append(torch.tensor(input_ids))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_all, batch_first=True, padding_value=self.PAD_TOKEN_ID
        )
        return input_ids

    def batch_process(self, batch):
        input_ids = self.process_document(batch['articles'])
        # get the input ids and attention masks together
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
        # put global attention on <s> token

        global_attention_mask[:, 0] = 1
        global_attention_mask[input_ids == self.DOCSEP_TOKEN_ID] = 1
        generated_ids = self.model.generate(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask,
            use_cache=True,
            max_length=self.max_target_length,
            num_beams=5,
        )
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )

        result = {'articles': generated_str, 'summaries': batch['summaries']}
        return result
