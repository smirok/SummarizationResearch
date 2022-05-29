from datasets import load_metric
from nltk import sent_tokenize

import nltk

nltk.download("punkt")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, labels):
    rouge_score = load_metric("rouge")
    preds, labels = postprocess_text(preds, labels)

    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=preds, references=labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def delete_last_sep(example):
    example['document'] = example['document'][:-5]
    return example


def preprocess(example):
    example['articles'] = ' '.join(example['articles'].replace('\n', ' ').split())
    return example


def process_separator(example, doc_sep):
    example['articles'] = example['articles'].replace(doc_sep, ' ')
    return example
