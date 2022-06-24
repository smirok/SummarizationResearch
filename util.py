import numpy as np
from datasets import load_metric
from nltk import sent_tokenize

import nltk

from model.util.barycenter import BarycenterModel

nltk.download("punkt")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(preds, labels, val_dataset):
    rouge_score = load_metric("rouge")
    preds, labels = postprocess_text(preds, labels)

    rouge_result = rouge_score.compute(
        predictions=preds,
        references=labels,
        use_stemmer=True
    )

    rouge_result = {key: round(value.mid.fmeasure * 100, 4) for key, value in rouge_result.items()}

    barycenter_result = np.array(BarycenterModel.calculate_texts_barycenter(preds))
    barycenter_labels = np.array(val_dataset['barycenters'])
    barycenter_result = {'barycenters': np.mean(np.linalg.norm(barycenter_result - barycenter_labels, axis=1))}

    return rouge_result.update(barycenter_result)


def delete_last_sep(example):
    example['document'] = example['document'][:-5]
    return example


def add_barycenter(example, doc_sep):
    articles = example['articles'].split(doc_sep)
    example['barycenters'] = BarycenterModel.calculate_texts_barycenter(articles)
    return example


def preprocess(example):
    example['articles'] = ' '.join(example['articles'].replace('\n', ' ').split())
    return example


def process_separator(example, doc_sep):
    example['articles'] = example['articles'].replace(doc_sep, ' ')
    return example
