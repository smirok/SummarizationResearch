from typing import List

import gensim.downloader as api
import gensim.models
import numpy as np
from nltk import word_tokenize


class BarycenterModel(object):
    model = api.load('word2vec-google-news-300')

    @staticmethod
    def calculate_text_barycenter(document: str):
        words = word_tokenize(document)

        cnt = 0
        barycenter = np.zeros((300,))
        for word in words:
            try:
                barycenter += BarycenterModel.model[word.lower()]
                cnt += 1
            except KeyError:
                pass

        return barycenter / cnt

    @staticmethod
    def calculate_texts_barycenter(document_list: List[str]):
        return [BarycenterModel.calculate_text_barycenter(document) for document in document_list]

    @staticmethod
    def calculate_vectors_barycenter(vectors: np.ndarray):
        return vectors.mean(axis=0)
