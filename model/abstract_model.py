from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, train_dataset, val_dataset=None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, test_dataset):
        raise NotImplementedError()
