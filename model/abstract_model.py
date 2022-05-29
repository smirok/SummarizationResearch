from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(
            self,
            max_target_length=64,
            max_source_length=1024,
            save_path="./"
    ):
        self.save_path = save_path
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length

    @abstractmethod
    def train(self, train_dataset, val_dataset=None):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, test_dataset):
        raise NotImplementedError()
