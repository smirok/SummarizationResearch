from torch.nn import MSELoss
from transformers import Trainer


class TransferTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss = MSELoss()
        loss = loss(logits.view(-1, len(logits)),
                    labels.view(-1, len(labels)))

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = 'eval', max_length=None,
                 num_beams=None):
        pass
