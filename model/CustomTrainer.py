from torch.nn import BCEWithLogitsLoss
from transformers import Seq2SeqTrainer, PreTrainedModel
from model import TransferBartModule


class CustomTrainer(Seq2SeqTrainer, PreTrainedModel):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['labels']

        outputs = model(**inputs)
        logits = outputs["logits"]
        print(logits)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, TransferBartModule.SENTENCE_NUMBER),
                            labels.view(-1, TransferBartModule.SENTENCE_NUMBER))
        else:
            loss = logits

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset, ignore_keys=None, metric_key_prefix: str = 'eval', max_length=None, num_beams=None):
        super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, max_length, num_beams)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix: str = 'test', max_length=None, num_beams=None):
        result = super().predict(test_dataset, ignore_keys, metric_key_prefix, max_length, num_beams)
