from torch import nn
from transformers import AutoModel


class TransferBartModule(nn.Module):

    def __init__(self, model_checkpoint, sentence_number=128, importance_threshold=0.7):
        super(TransferBartModule, self).__init__()
        self.bart = AutoModel.from_pretrained(model_checkpoint)
        self.sentence_number = sentence_number
        self.importance_threshold = importance_threshold

        self.fc1 = nn.Linear(768, self.sentence_number)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.bart(
            input_ids,
            attention_mask=attention_mask
        )

        logits = self.fc1(outputs.last_hidden_state[:, 0, :].view(-1, 768))  ## extract the 1st token's embeddings
        return logits
