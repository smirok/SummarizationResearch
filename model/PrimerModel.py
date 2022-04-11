import copy

import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration

from AbstractModel import AbstractModel


class PrimerModel(AbstractModel):
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

            if PrimerModel.DOCSEP_TOKEN in data:
                all_docs = data.split(PrimerModel.DOCSEP_TOKEN)  # [:-1]
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
