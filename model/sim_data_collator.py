from typing import Any, Tuple, Optional, Union, Dict, List

import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from transformers.data.data_collator import _torch_collate_batch

from rank_text import mask_rank_texts


def postprocess_labels(masked_text, tokenizer, estimated_size):
    return torch.LongTensor(
        list(map(lambda list_: (list(filter(lambda token_id: token_id != 1437, list_)) + [2] * 20)[0:estimated_size],
                 tokenizer(masked_text, add_special_tokens=True)['input_ids'])))


class SimDataCollator(DataCollatorForLanguageModeling):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        estimated_size = len(labels[0])
        text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        masked_text = mask_rank_texts(text, self.tokenizer)
        labels = postprocess_labels(masked_text, self.tokenizer, estimated_size)

        return inputs, labels
