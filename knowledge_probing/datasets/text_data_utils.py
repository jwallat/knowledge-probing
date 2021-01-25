import torch
from typing import Tuple, List
from transformers import AutoTokenizer
import functools
from torch.nn.utils.rnn import pad_sequence


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def mask_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. 
        This is the standard script used in the huggingface libaray with slight adjustments for pytorch-lightning. 
        That is only adjusting how tensors are casted to the device (e.g. probability_matrix = probability_matrix.to(inputs.device)).
    """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    probability_matrix = probability_matrix.to(inputs.device)

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask_tensor = torch.tensor(
        special_tokens_mask, dtype=torch.bool)
    special_tokens_mask_tensor = special_tokens_mask_tensor.to(inputs.device)

    probability_matrix.masked_fill_(special_tokens_mask_tensor, value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices = masked_indices.to(inputs.device)

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    full_tensor = torch.full(labels.shape, 0.8)
    full_tensor = full_tensor.to(inputs.device)

    indices_replaced = torch.bernoulli(full_tensor).bool() & masked_indices
    indices_replaced = indices_replaced.to(inputs.device)

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
        tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    other_full_tensor = torch.full(labels.shape, 0.5)
    other_full_tensor = other_full_tensor.to(inputs.device)

    indices_random = torch.bernoulli(
        other_full_tensor).bool() & masked_indices & ~indices_replaced
    indices_random = indices_random.to(inputs.device)

    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long)
    random_words = random_words.to(inputs.device)

    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def collate(examples, tokenizer=None):
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
