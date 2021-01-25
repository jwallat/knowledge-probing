from transformers import BertConfig, BertModel, AutoTokenizer, BertForMaskedLM
from transformers.activations import gelu, gelu_new, swish
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Tuple, List
import torch


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu,
          "swish": swish, "gelu_new": gelu_new, "mish": mish}

BertLayerNorm = torch.nn.LayerNorm


class DecoderTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DecoderPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DecoderTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MyDecoderHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.predictions = DecoderPredictionHead(config)

        self.apply(self._init_weights)

    def forward(self, sequence_output,
                masked_lm_labels=None,
                lm_labels=None,):
        prediction_scores = self.predictions(sequence_output)

        outputs = (prediction_scores,)

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        # if lm_labels is not None:
        #     # we are doing next-token prediction; shift prediction scores and input ids by one
        #     prediction_scores = prediction_scores[:, :-1, :].contiguous()
        #     lm_labels = lm_labels[:, 1:].contiguous()
        #     loss_fct = CrossEntropyLoss()
        #     ltr_lm_loss = loss_fct(
        #         prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
        #     outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
