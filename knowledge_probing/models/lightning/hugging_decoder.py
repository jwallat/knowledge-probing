from knowledge_probing.models.lightning.decoder import Decoder
from knowledge_probing.datasets.text_data_utils import collate
from transformers import BertForMaskedLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
import functools


class HuggingDecoder(Decoder):
    def __init__(self, hparams, bert, config, decoder=None):
        super(Decoder, self).__init__()
        print('Hparams in init: {}'.format(hparams))
        print('Using the huggingface pre-trained decoder...')
        self.hparams = hparams
        if decoder == None:
            self.decoder = BertForMaskedLM.from_pretrained(
                self.hparams.bert_model_type, config=config).cls
        else:
            self.decoder = decoder
        self.decoder.train()

        self.bert = bert
        self.bert.eval()
        self.bert.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.bert_model_type, use_fast=False)
        self.config = config

        self.total_num_training_steps = 0

        self.collate = functools.partial(collate, tokenizer=self.tokenizer)

    def forward(self, inputs, masked_lm_labels, attention_mask=None, lm_label=None, layer=None, all_layers=False):
        # get attention mask
        if attention_mask == None:
            attention_mask = inputs.clone()
            attention_mask[attention_mask != self.tokenizer.pad_token_id] = 1
            attention_mask[attention_mask == self.tokenizer.pad_token_id] = 0

        # get Berts embeddings
        bert_outputs = self.bert(inputs, attention_mask=attention_mask)
        if not all_layers:
            if layer == 12:
                embeddings = bert_outputs[0]
            else:
                embeddings = bert_outputs[2][layer]

            # Feed embeddings into decoder
            prediction_scores = self.decoder(embeddings)
            outputs = (prediction_scores,)

            if masked_lm_labels is not None:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                outputs = (masked_lm_loss,) + outputs

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            prediction_scores = outputs[1]

            return loss, prediction_scores

        if all_layers:
            # Run all embeddings through the decoder and bundle their predictions
            layer_outputs = []

            for i, layer_embeddings in enumerate(bert_outputs[2]):
                if i == 0:
                    # Step over input_embeddings
                    continue
                   # Feed embeddings into decoder
                prediction_scores = self.decoder(layer_embeddings)

                layer_outputs.append(prediction_scores)

            return layer_outputs
