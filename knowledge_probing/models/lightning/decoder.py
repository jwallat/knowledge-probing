from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import functools
from typing import Tuple, List
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from knowledge_probing.models.decoder_head import MyDecoderHead
from knowledge_probing.datasets.text_dataset import TextDataset
from knowledge_probing.datasets.text_data_utils import mask_tokens, collate
from argparse import ArgumentParser
import sys


class Decoder(LightningModule):
    def __init__(self, hparams, bert, config):
        super(Decoder, self).__init__()
        self.decoder = MyDecoderHead(config)

        self.hparams = hparams
        self.bert = bert
        self.config = config

        self.bert.eval()
        self.bert.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.bert_model_type, use_fast=False)

        print('Model is using LR {}'.format(self.hparams.lr))

        self.total_num_training_steps = 0

        self.collate = functools.partial(collate, tokenizer=self.tokenizer)

    def forward(self, inputs, masked_lm_labels, attention_mask=None, layer=None, all_layers=False):

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
            decoder_outputs = self.decoder(
                embeddings, masked_lm_labels=masked_lm_labels)
            loss = decoder_outputs[0]
            prediction_scores = decoder_outputs[1]

            return loss, prediction_scores

        if all_layers:
            # Run all embeddings through the decoder and bundle their predictions
            layer_outputs = []

            for i, layer_embeddings in enumerate(bert_outputs[2]):
                if i == 0:
                    # Step over input_embeddings
                    continue
                   # Feed embeddings into decoder
                decoder_outputs = self.decoder(
                    layer_embeddings, masked_lm_labels=masked_lm_labels)
                prediction_scores = decoder_outputs[1]

                layer_outputs.append(prediction_scores)

            return layer_outputs

    def prepare_data(self):
        self.train_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.train_file, block_size=self.tokenizer.max_len)
        self.eval_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.valid_file, block_size=self.tokenizer.max_len)
        self.test_dataset = TextDataset(
            self.tokenizer, self.hparams, file_path=self.hparams.test_file, block_size=self.tokenizer.max_len)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate, pin_memory=False)

        self.total_num_training_steps = self.hparams.max_epochs * \
            len(train_dataloader)
        # print('Total number steps: ', self.total_num_training_steps)
        # self.configure_optimizers()

        return train_dataloader

    def val_dataloader(self):
        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate, pin_memory=False)

        return eval_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, collate_fn=self.collate, pin_memory=False)

        return test_dataloader

    def configure_optimizers(self):
        adam = AdamW([p for p in self.decoder.parameters(
        ) if p.requires_grad], lr=self.hparams.lr, eps=1e-08)

        print('Configuring optimizer, total number steps: ',
              self.total_num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            adam, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)

        # scheduler = ReduceLROnPlateau(adam, mode='min')

        return [adam], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]

        tensorboard_logs = {'training_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)

        tensorboard_logs = {
            'avg_test_loss': avg_loss, 'perplexity': perplexity}
        return {"avg_test_loss": avg_loss, 'perplexity': perplexity, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, labels = mask_tokens(batch, self.tokenizer, self.hparams)

        loss = self.forward(inputs, masked_lm_labels=labels,
                            layer=self.hparams.probing_layer)[0]
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        perplexity = torch.exp(avg_loss)
        print(perplexity)

        tensorboard_logs = {'avg_val_loss': avg_loss, 'perplexity': perplexity}
        return {"val_loss": avg_loss, 'perplexity': perplexity, "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', default=0.02, type=float)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--warmup_steps', default=5000, type=int,
                            help='Number of steps that is used for the linear warumup')
        parser.add_argument('--total_steps', default=60000, type=int,
                            help='Number of steps that is used for the decay after the warumup')

        parser.add_argument('--mlm_probability', default=0.15, type=float)

        parser.add_argument('--use_model_from_dir',
                            default=False, action='store_true')
        parser.add_argument(
            '--model_dir', required='--use_model_from_dir' in sys.argv)
        parser.add_argument('--bert_model_type', default='bert-base-uncased',
                            choices=['bert-base-uncased', 'bert-base-cased'],)

        parser.add_argument(
            '--train_file', default="data/training_data/wikitext-2-raw/wiki.train.raw")
        parser.add_argument(
            '--valid_file', default="data/training_data/wikitext-2-raw/wiki.valid.raw")
        parser.add_argument(
            '--test_file', default="data/training_data/wikitext-2-raw/wiki.test.raw")

        parser.add_argument('--probing_data_dir', default="data/probing_data/")
        parser.add_argument('--probing_batch_size', default=16, type=int)
        parser.add_argument('--precision_at_k', default=100, type=int,
                            help='When probing, we compute precision at 1, 10, and k. Feel free to set the k here')

        return parser
