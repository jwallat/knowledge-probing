from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import pickle
import json
import torch
from knowledge_probing.datasets.cloze_data_utils import lowercase_samples, filter_samples, get_index_for_mask, parse_template
from knowledge_probing.file_utils import load_file

# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)


class ClozeDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, vocab, block_size=512, output_debug_info=False):
        if not os.path.isfile(args.relation_args.dataset_filename):
            print("Could not create features from dataset %s. File not found",
                  args.relation_args.dataset_filename)
            return
        assert os.path.isfile(args.relation_args.dataset_filename)
        print("Creating features from dataset file at %s",
              args.relation_args.dataset_filename) if output_debug_info else None

        self.samples = []

        samples = load_file(args.relation_args.dataset_filename)
        print('number samples: {}'.format(len(samples))
              ) if output_debug_info else None

        # Lowercase if needed
        if args.lowercase:
            print("lowercasing all samples...") if output_debug_info else None
            samples = lowercase_samples(samples, tokenizer.mask_token)

        # Filter samples
        print('filtering the samples') if output_debug_info else None
        samples, _ = filter_samples(
            samples, tokenizer, vocab, args.relation_args.template)
        print('number filtered samples: {}'.format(
            len(samples))) if output_debug_info else None

        # Make sure every sub/obj pair is only used once
        if args.relation_args.template and args.relation_args.template != "":
            facts = []
            for sample in samples:
                sub = sample["sub_label"]
                obj = sample["obj_label"]
                if 'judgments' in sample and ((sub, obj) not in facts):
                    facts.append((sub, obj, sample['judgments']))
                elif (sub, obj) not in facts:
                    facts.append((sub, obj))
            print("distinct template facts: {}".format(
                len(facts))) if output_debug_info else None
            all_samples = []
            for fact in facts:
                sample = {}
                if len(fact) == 2:
                    (sub, obj) = fact
                elif len(fact) == 3:
                    (sub, obj, judgments) = fact
                    sample['judgments'] = judgments
                sample["sub_label"] = sub
                sample["obj_label"] = obj

                # substitute all sentences with a standard template
                sample["masked_sentences"] = parse_template(
                    args.relation_args.template.strip(
                    ), sample["sub_label"].strip(), tokenizer.mask_token
                )
                all_samples.append(sample)
            samples = all_samples

        # Give every sample a uuid
        i = 0
        for sample in samples:
            if "uuid" not in sample:
                sample["uuid"] = i
            i += 1

        # Encode sentences and object label
        encoded_samples = []
        for sample in samples:
            encoded_sample = {}
            encoded_sample['masked_sentences'] = tokenizer.encode_plus(sample['masked_sentences'][0], add_special_tokens=True, return_tensors='pt')[
                'input_ids'][0]
            encoded_sample['obj_label'] = sample['obj_label']
            encoded_sample['mask_index'] = get_index_for_mask(
                encoded_sample['masked_sentences'], tokenizer.mask_token_id)
            encoded_sample['uuid'] = sample["uuid"]
            if 'judgments' in sample:
                encoded_sample['judgments'] = sample['judgments']
            encoded_samples.append(encoded_sample)

        self.samples = encoded_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]
