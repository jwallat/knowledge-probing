from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer
import os
import pickle
from tqdm import tqdm
import torch
from knowledge_probing.datasets.text_data_utils import chunks


class TextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, args, file_path: str, block_size=512):
        print(file_path)
        assert os.path.isfile(file_path)

        block_size = block_size - \
            (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.bert_model_type +
            "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from dataset file at %s", directory)

            # Get the faster tokenizer from tokenizers package
            tokenizer.save_vocabulary(vocab_path='.')
            fast_tokenizer = BertWordPieceTokenizer(
                "vocab.txt", lowercase=args.lowercase)
            fast_tokenizer.enable_truncation(tokenizer.max_len)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            text_chunks = list(chunks(text, 300000))

            for chunk in tqdm(text_chunks):
                batch = fast_tokenizer.encode(chunk)
                self.examples.append(batch.ids)

                for encoding in batch.overflowing:
                    # if len(encoding.ids) == tokenizer.max_len:
                    self.examples.append(encoding.ids)

            # tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            # for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            #     self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
