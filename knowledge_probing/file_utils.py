from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from dotmap import DotMap
import yaml
import torch
import pickle
import json
import os


def stringify_dotmap(args):
    stringy = 'Args:\n'
    for k, v in args.items():
        if type(v) != DotMap:
            # print(k)
            if (str(k) == 'getdoc') or (str(k) == 'shape'):
                continue
            else:
                stringy = stringy + '\t' + k + ':' + '\t' + str(v) + '\n'
        elif type(v) == DotMap:
            stringy = stringy + '\t' + k + ':\n'
            for key, value in v.items():
                # print(key)
                if (str(k) == 'getdoc') or (str(k) == 'shape'):
                    continue
                else:
                    stringy = stringy + '\t\t' + key + \
                        ':' + '\t' + str(value) + '\n'
    return stringy

# Write args to args.execution_log


def write_to_execution_log(text, path, append_newlines=False):
    # print("Saving args into file %s", args.execution_log)
    with open(path, "a") as handle:
        handle.write(text)
        handle.write('\n\n') if append_newlines else None


def load_vocab(path):
    assert os.path.exists(path)

    with open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab


def get_vocab(bert_model_type):
    tokenizer = BertTokenizer.from_pretrained(bert_model_type)
    tokenizer.save_vocabulary(vocab_path='.')

    # Load vocabulary
    vocab = load_vocab('./vocab.txt')
    return vocab


def load_config(path):
    assert os.path.exists(path)

    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def load_file(filename):
    assert os.path.exists(filename)
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def load_model_config(filename):
    assert os.path.exists(filename)
    with open(filename, 'r') as f:
        data = f.read()

    return data


def write_metrics(run_identifier, dataset, relation, dir, metrics):
    metrics_file = os.path.join(
        dir, dataset + "_" + relation + "_" + run_identifier
    )
    # print("Saving metrics into file %s", metrics_file)
    with open(metrics_file, "w") as handle:
        handle.write('Results: {}'.format(metrics))


def find_checkpoint_in_dir(dir):
    for file in os.listdir(dir):
        if file.endswith(".ckpt"):
            checkpoint = os.path.join(dir, file)
            return checkpoint
