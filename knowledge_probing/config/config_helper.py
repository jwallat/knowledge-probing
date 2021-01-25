from knowledge_probing.file_utils import load_config
from datetime import datetime
from dotmap import DotMap
import torch
import os


def handle_config(args):
    # The idea is to handle all necessary implications (e.g. lowercase True/False from bert_model_type)
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    args.run_identifier = build_run_identifier(args)

    assert os.path.exists(args.probing_data_dir)

    args.output_dir = '{}{}'.format(args.output_base_dir, args.run_identifier)
    print(args.output_dir)
    os.makedirs(args.output_dir)

    args.decoder_save_dir = '{}/decoder/'.format(
        args.output_dir)
    os.makedirs(args.decoder_save_dir, exist_ok=True)

    args.execution_log = args.output_dir + '/execution_log.txt'

    if args.use_model_from_dir:
        assert os.path.exists(args.model_dir)

    args.lowercase = False
    if 'uncased' in args.bert_model_type:
        args.lowercase = True

    return args


def build_run_identifier(args):

    time = datetime.now()
    timestamp = '{}_{}_{}__{}-{}'.format(time.day,
                                         time.month, time.year, time.hour + 1, time.minute)

    model_type_postfix = args.bert_model_type.split('-')[-1]

    run_identifier = '{}_{}_trained-{}_{}_{}'.format(
        args.run_name, model_type_postfix, args.do_training, args.decoder_type, timestamp)

    print('Run identifier: ', run_identifier)
    return run_identifier
