from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from dotmap import DotMap
from transformers import BertConfig, AutoTokenizer, BertModel
from knowledge_probing.models.lightning.decoder import Decoder
from knowledge_probing.models.lightning.hugging_decoder import HuggingDecoder
from knowledge_probing.training.training import training
from knowledge_probing.models.models_helper import get_model
from knowledge_probing.probing.probing import probing
from knowledge_probing.config.config_helper import handle_config
import sys


def main(args):

    seed_everything(args.seed)

    print('Learning rate {}'.format(args.lr))

    args = handle_config(args)
    decoder = get_model(args)

    if args.do_training:
        training(args, decoder)

    if args.do_probing:
        probing(args, decoder)

    return args.run_name


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser = Decoder.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)

    # Decoder
    parser.add_argument('--decoder_type', required='--do_training' in sys.argv,
                        choices=['Huggingface_pretrained_decoder', 'Decoder'],
                        help='Use either the huggingface_pretrained_decoder, which was used during pre-training of BERT, or a randomly initialized decoder')

    # Training
    parser.add_argument('--do_training', default=False, action='store_true')
    parser.add_argument('--training_early_stop_delta', default=0.01, type=int,
                        help='The minimum validation-loss-delta between #patience iterations that has to happen for the computation not to stop')
    parser.add_argument('--training_early_stop_patience', default=15, type=int,
                        help='The patience for the models validation loss to improve by [training_early_stop_delta] for the computation not to stop')

    # Probing
    parser.add_argument('--do_probing', default=False, action='store_true')
    parser.add_argument('--probing_layer', default=12,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], type=int)

    # Other
    parser.add_argument('--run_name', default='',
                        help='Name of the run that will be used when building the run_identifier')
    parser.add_argument('--output_base_dir', default='data/outputs/',
                        help='Path to the output dir that will contain the logs and trained models')
    parser.add_argument('--seed', default=42, type=int)

    # Wandb
    parser.add_argument('--use_wandb_logging', default=False, action='store_true',
                        help='Use this flag to use wandb logging. Otherwise we will use the pytorch-lightning tensorboard logger')
    parser.add_argument('--wandb_project_name', required='--use_wandb_logging' in sys.argv, type=str,
                        help='Name of wandb project')
    parser.add_argument('--wandb_run_name', default='',
                        type=str, help='Name of wandb run')
    parser.add_argument('--python_executable', required='--use_wandb_logging' in sys.argv, type=str, default='/usr/bin/python3',
                        help='Some cluster environments might require to set the sys.executable for wandb to work')

    args = parser.parse_args()

    main(args)
