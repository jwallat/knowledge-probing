from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from knowledge_probing.file_utils import write_to_execution_log
from knowledge_probing.models.models_helper import load_best_model_checkpoint
import wandb
import sys


def training(args, decoder):
    checkpoint_callback = ModelCheckpoint(
        filepath=args.decoder_save_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=args.training_early_stop_delta,
        patience=args.training_early_stop_patience,
        verbose=True,
        mode='min'
    )

    if args.use_wandb_logging:
        print('If you are having issues with wandb, make sure to give the correct python executable to --python_executable')
        sys.executable = args.python_executable
        logger = WandbLogger(project=args.wandb_project_name,
                             name=args.wandb_run_name)
    else:
        logger = TensorBoardLogger("{}/tb_logs".format(args.output_dir))

    trainer = Trainer.from_argparse_args(
        args, checkpoint_callback=checkpoint_callback, early_stop_callback=early_stop_callback, logger=logger)

    write_to_execution_log('Run: {} \nArgs: {}\n'.format(
        args.run_identifier, args), path=args.execution_log)

    trainer.fit(decoder)

    decoder = load_best_model_checkpoint(decoder, args)

    trainer.test(decoder)
