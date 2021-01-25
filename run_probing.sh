#!/bin/bash
mkdir test_dir
python run_probing.py \
        --decoder_type=Huggingface_pretrained_decoder \
        --do_training --max_epochs=100 \
        --train_file data/training_data/wikitext-2-raw/wiki.train.raw \
        --valid_file data/training_data/wikitext-2-raw/wiki.valid.raw \
        --test_file data/training_data/wikitext-2-raw/wiki.test.raw \
        --do_probing \
        --probing_layer 12 \
        --probing_data_dir data/probing_data/ \
        --gpus 1 \
        --output_base_dir=data/outputs/probe_bert/ \
        --run_name bert_layer_12 
        --use_wandb_logging \
        --wandb_project_name=probe_bert \
        --wandb_run_name bert_layer_12 \