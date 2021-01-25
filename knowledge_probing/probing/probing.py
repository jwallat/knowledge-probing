from knowledge_probing.datasets.cloze_data_utils import collate
from knowledge_probing.datasets.cloze_dataset import ClozeDataset
from knowledge_probing.probing.metrics import calculate_metrics, mean_precisions, aggregate_metrics_elements
from knowledge_probing.probing.probing_args import build_args
from knowledge_probing.plotting.plots import handle_mean_values_string
from knowledge_probing.file_utils import write_metrics, write_to_execution_log, stringify_dotmap, get_vocab
from transformers import BertTokenizer, BertForMaskedLM
from dotmap import DotMap
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import gc
import os
import json
import functools
import wandb


def probing(args, decoder):
    # Use the correct probing model
    probing_model, tokenizer = get_probing_model(args, decoder)

    # TODO: Compute the shared vocab between models with different vocabs (e.g. for comparison of BERT and ELECTRA)
    vocab = get_vocab(args.bert_model_type)

    # Choose datasets to probe
    # The dataset loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)
    dataset_args = []
    dataset_args.append(('Google_RE', build_args(
        'Google_RE', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    # dataset_args.append(('Google_RE_UHN', build_args(
    #     'Google_RE_UHN', args.lowercase, args.probing_data_dir, args.precision_at_k, bert_model_type=args.bert_model_type)))
    dataset_args.append(('TREx', build_args(
        'TREx', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    # dataset_args.append(('TREx_UHN', build_args(
    #     'TREx_UHN', args.lowercase, args.probing_data_dir, args.precision_at_k, bert_model_type=args.bert_model_type)))
    dataset_args.append(('ConceptNet', build_args(
        'ConceptNet', args.lowercase, args.probing_data_dir, args.precision_at_k)))
    dataset_args.append(('Squad', build_args(
        'Squad', args.lowercase, args.probing_data_dir, args.precision_at_k)))

    # Do the probing
    data = probe(args, probing_model, tokenizer,
                 dataset_args, layer=args.probing_layer, vocab=vocab)

    save_probing_data(args, data)

    return


def get_probing_model(args, decoder):
    tokenizer = decoder.tokenizer
    probing_model = decoder

    probing_model.to(args.device)
    probing_model.zero_grad()

    return probing_model, tokenizer


def probe(args, probing_model, tokenizer, dataset_args, layer, vocab):
    write_to_execution_log(
        100 * '+' + '\t Probing layer: {} \t'.format(layer) + 100 * '+', append_newlines=True, path=args.execution_log)
    print('##################################      Layer: {}      #########################################'.format(layer))

    layer_data = {}

    google_re_metrices = []
    trex_metrices = []

    my_collate = functools.partial(collate, tokenizer=tokenizer)

    print('$$$$$$$$$$$$$$$$$$$$$$$    Probing model of type: {}      $$$$$$$$$$$$$$$$$$$$$$$$'.format(
        args.bert_model_type))
    for ele in dataset_args:
        ds_name, relation_args_list = ele

        layer_data[ds_name] = {}
        layer_data[ds_name]['means'] = []

        print('*****************   {}   **********************'.format(ds_name))

        for relation_args in relation_args_list:
            relation_args = DotMap(relation_args)
            args.relation_args = relation_args
            print(
                '---------------- {} ----------------------'.format(args.relation_args.template))
            print(stringify_dotmap(args.relation_args))

            layer_data[ds_name][args.relation_args.relation] = []

            dataset = ClozeDataset(
                tokenizer, args, vocab, tokenizer.max_len, output_debug_info=False)

            # Create dataloader
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(
                dataset, sampler=sampler, batch_size=args.probing_batch_size, collate_fn=my_collate)

            metrics_elements = []
            input_ids_batch = None
            outputs = None
            batch_prediction_scores = None
            prediction_scores = None

            for _, batch in enumerate(tqdm(dataloader)):
                input_ids_batch = batch['masked_sentences']
                attention_mask_batch = batch['attention_mask']

                input_ids_batch = input_ids_batch.to(args.device)
                attention_mask_batch = attention_mask_batch.to(args.device)

                # Get predictions from models
                outputs = probing_model(
                    input_ids_batch, masked_lm_labels=input_ids_batch, attention_mask=attention_mask_batch, layer=layer)

                batch_prediction_scores = outputs[1]

                for i, prediction_scores in enumerate(batch_prediction_scores):
                    prediction_scores = prediction_scores[None, :, :]
                    metrics_element = calculate_metrics(
                        batch, i, prediction_scores, precision_at_k=args.relation_args.precision_at_k, tokenizer=tokenizer)

                    metrics_elements.append(metrics_element)

                # Reset vars and clear memory - avoid crashes
                del input_ids_batch
                del outputs
                del batch_prediction_scores
                del prediction_scores
                gc.collect()

            print('Number metrics elements: {}'.format(len(metrics_elements)))
            aggregated_metrics = aggregate_metrics_elements(metrics_elements)
            print('Aggregated: {}'.format(aggregated_metrics['P_AT_1']))

            if ds_name == 'Google_RE':
                google_re_metrices.append(aggregated_metrics)
            elif ds_name == 'TREx':
                trex_metrices.append(aggregated_metrics)

            layer_data[ds_name][args.relation_args.relation].append(
                aggregated_metrics)
            write_to_execution_log(ds_name + ': ' + args.relation_args.relation +
                                   '\t' + str(aggregated_metrics['P_AT_1']), append_newlines=True, path=args.execution_log)

    # Write results to logfile
    if len(google_re_metrices) > 0:
        write_to_execution_log(
            '\n\nGoogle_RE: ' + mean_precisions(google_re_metrices), append_newlines=True, path=args.execution_log)
        layer_data['Google_RE']['means'].append(
            mean_precisions(google_re_metrices))
    if len(trex_metrices) > 0:
        write_to_execution_log(
            'Trex: ' + mean_precisions(trex_metrices), append_newlines=True, path=args.execution_log)
        layer_data['TREx']['means'].append(mean_precisions(trex_metrices))
    write_to_execution_log(
        220 * '-', append_newlines=True, path=args.execution_log)

    if args.use_wandb_logging:
        wandb.init(name=args.wandb_run_name, project=args.wandb_project_name)
        wandb_log_metrics(layer_data, layer)

    return layer_data


def save_probing_data(args, data):
    with open('{}/{}_data.json'.format(args.output_dir, 'model'), 'w') as outfile:
        json.dump(data, outfile)


def wandb_log_metrics(layer_data, layer):

    # log datasets with just one relation
    wandb.log(
        {'layer': layer, 'ConceptNet P@1': layer_data['ConceptNet']['test'][0]['P_AT_1']})
    wandb.log(
        {'layer': layer, 'Squad P@1': layer_data['Squad']['test'][0]['P_AT_1']})

    # log dataset with messy means
    # Google_RE
    p1 = handle_mean_values_string(layer_data['Google_RE']['means'][0])[0]
    wandb.log({'layer': layer, 'Google_RE P@1': p1})

    # TREx
    p1 = handle_mean_values_string(layer_data['TREx']['means'][0])[0]
    wandb.log({'layer': layer, 'TREx P@1': p1})
