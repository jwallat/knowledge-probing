from knowledge_probing.datasets.cloze_data_utils import topk


def calculate_metrics(batch, index, prediction_scores, precision_at_k, tokenizer=None, total_top_k_words=1000):
    metrics_element = {}

    # print(batch)

    # sample information (masked sentences, obj_label, uuid)
    metrics_element['sample'] = {
        'masked_sentences': tokenizer.convert_ids_to_tokens(batch['masked_sentences'][index]),
        'obj_label': batch['obj_label'][index],
        'uuid': batch['uuid'][index]
    }

    # Initialize values
    metrics_element['P_AT_K'] = 0.0
    metrics_element['P_AT_10'] = 0.0
    metrics_element['P_AT_1'] = 0.0
    metrics_element['MRR'] = 0.0
    metrics_element['PERPLEXITY'] = None

    # get topk predictions
    topk_tokens = topk(prediction_scores,
                       batch['mask_index'][index], k=total_top_k_words, tokenizer=tokenizer)
    # print(topk_tokens)
    metrics_element['top_k_tokens'] = topk_tokens[:precision_at_k]

    try:
        # get rank of our expected word
        rank = topk_tokens.index(batch['obj_label'][index])
        # print(rank)
        metrics_element['rank'] = rank

        # MRR
        if rank > 0:
            metrics_element['MRR'] = (1 / rank)

        #precision @ 1, 10, k
        if rank <= precision_at_k:
            metrics_element['P_AT_K'] = 1
        if rank <= 10:
            metrics_element['P_AT_10'] = 1
        if rank == 0:
            metrics_element['P_AT_1'] = 1

        # perplexity
        # perplexity = 0

        # metrics_element['PERPLEXITY'] = perplexity

    except:
        metrics_element['rank'] = 'not found in top {} words'.format(
            total_top_k_words)

    # judgement
    if 'judgments' in batch:
        num_yes = 0
        num_no = 0
        for judgment_ele in batch['judgments']:
            for judgment in judgment_ele:
                if judgment['judgment'] == 'yes':
                    num_yes += 1
                else:
                    num_no += 1

        if num_yes > num_no:
            metrics_element['sample']['judgment'] = 'positive'
        elif num_no <= num_yes:
            metrics_element['sample']['judgment'] = 'negative'

    # print(metrics_element)

    return metrics_element


def aggregate_metrics_elements(metrics_elements):

    # Calc mean p1,p10, pk, MRR

    # Mean reciprocal rank
    MRR = sum([x['MRR'] for x in metrics_elements]) / \
        len([x['MRR'] for x in metrics_elements])
    MRR_negative = 0.0
    MRR_positive = 0.0

    # Precision at (default 10)
    Precision10 = sum([x['P_AT_10'] for x in metrics_elements]) / \
        len([x['P_AT_10'] for x in metrics_elements])
    Precision1 = sum([x['P_AT_1'] for x in metrics_elements]) / \
        len([x['P_AT_1'] for x in metrics_elements])
    PrecisionK = sum([x['P_AT_K'] for x in metrics_elements]) / \
        len([x['P_AT_K'] for x in metrics_elements])
    Precision_negative = 0.0
    Precision_positive = 0.0

    total_positive = 0
    total_negative = 0

    # the judgment of the annotators recording whether they are
    # evidence in the sentence that indicates a relation between two entities.
    for element in metrics_elements:
        if 'judgment' in element['sample']:
            if element['sample']['judgment'] == 'negative':
                total_negative += 1
                MRR_negative += element['MRR']
                Precision_negative += element['P_AT_K']
            else:
                total_positive += 1
                MRR_positive += element['MRR']
                Precision_positive += element['P_AT_K']

    if total_negative > 0:
        Precision_negative = Precision_negative / total_negative
        MRR_negative = MRR_negative / total_negative

    if total_positive > 0:
        Precision_positive = Precision_positive / total_positive
        MRR_positive = MRR_positive / total_positive

    aggregated = {
        'MRR': MRR,
        'MRR_negative': MRR_negative,
        'MRR_positive': MRR_positive,
        'P_AT_1': Precision1,
        'P_AT_10': Precision10,
        'P_AT_K': PrecisionK,
        'P_AT_K_positive': Precision_positive,
        'P_AT_K_negative': Precision_negative,
        'individual_predictions': metrics_elements
    }

    return aggregated


# Calc means for google-re and trex
def mean_precisions(data):
    p1s = []
    p10s = []
    pks = []

    for relation_metric in data:
        p1s.append(relation_metric['P_AT_1'])
        p10s.append(relation_metric['P_AT_10'])
        pks.append(relation_metric['P_AT_K'])

    mean_p1 = sum(p1s) / len(p1s)
    mean_p10 = sum(p10s) / len(p10s)
    mean_pk = sum(pks) / len(pks)

    return 'Mean P@1,10,k: {}, {}, {}'.format(mean_p1, mean_p10, mean_pk)
