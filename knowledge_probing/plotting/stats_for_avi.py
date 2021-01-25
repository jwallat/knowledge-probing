import os
import json
import gc
from tqdm import tqdm
from plotly import graph_objects as go

metrics = ['P_AT_1', 'P_AT_10', 'P_AT_K']
nice_metric_names = {
    'P_AT_1': 'P@1',
    'P_AT_10': 'P@10',
    'P_AT_K': 'P@100'
}
layer_range = range(1, 13)


output_dir = '/home/jonas/git/knowledge-probing/data/plots/stats_for_avi/'


def test_individual_prediction_order_constant(model_data):

    dataset = 'Google_RE'
    relation = 'place_of_birth'

    for layer in layer_range:
        print(model_data[str(layer)][dataset][relation]
              [0]['individual_predictions'][0])


def stats_for_avi(model_data, model_name):
    # model_data = model['data']

    dataset = 'Squad'
    relation = 'test'
    metric = 'P_AT_1'

    plot_data = []

    datasets = model_data['1'].keys()
    for dataset in datasets:
        print(60 * '*' + '    {}'.format(dataset))
        total_number_instance_over_all_relations = 0
        total_number_correct = 0
        total_number_correct_last_layer = 0

        for relation in model_data['1'][dataset].keys():
            if relation == 'means':
                continue
            print('Relation: {}'.format(relation))

            indidvidual_predictions_data = []

            # fill individual predictions data:
            indidvidual_predictions = model_data['1'][dataset][relation][0]['individual_predictions']
            for prediction in indidvidual_predictions:
                indidvidual_predictions_data.append({
                    'correct': False,
                    'correct_last_layer': False,
                    'sample': prediction['sample'],
                    'layer_correct': 0
                })

            for layer in layer_range:
                layer_data = model_data[str(layer)]

                indidvidual_predictions = layer_data[dataset][relation][0]['individual_predictions']

                for i, prediction in enumerate(indidvidual_predictions):
                    is_correct_prediction = got_prediction_right(
                        prediction, metric)

                    matching_prediction = find_matching_prediction(
                        prediction, indidvidual_predictions_data)

                    if layer == 12:
                        matching_prediction['correct_last_layer'] = is_correct_prediction

                    matching_prediction['correct'] = is_correct_prediction or matching_prediction['correct']
                    if is_correct_prediction:
                        matching_prediction['layer_correct'] = layer

            num_correct = 0
            num_correct_last_layer = 0
            for data in indidvidual_predictions_data:
                if data['correct']:
                    num_correct = num_correct + 1
                if data['correct_last_layer']:
                    num_correct_last_layer = num_correct_last_layer + 1

            print('{} {} had {} correct and in the last layer {}'.format(
                relation, metric, num_correct, num_correct_last_layer))

            ################################################################################################################
            # print('Some examples: ')
            # num = 0
            # for i, pred in enumerate(indidvidual_predictions_data):
            #     if pred['correct'] and not pred['correct_last_layer']:
            #         if num > 100:
            #             break
            #         # print(indidvidual_predictions_data[i])
            #         uuid = indidvidual_predictions_data[i]['sample']['uuid']
            #         layer_correct = indidvidual_predictions_data[i]['layer_correct']
            #         # print('Layer: {}'.format(layer_correct))
            #         rank = get_rank(model_data, dataset,
            #                         relation, layer_correct, uuid)
            #         # print('Layer: 12')
            #         last_layer_rank = get_rank(
            #             model_data, dataset, relation, 12, uuid)

            #         if type(last_layer_rank) == int:

            #             if 7 < int(last_layer_rank) < 30:
            #                 print('Layer {} predicts the correct token at rank {}. Layer that got it right: {}'.format(
            #                     layer, last_layer_rank, layer_correct))
            #                 print(pred['sample'])

            #                 num = num + 1

            ##################################################################################################################

            total_number_instance_over_all_relations = total_number_instance_over_all_relations + \
                len(indidvidual_predictions_data)
            total_number_correct = total_number_correct + num_correct
            total_number_correct_last_layer = total_number_correct_last_layer + num_correct_last_layer

        print('\nResults:')
        print('Total number of instances in this dataset {}'.format(
            total_number_instance_over_all_relations))
        print('Total number correct: {}'.format(total_number_correct))
        print('Total number correct in the last layer {}'.format(
            total_number_correct_last_layer))

        print(60 * '*')

        plot_data.append({
            'dataset': dataset,
            'total_instances': total_number_instance_over_all_relations,
            'correct': total_number_correct,
            'correct_last_layer': total_number_correct_last_layer
        })

    do_plot(plot_data, model_name)

    return {'model_name': model_name, 'plot_data': plot_data}


def get_rank(model_data, dataset, relation, layer, uuid):
    individual_predictions = model_data[str(
        layer)][dataset][relation][0]['individual_predictions']

    # topk_predictions = []
    for pred in individual_predictions:
        if pred['sample']['uuid'] == uuid:
            rank = pred['rank']
            # topk_predictions = pred['top_k_tokens']
            # print('Top 10 predictions: {}'.format(topk_predictions[:10]))

    return rank


def find_matching_prediction(cur_prediction, indidvidual_predictions_data):
    cur_uuid = cur_prediction['sample']['uuid']

    for prediction in indidvidual_predictions_data:
        if prediction['sample']['uuid'] == cur_uuid:
            return prediction

    print('ERROR: NO MATCHING prediction')


def do_plot(plot_data, model_name):
    datasets = []

    last_layer_values = []
    union_values = []
    last_layer_labels = []
    union_labels = []

    for ds_data in plot_data:
        datasets.append(ds_data['dataset'])
        total_instances = ds_data['total_instances']
        num_correct = ds_data['correct']
        num_correct_last_layer = ds_data['correct_last_layer']

        last_layer_values.append(num_correct_last_layer / total_instances)
        last_layer_labels.append(
            round(num_correct_last_layer / total_instances, 2))

        union_values.append(
            (num_correct - num_correct_last_layer) / total_instances)
        union_labels.append(
            round(num_correct / total_instances, 2))

    fig = go.Figure(data=[
        go.Bar(name='Last layer', x=datasets, y=last_layer_values,
               text=last_layer_labels, textposition='auto'),
        go.Bar(name='Union', x=datasets, y=union_values,
               text=union_labels, textposition='auto')
    ])
    # Change the bar mode
    fig.update_layout(
        barmode='stack', plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0, y=1))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)', nticks=6)
    fig.update_yaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)', nticks=6)

    fig.write_image(
        "{}{}.png".format(output_dir, model_name))


def got_prediction_right(prediction_data, metric):
    return prediction_data[metric] == 1.0


def smart_load_data(dir):
    all_data = {}

    for layer in tqdm(layer_range):
        print('Loading layer file: ', layer)
        data_file = get_json_data_file_for_layer(
            dir, layer)
        layer_data = load_json_data(data_file)

        # remove topk tokens
        for dataset in layer_data.keys():
            for relation in layer_data[dataset].keys():
                if relation != 'means':
                    if 'individual_predictions' in layer_data[dataset][relation][0]:
                        for individual_prediction in layer_data[dataset][relation][0]['individual_predictions']:
                            del individual_prediction['top_k_tokens']

        gc.collect()

        all_data[str(layer)] = layer_data

    return all_data


def load_json_data(file):
    with open(file) as json_data:
        data = json.load(json_data)
    return data


def get_subfolders(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def get_layer_folder(data_base_dir, layer):
    list_subfolders_with_paths = get_subfolders(data_base_dir)
    matching = [
        s for s in list_subfolders_with_paths if "layer_{}".format(layer) in s]

    if len(matching) > 1:
        print('Found more than one dir with for layer {}. Using this one: {}'.format(
            layer, sorted(matching)[0]))
    return sorted(matching)[0]


def get_files_in_folder(dir):
    return [f.path for f in os.scandir(dir) if f.is_file()]


def get_json_data_file_for_layer(dir, layer):
    layer_dir = get_layer_folder(dir, layer)
    all_files = get_files_in_folder(layer_dir)
    json_files = [f for f in all_files if ".json" in f]
    if len(json_files) > 1:
        print('Found more than one json data file in dir. Using this one: {}'.format(
            json_files[0]))
    if len(json_files) == 0:
        print('No json files found in {}'.format(layer_dir))
        return None
    return json_files[0]


def get_example_predictions(data_dir, layer, relation, dataset, uuid):
    data_file = get_json_data_file_for_layer(data_dir, layer)
    layer_data = load_json_data(data_file)

    individual_predictions = layer_data[dataset][relation][0]['individual_predictions']

    topk_predictions = []
    for pred in individual_predictions:
        if pred['sample']['uuid'] == uuid:
            print(pred['sample'])
            topk_predictions = pred['top_k_tokens']
            print(topk_predictions[:10])


def do_wide_bar_chart(plot_datas):

    datasets = ['Google RE', 'T-REx', 'ConceptNet', 'Squad']

    fig = go.Figure()

    model_colors = {
        'BERT': ['rgba(0, 0, 0, 0.6)', 'rgba(0, 0, 0, 0.45)'],
        'QA-SQUAD-1': ['rgba(255, 127, 80, 0.6)', 'rgba(255, 127, 80, 0.45)'],
        'QA-SQUAD-2': ['rgba(255, 0, 0, 0.6)', 'rgba(255, 0, 0, 0.45)'],
        'MLM-SQUAD': ['rgba(117, 0, 0, 0.6)', 'rgba(117, 0, 0, 0.45)'],
        'RANK-MSMARCO': ['rgba(0, 0, 255, 0.6)', 'rgba(0, 0, 255, 0.45)'],
        'MLM-MSMARCO': ['rgba(30, 144,255, 0.6)', 'rgba(30, 144,255, 0.45)'],
        'NER-CONLL': ['rgba(160,32,240, 0.6)', 'rgba(160,32,240, 0.45)']
    }

    model_offset_index = {
        'BERT': 1,
        'QA-SQUAD-1': 2,
        'QA-SQUAD-2': 3,
        'MLM-SQUAD': 4,
        'MLM-MSMARCO': 5,
        'RANK-MSMARCO': 6,
        'NER-CONLL': 7
    }

    for offsetgroup_index, data in enumerate(plot_datas):
        model_name = data['model_name']

        offsetgroup_index = model_offset_index[model_name]

        plot_data = data['plot_data']
        model_color = model_colors[model_name][0]
        union_color = model_colors[model_name][1]

        last_layer_values = []
        union_values = []
        last_layer_labels = []
        union_labels = []

        print('Plot data: {}'.format(plot_data))

        for ds_data in plot_data:

            print('Ds data: {}'.format(ds_data))

            total_instances = ds_data['total_instances']
            num_correct = ds_data['correct']
            num_correct_last_layer = ds_data['correct_last_layer']

            last_layer_values.append(num_correct_last_layer / total_instances)
            last_layer_labels.append('{}'.format(
                round(num_correct_last_layer / total_instances, 2)))

            union_values.append(
                (num_correct - num_correct_last_layer) / total_instances)
            union_labels.append('<b>{}</b>'.format(
                round(num_correct / total_instances, 2)))

        # Plot last layer and union on top of each other
        last_layer_bar = go.Bar(name=model_name, x=datasets, y=last_layer_values, marker=dict(color=model_color),
                                text=last_layer_labels, textposition='auto', offsetgroup=offsetgroup_index)
        union_bar = go.Bar(showlegend=False, name='{} Union'.format(model_name), x=datasets, y=union_values, marker=dict(color=union_color),
                           text=union_labels, textposition='auto', base=last_layer_values, offsetgroup=offsetgroup_index)

        fig.add_trace(last_layer_bar)
        fig.add_trace(union_bar)

    # Change the bar mode
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0, y=1), autosize=False,
                      width=1500,
                      height=650,
                      font=dict(size=15)
                      )
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)', nticks=6)
    fig.update_yaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)', nticks=6)
    fig.show()

    fig.write_image(
        "{}wide.png".format(output_dir))


if __name__ == "__main__":

    plot_datas = []

    # default = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/default/',
    #     'name': 'Default'
    # }

    # get_example_predictions(
    #     default['data_dir'], 6, 'P31', 'TREx', 29)
    # print('\n\nLayer 12:')
    # get_example_predictions(
    #     default['data_dir'], 12, 'P31', 'TREx', 29)

    # model_data = smart_load_data(default['data_dir'])
    # plot_datas.append(stats_for_avi(model_data, 'Default'))

    default = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/bert/',
        'name': 'BERT'
    }

    # get_example_predictions(
    #     default['data_dir'], 10, 'P19', 'TREx', 724)
    # print('\n\nLayer 12:')
    # get_example_predictions(
    #     default['data_dir'], 12, 'P19', 'TREx', 724)

    model_data = smart_load_data(default['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'BERT'))

    # test_individual_prediction_order_constant(model_data)
    del model_data
    gc.collect()

    squad_uncased = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_qa_1/',
        'name': 'QA-SQUAD-1'
    }
    model_data = smart_load_data(squad_uncased['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'QA-SQUAD-1'))

    del model_data
    gc.collect()

    squad_2_uncased = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_qa_2/',
        'name': 'QA-SQUAD-2'
    }
    model_data = smart_load_data(squad_2_uncased['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'QA-SQUAD-2'))

    del model_data
    gc.collect()

    squad_mlm = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/10/',
        'name': 'MLM-SQUAD'
    }
    model_data = smart_load_data(squad_mlm['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'MLM-SQUAD'))

    del model_data
    gc.collect()

    msmarco_ranking = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/marco_rank/',
        'name': ' RANK-MSMARCO'
    }
    model_data = smart_load_data(msmarco_ranking['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'RANK-MSMARCO'))

    del model_data
    gc.collect()

    msmarco_mlm = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/marco_mlm/',
        'name': 'MLM-MSMARCO'
    }
    model_data = smart_load_data(msmarco_mlm['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'MLM-MSMARCO'))

    del model_data
    gc.collect()

    ner = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/ner/',
        'name': 'NER-CONLL'
    }
    model_data = smart_load_data(ner['data_dir'])
    plot_datas.append(stats_for_avi(model_data, 'NER-CONLL'))

    del model_data
    gc.collect()

    do_wide_bar_chart(plot_datas)
