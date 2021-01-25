import os
import json
import gc
from tqdm import tqdm
from plotly import graph_objects as go
from scipy import stats

metrics = ['P_AT_1', 'P_AT_10', 'P_AT_K']
nice_metric_names = {
    'P_AT_1': 'P@1',
    'P_AT_10': 'P@10',
    'P_AT_K': 'P@100'
}

nice_dataset_names = {
    'Google_RE': 'Google-RE',
    'TREx': 'T-REx',
    'ConceptNet': 'ConceptNet',
    'Squad': 'SQUAD'
}

relation_label_map = {
    'P407': 'language of',
    'P27': 'country of citizenship',
    'P530': 'diplomatic relation',
    'P449': 'originally aired on',
    'P37': 'official language of',
    'P101': 'field of work',
    'P279': 'subclass of',
    'P463': 'member of',
}

relation_marker_map = {
    'P530': 'star',
    'P37': 'square-dot',
    'P463': 'triangle-up',
    'P279': 'cross',
    'P449': 'diamond',
}

relation_group_order_map = {
    'P407': "0",
    'P27': "1",
    'P530': "2",
    'P449': "3",
    'P37': "4",
    'P101': "5",
    'P279': "6",
    'P463': "7",
}

layer_range = range(1, 13)


font_size_wide_plots = 30
# font_size_wide_plots = 20
font_size_legend_text = 32
# font_size_legend_text = 20
# output_dir = '/home/jonas/git/knowledge-probing/data/plots/new/default_font/default/'
output_dir = '/home/jonas/git/knowledge-probing/data/plots/new_vs_old_mlm_marco_large_font/'
# output_dir = '/home/jonas/git/knowledge-probing/data/plots/mlm_vs_mlm_long/'


def main():
    make_plots_dir(output_dir)

    selected_models = select_model_for_comparison()

    for model in selected_models:
        print(60 * "*" + '    Loading model {}    '. format(model['name']))
        model['data'] = smart_load_data(model['data_dir'])

        # print_win_loss_statistics(model)
        print_double_last_layer_statistics(model)

    # all_data = smart_load_data()

    # print('Sample path: {}'.format(selected_models[0]['data_dir']))
    data_sample = selected_models[0]['data']['1']

    datasets = data_sample.keys()

    for dataset in datasets:
        for relation in tqdm(data_sample[dataset].keys()):
            print(relation)
            if relation == 'means':
                # Do mean plot
                if len(data_sample[dataset][relation]) > 0:
                    do_mean_plot(selected_models, dataset, relation)

            else:
                do_layer_plots(selected_models, dataset, relation)

    do_parallel_plots(selected_models)
    do_multiple_means_plot(selected_models)


def do_parallel_plots(models):
    data_sample = models[0]['data']['1']
    datasets = data_sample.keys()

    for model in models:
        parallel_plots_dir = output_dir + '/paralel/' + model['name'] + '/'
        make_plots_dir(parallel_plots_dir)

        for dataset in datasets:
            for metric in metrics:

                relations_data = []
                relations = []

                for relation in tqdm(data_sample[dataset].keys()):
                    if relation != 'means':
                        if relation in ['P530', 'P37', 'P463', 'P449', 'P279']:
                            relations.append(relation)

                            relations_item = {
                                'relation': relation,
                                'precision': []
                            }

                            for layer in layer_range:
                                relations_item['p_{}'.format(
                                    layer + 1)] = model['data'][str(layer)][dataset][relation][0][metric] * 100
                                relations_item['precision'].append(
                                    model['data'][str(layer)][dataset][relation][0][metric] * 100)

                            relations_data.append(relations_item)

                # fig.show()

                fig = go.Figure()

                relations_data.sort(key=lambda x: len(
                    relation_label_map[x['relation']]), reverse=True)

                for item in relations_data:
                    fig.add_trace(go.Scatter(
                        x=list(layer_range), y=item['precision'], name=relation_label_map[item['relation']], hoverinfo=['all'], mode='lines+markers', marker=dict(
                            symbol=relation_marker_map[item['relation']],
                            size=12
                        )))

                fig.update_layout(legend=dict(
                    bgcolor="rgba(255,255,255,0)",
                ))

                fig.update_yaxes(range=[0, 80])

                # fig.update_layout(title='{}  -  {}'.format(dataset, metric),
                #                   xaxis_title='Layer',
                #                   #   autosize=False,
                #                   #   width=650,
                #                   #   height=1000,
                #                   yaxis_title='{}'.format(nice_metric_names[metric]), plot_bgcolor='rgb(255,255,255)')

                # # fig.show()

                # fig.write_image(
                #     "{}{}-{}.png".format(parallel_plots_dir, dataset, metric))

                do_narrow_plot(fig, parallel_plots_dir + 'narrow/',
                               dataset, relation, metric)

                do_wide_plot(fig, parallel_plots_dir,
                             dataset, relation, metric)


def print_win_loss_statistics(model):
    # For every relation, find in which layer the value is the hightest
    model_data = model['data']
    # print('Model data {}'.format(model['data']))
    sample = model_data['1']

    print(80 * '*' + '     Model: {}    '.format(model['name']))

    for dataset in sample.keys():
        print(40 * '-' + '    Dataset {}'.format(dataset))

        for metric in metrics:
            print(20 * '~' + '     Metric {}'.format(metric))
            num_wins = 0
            num_loss = 0
            num_relations = 0
            for relation in sample[dataset].keys():
                num_relations = num_relations + 1
                if relation != 'means':
                    highest_value = -1
                    highest_value_layer_index = 1
                    for layer in layer_range:
                        layer_data = model_data[str(layer)]

                        layer_value = layer_data[dataset][relation][0][metric] * 100

                        if layer == 12:
                            last_layer_value = layer_value

                        if layer_value > highest_value:
                            highest_value = layer_value
                            highest_value_layer_index = layer

                    if highest_value_layer_index != 12 and highest_value > 0 and highest_value > last_layer_value:
                        if significant_difference(model_data, dataset, relation, metric, highest_value_layer_index):
                            num_loss = num_loss + 1
                            print('Found loss: For {} the {} was highest in layer {} ({} vs {} in layer 12)'.format(
                                relation, metric, highest_value_layer_index, highest_value, last_layer_value))
                    else:
                        num_wins = num_wins + 1

            print('Last layer not highest performance in {} out of {} relations. Fraction: {}'.format(
                num_loss, num_relations, (num_loss/num_relations)))


def print_double_last_layer_statistics(model):
    # For every relation, find in which layer the value is the hightest
    model_data = model['data']
    # print('Model data {}'.format(model['data']))
    sample = model_data['1']

    print(80 * '*' + '     Model: {}    '.format(model['name']))

    for dataset in sample.keys():
        print(40 * '-' + '    Dataset {}'.format(dataset))

        for metric in metrics:
            print(20 * '~' + '     Metric {}'.format(metric))
            num_doubles = 0
            num_relations = 0

            for relation in sample[dataset].keys():
                if relation != 'means':
                    num_relations = num_relations + 1
                    highest_value = -1
                    highest_value_layer_index = 1
                    for layer in layer_range:
                        layer_data = model_data[str(layer)]
                        layer_value = layer_data[dataset][relation][0][metric] * 100

                        if layer == 12:
                            last_layer_value = layer_value
                        else:

                            if layer_value > highest_value:
                                highest_value = layer_value
                                highest_value_layer_index = layer

                    if highest_value_layer_index != 12 and highest_value > 0 and highest_value * 2 <= last_layer_value:
                        num_doubles = num_doubles + 1
                        print('Found relation that doubles in last layer: For {} the {} was highest in layer {} ({} vs {} in layer 12)'.format(
                            relation, metric, highest_value_layer_index, highest_value, last_layer_value))

            print('Last layer performance twice as high as in other layers for {} out of {} relations. Fraction: {}'.format(
                num_doubles, num_relations, (num_doubles/num_relations)))


def significant_difference(model_data, dataset, relation, metric, highest_value_layer_index):
    highest_layer_scores = []
    for individual_prediction in model_data[str(highest_value_layer_index)][dataset][relation][0]['individual_predictions']:
        highest_layer_scores.append(individual_prediction[metric])

    last_layer_scores = []
    for individual_prediction in model_data['12'][dataset][relation][0]['individual_predictions']:
        last_layer_scores.append(individual_prediction[metric])

    _, pvalue = stats.ttest_ind(highest_layer_scores, last_layer_scores)

    print('Pvalue: {}'.format(pvalue))

    return pvalue <= 0.05


def smart_load_data(dir):
    all_data = {}

    for layer in tqdm(layer_range):
        print('Loading layer file: ', layer)
        data_file = get_json_data_file_for_layer(
            dir, layer)
        layer_data = load_json_data(data_file)

        layer_data = compute_error_values(layer_data)

        all_data[str(layer)] = layer_data

    return all_data


def compute_error_values(layer_data):
    print('TODO: compute error values')

    # compute error values and add them to each relation
    # add error value to relation element

    # remove individual predictions for every relation
    layer_data = remove_individual_predictions(layer_data)

    return layer_data


def remove_individual_predictions(layer_data):
    for dataset in layer_data.keys():
        for relation in layer_data[dataset].keys():
            if relation != 'means':
                if 'individual_predictions' in layer_data[dataset][relation][0]:
                    for individual_prediction in layer_data[dataset][relation][0]['individual_predictions']:
                        del individual_prediction['top_k_tokens']
    gc.collect()

    return layer_data


def do_mean_plot(models, dataset, relation):
    mean_plot_dir = output_dir + '/means/'
    make_plots_dir(mean_plot_dir)

    for metric_index, metric in enumerate(metrics):
        fig = go.Figure()

        for model in models:
            model_data = model['data']
            model_mean_values = []

            for layer in layer_range:
                layer_data = model_data[str(layer)]

                means_string = layer_data[dataset]['means'][0]
                mean_values = handle_mean_values_string(means_string)
                model_mean_values.append(mean_values[metric_index] * 100)

            fig.add_trace(go.Scatter(x=list(layer_range),
                                     y=model_mean_values, name=model['name'], hoverinfo=['all'], mode='lines+markers', line_color=model['color'], marker=dict(
                color=model['color'],
                symbol=model['marker'],
                size=12
            )))

        # fig.update_layout(title='{} {} -  {}'.format(dataset, relation, metric),
        #                   xaxis_title='Layer',
        #                   yaxis_title='{}'.format(nice_metric_names[metric]), showlegend=True, plot_bgcolor='rgb(255,255,255)')

        # # fig.show()

        # fig.write_image(
        #     "{}/{}_{}_{}.png".format(mean_plot_dir, dataset, relation, metric))

        do_narrow_plot(fig, mean_plot_dir + 'narrow/',
                       dataset, relation, metric)

        do_wide_plot(fig, mean_plot_dir, dataset, relation, metric)


def do_multiple_means_plot(models):
    data_sample = models[0]['data']['1']
    datasets = data_sample.keys()

    dataset_marker_map = {
        'Google_RE': 'diamond',
        'TREx': 'star',
        'ConceptNet': 'triangle-up',
        'Squad': 'square'
    }

    for model in models:
        multiple_means_plots_dir = output_dir + \
            '/multiple_means_/' + model['name'] + '/'
        make_plots_dir(multiple_means_plots_dir)

        model_data = model['data']

        for metric_index, metric in enumerate(metrics):
            fig = go.Figure()
            for dataset in datasets:
                dataset_mean_values = []
                print('Dataset: {}'.format(dataset))
                if dataset in ['Google_RE', 'TREx']:
                    print('Means in dataset')
                    relation = 'means'
                    for layer in layer_range:
                        layer_data = model_data[str(layer)]
                        # print(layer_data)

                        means_string = layer_data[dataset]['means'][0]
                        mean_values = handle_mean_values_string(means_string)
                        dataset_mean_values.append(
                            mean_values[metric_index] * 100)
                else:
                    relation = 'test'
                    for layer in layer_range:
                        layer_data = model_data[str(layer)]

                        dataset_mean_values.append(
                            layer_data[dataset][relation][0][metric] * 100)

                fig.add_trace(go.Scatter(x=list(layer_range),
                                         y=dataset_mean_values, name=nice_dataset_names[dataset], hoverinfo=['all'], mode='lines+markers', marker=dict(
                    symbol=dataset_marker_map[dataset],
                    size=12
                )))
            do_narrow_plot(fig, multiple_means_plots_dir + 'narrow/',
                           dataset, relation, metric)

            do_wide_plot(fig, multiple_means_plots_dir,
                         dataset, relation, metric)

            # fig.update_layout(xaxis_title='Layer',
            #                   yaxis_title='{}'.format(nice_metric_names[metric]), showlegend=True, plot_bgcolor='rgb(255,255,255)')

            # fig.write_image(
            #     "{}/multiple_means_{}.png".format(multiple_means_plots_dir, metric))


def do_layer_plots(models, dataset, relation):
    # print('Layerwise plots')
    for metric in metrics:
        fig = go.Figure()

        for model in models:
            model_data = model['data']
            model_values = []

            for layer in layer_range:
                layer_data = model_data[str(layer)]

                model_values.append(
                    layer_data[dataset][relation][0][metric] * 100)

            fig.add_trace(go.Scatter(x=list(layer_range),
                                     y=model_values, name=model['name'], hoverinfo=['all'], mode='lines+markers', line_color=model['color'], marker=dict(
                color=model['color'],
                symbol=model['marker'],
                size=12
            )))

        do_narrow_plot(fig, output_dir + 'narrow/', dataset, relation, metric)

        do_wide_plot(fig, output_dir, dataset, relation, metric)


def do_narrow_plot(fig, output_dir, dataset, relation, metric):
    make_plots_dir(output_dir)

    fig.update_layout(xaxis_title='Layer',
                      autosize=False,
                      width=650,
                      height=1000,
                      font=dict(
                          family="Courier New, monospace",
                          size=20,
                          color="black"
                      ),
                      yaxis_title='{}'.format(nice_metric_names[metric]), showlegend=True, plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0, y=1))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)', nticks=6)
    fig.update_yaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)', nticks=6)

    # fig.show()

    fig.write_image(
        "{}/{}_{}_{}.png".format(output_dir, dataset, relation, metric))


def got_prediction_right(prediction_data, metric):
    return prediction_data[metric] > 0


def do_wide_plot(fig, output_dir, dataset, relation, metric):
    make_plots_dir(output_dir)

    fig.update_layout(xaxis_title='',
                      yaxis_title='',
                      autosize=True,
                      font=dict(
                          family="Courier New, monospace",
                          size=font_size_wide_plots,  # default for
                          color="black"
                      ),
                      # yaxis_title='{}'.format(nice_metric_names[metric])
                      showlegend=True, plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(x=0, y=1,
                                  font=dict(
                                      family="sans-serif",
                                      size=font_size_legend_text,
                                      color="black"
                                  ),
                                  bgcolor="rgba(255,255,255,0)",
                                  ))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)')
    fig.update_yaxes(showline=True, linewidth=1,
                     linecolor='black', showgrid=True, gridcolor='rgba(200,200,200,0.4)')

    # fig.show()

    fig.write_image(
        "{}/{}_{}_{}.png".format(output_dir, dataset, relation, metric))


def handle_mean_values_string(mean_vals_string):
    values_string = mean_vals_string.split(':')[1]

    values = values_string.split(',')

    numeric_vals = []

    for val in values:
        val = val.strip()
        numeric_vals.append(float(val))

    return numeric_vals


def make_plots_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_sample_data_file(dir):
    json_file = get_json_data_file_for_layer(dir, layer=12)
    sample_data = load_json_data(json_file)

    return sample_data


def get_subfolders(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def load_json_data(file):
    with open(file) as json_data:
        data = json.load(json_data)
    return data


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


def select_model_for_comparison():
    selected_models = []

    # default = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/bert/',
    #     'name': 'BERT',
    #     'marker': 'circle',
    #     'color': 'black'
    # }
    # selected_models.append(default)

    # squad_uncased = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_qa_1/',
    #     'name': 'QA-SQUAD-1',
    #     'marker': 'triangle-down',
    #     'color': 'coral'
    # }
    # selected_models.append(squad_uncased)

    # squad_2_uncased = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_qa_2/',
    #     'name': 'QA-SQUAD-2',
    #     'marker': 'triangle-up',
    #     'color': 'red'
    # }
    # selected_models.append(squad_2_uncased)

    # squad_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/10/',
    #     'name': 'MLM-SQUAD',
    #     'marker': 'diamond',
    #     'color': 'darkred'
    # }
    # selected_models.append(squad_mlm)

    # msmarco_ranking = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/marco_rank/',
    #     'name': 'RANK-MSMARCO',
    #     'marker': 'star',
    #     'color': 'blue'
    # }
    # selected_models.append(msmarco_ranking)

    msmarco_mlm = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/marco_mlm/',
        'name': 'MLM-MSMARCO',
        'marker': 'pentagon',
        'color': 'dodgerblue'
    }
    selected_models.append(msmarco_mlm)

    # ner = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/ner/',
    #     'name': 'NER-CONLL',
    #     'marker': 'cross',
    #     'color': 'darkgreen'
    # }
    # selected_models.append(ner)

    ######################################################################################################################################################

    # squad_mlm_pre_trained = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/rnd_vs_pre-trained/pre-trained/layer_data/',
    #     'name': 'Pre-trained',
    #     'marker': 'diamond',
    #     'color': 'orange'
    # }
    # selected_models.append(squad_mlm_pre_trained)

    # squad_mlm_rnd = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/rnd_vs_pre-trained/rnd/layer_data/',
    #     'name': 'Random',
    #     'marker': 'diamond-open',
    #     'color': 'purple'
    # }
    # selected_models.append(squad_mlm_rnd)

    # fc = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/fc_no_fc/fc/',
    #     'name': 'FC layer',
    #     'marker': 'star-triangle-up',
    #     'color': 'blue'
    # }
    # selected_models.append(fc)

    # no_fc = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/fc_no_fc/no_fc/',
    #     'name': 'No FC layer',
    #     'marker': 'star-triangle-up-open',
    #     'color': 'brown'
    # }
    # selected_models.append(no_fc)

    # warmup = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/warmup_no_warmup/warmup/',
    #     'name': 'Warmup',
    #     'marker': 'star-triangle-down',
    #     'color': 'saddlebrown'
    # }
    # selected_models.append(warmup)

    # no_warmup = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/warmup_no_warmup/no_warmup/',
    #     'name': 'No warmup',
    #     'marker': 'star-triangle-down-open',
    #     'color': 'olive'
    # }
    # selected_models.append(no_warmup)

    # sq_mlm_1 = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/1/',
    #     'name': 'MLM-SQUAD-1',
    #     'marker': 'diamond',
    #     'color': 'darkred'
    # }
    # selected_models.append(sq_mlm_1)

    # sq_mlm_6 = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/6/',
    #     'name': 'MLM-SQUAD-6',
    #     'marker': 'diamond-tall',
    #     'color': 'orange'
    # }
    # selected_models.append(sq_mlm_6)

    # sq_mlm_10 = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/squad_mlm_lens/10/',
    #     'name': 'MLM-SQUAD-10',
    #     'marker': 'diamond-wide',
    #     'color': 'darkgreen'
    # }
    # selected_models.append(sq_mlm_10)

    # old_squad_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/old_squad/',
    #     'name': 'MLM-SQUAD-OLD',
    #     'marker': 'diamond-open',
    #     'color': 'darkgreen'
    # }
    # selected_models.append(old_squad_mlm)

    old_marco_mlm = {
        'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/old_marco_mlm/',
        'name': 'MLM-MSMARCO-OLD',
        'marker': 'pentagon-open',
        'color': 'purple'
    }
    selected_models.append(old_marco_mlm)

    return selected_models


if __name__ == "__main__":
    main()


# msmarco_mlm = {
    #     'data_dir': '/home/jonas/git/knowledge-probing/data/outputs/msmarco_mlm/',
    #     'name': 'MLM-MSMARCO',
    #     'marker': 'pentagon',
    #     'color': 'dodgerblue'
    # }
    # selected_models.append(msmarco_mlm)
