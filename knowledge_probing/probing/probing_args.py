from knowledge_probing.file_utils import load_file

# The data loading is adapted from the LAMA repository by Petroni et. al. (https://github.com/facebookresearch/LAMA)


def build_args(dataset_name, lowercase, data_dir, k, bert_model_type='bert-base-uncased'):
    relations, data_path_pre, data_path_post = '', '', ''
    if dataset_name == 'Google_RE':
        relations, data_path_pre, data_path_post = get_GoogleRE_parameters(
            data_dir)
    elif dataset_name == 'Google_RE_UHN':
        relations, data_path_pre, data_path_post = get_GoogleRE_UHN_parameters(
            data_dir, bert_model_type)
    elif dataset_name == 'TREx':
        relations, data_path_pre, data_path_post = get_TREx_parameters(
            data_dir)
    elif dataset_name == 'TREx_UHN':
        relations, data_path_pre, data_path_post = get_TREx_UHN_parameters(
            data_dir, bert_model_type)
    elif dataset_name == 'ConceptNet':
        relations, data_path_pre, data_path_post = get_ConceptNet_parameters(
            data_dir)
    elif dataset_name == 'Squad':
        relations, data_path_pre, data_path_post = get_Squad_parameters(
            data_dir)
    else:
        print('Could not find dataset in supported datasets: {}'.format(dataset_name))
        return

    relation_args = []  # Array for args for each run that has to be done
    for relation in relations:
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "template": "",
            "relation": relation['relation'],
            "precision_at_k": k,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]

        relation_args.append(PARAMETERS)

    return relation_args


def get_TREx_parameters(data_dir):
    relations = load_file("{}relations.jsonl".format(data_dir))
    data_path_pre = "{}TREx/".format(data_dir)
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_TREx_UHN_parameters(data_dir, bert_model_type):
    relations = load_file("{}relations.jsonl".format(data_dir))
    data_path_pre = "{}TREx_UHN/{}/".format(data_dir, bert_model_type)
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters(data_dir):
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "{}Google_RE/".format(data_dir)
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_UHN_parameters(data_dir, bert_model_type):
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "{}Google_RE_UHN/{}/".format(data_dir, bert_model_type)
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_dir):
    relations = [{"relation": "test"}]
    data_path_pre = "{}ConceptNet/".format(data_dir)
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_dir):
    relations = [{"relation": "test"}]
    data_path_pre = "{}Squad/".format(data_dir)
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post
