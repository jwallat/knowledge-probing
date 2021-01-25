import os
import json


def main():

    trex_files = []
    for file in os.listdir("../data/probing_data/TREx/"):
        if file.endswith(".jsonl"):
            trex_files.append(file)

    trex_file_names = []

    for file in trex_files:
        filename = file.split('.')[0]
        trex_file_names.append(filename)

    relations = load_file('../data/probing_data/relations.jsonl')

    pruned_relations = []

    for relation in relations:
        rel_name = relation['relation']
        if rel_name in trex_file_names:
            pruned_relations.append(relation)

    os.remove('../data/probing_data/relations.jsonl')

    with open("../data/probing_data/relations.jsonl", "a") as outfile:
        for line in pruned_relations:
            outfile.write(json.dumps(line))
            outfile.write('\n')


def load_file(filename):
    assert os.path.exists(filename)
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    main()
