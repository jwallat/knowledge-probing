#!/bin/bash

mkdir ../data
mkdir ../data/outputs

# Training data
mkdir ../data/training_data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
rm wikitext-2-raw-v1.zip
mv wikitext-2-raw/ ../data/training_data/


# Probing data
mkdir ../data/probing_data
wget https://dl.fbaipublicfiles.com/LAMA/data.zip -O lama.zip
unzip lama.zip -d ../data/probing_data/
rm lama.zip

mv data/ConceptNet/ ../data/probing_data/
mv data/Squad/ ../data/probing_data/
mv data/TREx/ ../data/probing_data/
mv data/Google_RE/ ../data/probing_data/
mv data/relations.jsonl ../data/probing_data/

# Prune relations file. There are some relations in the relations.jsonl that are not present in the data. The script will remove these.
python prune_trex_relations.py