import json
import os

import pandas as pd

from utils import initialize_ner_model_with_dictionaries, concatenate_consecutive, add_noun_chunks

models_to_run = {}
dictionaries_to_run = [['esco','linkedin','employment']]

for dictionary_names_to_run in dictionaries_to_run:
    dictionaries_locations = [os.path.join('dictionaries', f'{dictionary_name}.txt') for dictionary_name in
                              dictionary_names_to_run]
    models_to_run['+'.join(dictionary_names_to_run)] = initialize_ner_model_with_dictionaries(dictionaries_locations)

save_extractions = True

general_input_dataset_directory = 'data/original_datasets'
general_output_dataset_directory = 'data/annotated_datasets'
datasets = {
    'annotated': {
        'input_file_location': f'annotated_data.csv',
        'columns_to_annotate': ['text'],
        'output_file_location': f'annotated_data.json'
    },

    'google': {
        'input_file_location': f'google_jobs_dataset.csv',
        'columns_to_annotate': ['Minimum Qualifications','Preferred Qualifications'],
        'output_file_location': f'google_jobs_dataset.json'
    },

    'amazon': {
        'input_file_location': f'amazon_jobs_dataset.csv',
        'columns_to_annotate': ['BASIC QUALIFICATIONS', 'PREFERRED QUALIFICATIONS'],
        'output_file_location': f'google_jobs_dataset.json'
    }

}

from collections import defaultdict

use_already_generated = False
for dataset_name in ['google', 'amazon']:

    df_to_annotate = pd.read_csv(os.path.join(general_input_dataset_directory, datasets[dataset_name]["input_file_location"]))
    output_data = []
    output_file_name = os.path.join(general_output_dataset_directory, datasets[dataset_name]["output_file_location"])
    if use_already_generated and os.path.isfile(output_file_name):
        with open(output_file_name, 'r') as out_file:
            output_data = json.load(out_file)
    print(models_to_run)
    print(df_to_annotate.shape)
    rows_to_save = output_data

    for index, row in df_to_annotate.iterrows():
        if index < len(output_data):
            continue
        print(index)
        row_to_save = row.copy().to_dict()
        for column_to_annotate in datasets[dataset_name]['columns_to_annotate']:
            all_unique = defaultdict(lambda: 0)

            for model_name, model in models_to_run.items():

                input_text = row[column_to_annotate]
                if type(input_text) is str:
                    entities = model(input_text).ents
                    all_entities = [{"label": ent.label_, "text": ent.text, 'start': ent.start_char} for ent in
                                    entities]
                    concatenated_entities = concatenate_consecutive(input_text, all_entities)

                    all_entities = add_noun_chunks(input_text, all_entities)
                    row_to_save[f'{column_to_annotate}_{model_name}'] = all_entities
                    concatenated_entities = add_noun_chunks(input_text, concatenated_entities)
                    row_to_save[f'{column_to_annotate}_{model_name}_concatenated_consecutive'] = concatenated_entities

            rows_to_save.append(row_to_save)
            if index % 100 == 0:
                with open(output_file_name, 'w') as out_file:
                    json.dump(rows_to_save, out_file, indent=4)

    with open(output_file_name, 'w') as out_file:
        json.dump(rows_to_save, out_file, indent=4)
