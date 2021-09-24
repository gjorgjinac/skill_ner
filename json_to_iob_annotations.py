import json
import re
import string
import pandas as pd
from sklearn.metrics import classification_report

import pandas as pd
from nltk import word_tokenize, defaultdict
import spacy
from spacy.tokens.token import Token

spacy_model = spacy.load('en_core_web_sm')
Token.set_extension('iob_tag', default='O', force=True)
Token.set_extension('label', default='O', force=True)
file = 'annotated_data.json'
with open(file, 'r') as in_file:
    data = json.load(in_file)
    print(len(data))
for consider_noun_chunks in [False, True]:


    things_to_save = defaultdict(lambda: [])

    method_names = {'skills'}
    for sample in data:
        for method_name in sample.keys():
            if 'text_' in method_name and 'unique' not in method_name:
                method_names.add(method_name)

    for sample_index, sample in enumerate(data):
        if 'skills' not in sample.keys() or pd.isnull(sample['skills']):
            continue
        annotations = {}

        for method_name in method_names:
            if method_name == 'skills' or consider_noun_chunks:
                annotations[method_name] = []
                skills = sample[method_name].split('|') if method_name == 'skills' else [n['noun_chunk'] for n in
                                                                                         sample[method_name]]
                for skill in skills:
                    for skill_match in re.finditer(re.escape(skill), sample['text']):
                        annotations[method_name].append(
                            {'text': skill, 'start': skill_match.start(), 'end': skill_match.start() + len(skill)})
            else:
                annotations[method_name] = sample[method_name]

        content_doc = spacy_model(sample['text'])

        for token in content_doc:
            for method in method_names:
                token_iob_tag = 'O'
                token_label = 'O'
                for entity in annotations[method]:
                    if entity['start'] <= token.idx and entity['start'] + len(entity['text']) > token.idx:
                        token_iob_tag = 'B' if token.idx == entity['start'] else 'I'
                        if token_iob_tag != 'O':
                            token_label = 'SKILL' if 'label' not in entity.keys() else entity['label']
                            token_label = f'{token_iob_tag}-{token_label}'

                # things_to_save[method + '_label'].append(token_label)
                things_to_save[method + '_iob_tag'].append(token_iob_tag)
            things_to_save['word'].append(token.text.replace('\n', ''))

    pd.DataFrame(things_to_save).to_csv(f'data/annotated_data_iob_{consider_noun_chunks}.csv')



    print(f'Consider noun chunks: {consider_noun_chunks}')
    iob_df = pd.read_csv(f'data/annotated_data_iob_{consider_noun_chunks}.csv', index_col=[0])
    for column in iob_df.columns:
        if column not in ['skills_iob_tag', 'word']:
            model_name = column.replace('text_', '').replace('_iob_tag', '')
            print(model_name)

            concatenate_consecutive = 'concatenated_consecutive' in model_name
            model_name = model_name.replace('_concatenated_consecutive','')
            print(model_name)
            with open(f'data/test_outputs_iob/{model_name}_{concatenate_consecutive}_{consider_noun_chunks}.tsv', 'w') as file:
                for index, row in iob_df.iterrows():
                    file.write(f'{row["word"]}\t{row["skills_iob_tag"]}\t{row[column]}\n')
                    if row["word"] == '.':
                        file.write('\n')
                file.close()
