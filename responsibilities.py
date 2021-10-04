import json
import os
from collections import defaultdict
import pandas as pd
import spacy

from utils import draw_graph_with_pyvis

english_model = spacy.load('en_core_web_sm')
dataset_dir = os.path.join('data','original_datasets')
responsibilities = []
for file in os.listdir(dataset_dir):
    dataset_key = file.split('_')[0]
    df = pd.read_csv(os.path.join(dataset_dir, file), index_col=[0]).fillna('')
    for index, row in df.iterrows():
        if dataset_key=='google':
            for r in row['Responsibilities'].split('\n'):
                responsibilities.append((dataset_key, row['Title'], r))
        if dataset_key=='amazon':
            description_doc = english_model(row['DESCRIPTION'])
            found_responsibilities=False

            for description_sentence in description_doc.sents:
                if found_responsibilities:
                    print('R:')
                    print(description_sentence.text)
                    responsibilities.append((dataset_key, row['Title'], description_sentence.text))
                if 'responsibilities' in description_sentence.text.lower() and ':' in description_sentence.text.lower():
                    print()
                    print('CLUE:')
                    print(description_sentence.text)
                    parts = description_sentence.text.split(':')
                    if len(parts)>1 and len(parts[1])>3:
                        responsibilities.append((dataset_key, row['Title'], parts[1]))
                        print('R-P:')
                        print(parts[1])
                    found_responsibilities=True
