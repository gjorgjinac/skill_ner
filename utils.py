
import pandas as pd
import re
import spacy
from spacy.lang.en import English
import json
from spacy.pipeline import EntityRuler
import os
import string



project_dir = os.path.join('drive', 'My Drive', 'Skills')


def get_path_from_project_dir(file_name):
    return os.path.join(project_dir, file_name)

en_core_web_sm = spacy.load('en_core_web_sm')



def concatenate_consecutive(text, extracted_skills):
    for skill1 in extracted_skills:
        for skill2 in extracted_skills:
            if skill1['start'] + len(skill1['text']) + 1 == skill2['start'] and text[skill2['start'] - 1] == ' ':
                combined_skill = {'start': skill1['start'], 'text': skill1['text'] + ' ' + skill2['text']}
                extracted_skills = list(
                    filter(lambda s: s['start'] != skill1['start'] and s['start'] != skill2['start'], extracted_skills))

                extracted_skills.append(combined_skill)
                return extracted_skills
    return extracted_skills


def add_noun_chunks(text, extracted_skills):
    doc = en_core_web_sm(text)
    for skill in extracted_skills:
        skill['noun_chunk'] = skill['processed_noun_chunk'] = skill['text']
        for noun_chunk in doc.noun_chunks:

            if (skill['start'] >= noun_chunk.start_char and skill['start'] < noun_chunk.end_char) or (
                    skill['start'] + len(skill['text']) >= noun_chunk.start_char and skill['start'] + len(
                    skill['text']) <= noun_chunk.end_char):

                noun_chunk_text = ''
                for token in noun_chunk:
                    if not token.is_stop:
                        noun_chunk_text += token.text + ' '

                noun_chunk_parts = re.split('[^\d\s\w\-//]+',
                                            noun_chunk_text)  # split by any character that is not letter, digit or space
                if len(noun_chunk_parts) > 1:
                    correct_noun_chunk_part = None
                    for part in noun_chunk_parts:
                        if part.find(skill['text']) > -1:
                            correct_noun_chunk_part = part
                else:
                    correct_noun_chunk_part = noun_chunk_parts[0]

                correct_noun_chunk_part = correct_noun_chunk_part if correct_noun_chunk_part is not None else skill[
                    'text']
                correct_noun_chunk_part = re.sub('( )*the ', ' ', correct_noun_chunk_part, flags=re.IGNORECASE).replace(
                    '\n', ' ').strip()  # remove 'the'
                correct_noun_chunk_part = correct_noun_chunk_part if len(correct_noun_chunk_part) > len(
                    skill['text']) else skill['text']
                skill['noun_chunk'] = noun_chunk.text
                skill['processed_noun_chunk'] = correct_noun_chunk_part

    return extracted_skills



def generate_patterns_from_dictionaries(dictionary_locations):
    patterns = []
    for dictionary_location in dictionary_locations:
        data = pd.read_csv(dictionary_location, sep="|").dropna()
        for index, row in data.iterrows():
            words = row['term'].split(' ') if row['term'].find(' ') > -1 else [row['term']]
            pattern = [{'LOWER': word.lower()} if not word.isupper() else {'TEXT': word} for word in words]
            patterns.append({"label": row['label'], "pattern": pattern})
    return patterns


def initialize_ner_model_with_dictionaries(dictionary_locations):
    patterns = generate_patterns_from_dictionaries(dictionary_locations)
    print(dictionary_locations)
    print(f'Total patterns: {len(patterns)}')
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    return nlp
