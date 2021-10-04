import json
import os
from collections import defaultdict
import pandas as pd

from utils import draw_graph_with_pyvis

occupations_skills=[]
model_name = 'support_2'
dataset_config = {
    'amazon': {'occupation_column': 'Title',
               'basic_column': 'BASIC QUALIFICATIONS',
               'preferred_column': 'PREFERRED QUALIFICATIONS'},
'google': {'occupation_column': 'Title',
               'basic_column': 'Minimum Qualifications',
               'preferred_column': 'Preferred Qualifications'},
'indeed': {'occupation_column': 'Job Title',
               'basic_column': 'Job Description',
           'preferred_column':''},
}

for file in os.listdir('data/annotated_datasets'):
    dataset_key = file.split('_')[0]
    basic_column = f"{dataset_config[dataset_key]['basic_column']}_{model_name}"
    preferred_column = f"{dataset_config[dataset_key]['preferred_column']}_{model_name}"
    print(dataset_key)
    with open(os.path.join('data', 'annotated_datasets', file), 'r') as f:
        annotated_data = json.load(f)
        for annotated_job in annotated_data:
            occupation = annotated_job[dataset_config[dataset_key]['occupation_column']]
            if basic_column in annotated_job.keys():
                for skill in annotated_job[basic_column]:
                    occupations_skills.append((occupation, skill['text'], skill['label'], 'basic'))
            if preferred_column in annotated_job.keys():
                for skill in annotated_job[preferred_column]:
                    occupations_skills.append((occupation, skill['text'], skill['label'], 'preferred'))

results = pd.DataFrame(occupations_skills, columns=['occupation','skill','label','type'])
occurence_counts_df=results.groupby(['occupation','skill']).count().reset_index()
print(occurence_counts_df)
occurence_counts_df=occurence_counts_df[occurence_counts_df['type']>5]
print(occurence_counts_df)

draw_graph_with_pyvis(occurence_counts_df,'occupation','skill')