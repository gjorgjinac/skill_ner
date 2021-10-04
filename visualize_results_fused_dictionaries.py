import os
import json

import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib.pyplot import figure

postprocessing_description = {
    '_True_True': 'CSC_WNCE',
    '_True_False': 'CSC',
    '_False_False': 'None',
    '_False_True': 'WNCE'
}
dictionary_formatting = {
    'esco': 'ESCO',
    'pdl': 'PDL',
    'employment': 'Employment',
    'linkedin': 'LinkedIn'
}
matplotlib.rcParams.update({'font.size': 14})
all_result_df = pd.DataFrame()
single_dictionary_result_df = pd.DataFrame()
multiple_dictionary_result_df = pd.DataFrame()
for result_file in os.listdir('data/evaluation_results'):
    with open(os.path.join('data', 'evaluation_results', result_file), 'r') as f:
        result = json.load(f)
        performance = result['model']['results']['overall']['performance']
        errors = result['model']['results']['overall']['error_case']
        false_negatives = list(filter(lambda e: e.split('|||')[3] == 'O', errors))
        false_positives = list(filter(lambda e: e.split('|||')[2] == 'O', errors))
        false_positives, false_negatives = [[e.split('|||')[0] for e in err] for err in
                                            [false_positives, false_negatives]]
        model_name = result_file.replace('.json', '').replace('iob_tag_', '').replace('text_', '')

        results = performance.copy()
        dictionary = model_name.replace('_True','').replace('_False','')
        if dictionary=='support_2':
            continue
        results['dictionary']= dictionary
        results['model_name'] = model_name
        results['postprocessing']=postprocessing_description[model_name.replace(dictionary,'')]
        results['false_positives'] = false_positives
        results['false_negatives'] = false_negatives
        results['dictionaries_included']=model_name.count('+')
        all_result_df = all_result_df.append(results, ignore_index=True)

        if '+' in model_name:
            multiple_dictionary_result_df = multiple_dictionary_result_df.append(results, ignore_index=True)
        else:
            single_dictionary_result_df = single_dictionary_result_df.append(results, ignore_index=True)
print(all_result_df)

'''for metric_name in ['f1', 'precision', 'recall']:
    for result_name, result_df in [('all', all_result_df), ('single', single_dictionary_result_df),
                                   ('multiple', multiple_dictionary_result_df)]:
        fig = figure(figsize=(10, 6), dpi=300)
        matplotlib.rc('ytick', labelsize=6)

        sns.barplot(y="model_name", x=metric_name, data=result_df)

        fig.subplots_adjust(left=0.3)

        plt.savefig(f'evaluation_visualization/{result_name}_{metric_name}.png', dpi=None,
                    orientation='portrait', transparent=True)'''

for result_name, result_df in [#('all', all_result_df),
                               #('single', single_dictionary_result_df),
                               ('multiple', multiple_dictionary_result_df)]:

    result_df=result_df[result_df['postprocessing']=='None']
    result_df=result_df.sort_values('dictionaries_included')
    result_df['dictionary']=result_df['dictionary'].apply(lambda dictionary: '+'.join([dictionary_formatting[d] for d in dictionary.split('+')]))
    result_df.plot(y=['f1','precision','recall'], x='dictionary', kind='bar', color=['#77019b','#afd92f','#42a2c8'])
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels, loc='lower center')
    ax.get_legend().remove()
    plt.tight_layout()
    plt.show()
