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
matplotlib.rcParams.update({'font.size': 16})
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
        results['dictionary']= '+'.join([dictionary_formatting[d] for d in dictionary.split('+')])
        results['model_name'] = model_name
        results['postprocessing']=postprocessing_description[model_name.replace(dictionary,'')]
        results['false_positives'] = false_positives
        results['false_negatives'] = false_negatives
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
                               ('single', single_dictionary_result_df),
                               #('multiple', multiple_dictionary_result_df)
    ]:
    fig, ax = plt.subplots(1, result_df['dictionary'].drop_duplicates().shape[0], sharey=True, sharex=True)
    for index, dictionary in enumerate(result_df['dictionary'].drop_duplicates().values):
        dictionary_df = result_df[result_df['dictionary']==dictionary]

        ax[index].set_title(dictionary)
        dictionary_df.plot(y=['f1','precision','recall'], x='postprocessing', kind='bar', color=['#77019b','#afd92f','#42a2c8'], ax = ax[index])
        ax[index].get_legend().remove()

    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center')
    plt.tight_layout()
    plt.show()

print(single_dictionary_result_df.columns)

single_dictionary_no_postprocessing_results = single_dictionary_result_df[single_dictionary_result_df['postprocessing']=='None']
for index, row in single_dictionary_no_postprocessing_results.iterrows():
    for error_column in ['false_positives', 'false_negatives']:
        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              min_font_size=10).generate(' '.join([e.replace(' ', '_') for e in row[error_column]]))

        # plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plot_title = f'{row["model_name"]} {error_column}'
        plt.title(plot_title)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(f'evaluation_visualization/{plot_title}.png', dpi=None,
                    orientation='portrait', transparent=True, pad_inches=0.1)
