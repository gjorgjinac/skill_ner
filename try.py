import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
df = pd.read_csv('annotated_data.csv', index_col=[0])
print(df.groupby('source').count()['text'])
print(df.groupby('Title').count()['text'].drop_duplicates())
print(df['Title'].drop_duplicates())



wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              min_font_size=10).generate(' '.join( df['text'].drop_duplicates()))

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plot_title = f'Job posting title'
plt.title(plot_title)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(f'evaluation_visualization/job_posting_text.png', dpi=None,
            orientation='portrait', transparent=True, pad_inches=0.1)