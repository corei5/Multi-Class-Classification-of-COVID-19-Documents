import pandas as pd
from Data_prepossessing import Count_and_tag
from Data_prepossessing import Dotplot

def vaccination(df):
    vaccination_synonyms = [
        'vaccination',
        'vaccine'
    ]
    df, vaccination_counts = Count_and_tag.count_and_tag(df, vaccination_synonyms, 'target_class', 'vaccination')
    #dotplot(vaccination_counts, 'vaccination synonyms in title / abstract metadata')
    vaccination_counts.sort_values(ascending=False)
    #n = df.tag_vaccination.sum()
    #print(f'There are {n} papers on Covid-19 and vaccination')
    return df

