import pandas as pd
from Data_prepossessing import Count_and_tag
from Data_prepossessing import Dotplot

def risk_factor(df):
    risk_factors_synonyms = [
        'risk factors'
    ]
    df, risk_factors_counts = Count_and_tag.count_and_tag(df, risk_factors_synonyms, 'target_class', 'risk_factors')
    #print(Dotplot.dotplot(risk_factors_counts, 'risk factors synonyms in title / abstract metadata'))
    risk_factors_counts.sort_values(ascending=False)
    #n = df.tag_risk_factors.sum()
    #print(f'There are {n} papers on Covid-19 and risk factors')
    return df
