import pandas as pd
from Data_prepossessing import Count_and_tag
from Data_prepossessing import Dotplot

def ards(df):
    ards_synonyms = [
        'acute respiratory distress syndrome', r'\bards\b'
    ]
    df, ards_counts = Count_and_tag.count_and_tag(df, ards_synonyms, 'target_class', 'ards')
    #n = df.tag_disease_ards.sum()
    #print(f'There are {n} papers on Covid-19 and ARDS.')
    return  df



#Vaccines/Immunology, Genomics, Public Health Policies, Epidemiology, Virology, Influenza, Healthcare Industry, Lab Trials (human) and Pulmonary infections