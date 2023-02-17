import pandas as pd
from Data_prepossessing import Abstract_title_filter
# Helper function which counts synonyms and adds tag column to DF
#def count_and_tag(df: pd.DataFrame, synonym_list, tag_suffix: str) -> (pd.DataFrame, pd.Series):
def count_and_tag(df, synonym_list, tag_suffix, row_name):
    counts = {}
    #df[f'tag_{tag_suffix}'] = False
    for s in synonym_list:
        synonym_filter = Abstract_title_filter.abstract_title_filter(df, s)
        counts[s] = sum(synonym_filter)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = row_name
    return df, pd.Series(counts)

