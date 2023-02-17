import pandas as pd
from Data_enhancement.Ncbi_api import ncbi_api

def extract_doi_from_ncbi(pmid_list, df):
    for i in range(0, len(pmid_list), 1):
        pmid_info = str(pmid_list[i])
        str_pmid = str(pmid_list[i])

        row_indexes = df[df['pmid'] == pmid_list[i]].index

        try:
            df.loc[row_indexes, 'doi'] = ncbi_api([str_pmid])[0]['doi']

        except:
            pass

    return df