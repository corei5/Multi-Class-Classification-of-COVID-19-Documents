# #pip install crossrefapi
from crossref.restful import Works

def extract_paper_info(doi,df):
    works = Works()
    info = works.doi(doi)
    row_indexes = df[df['doi'] == doi].index

    try:
        #print(info['short-container-title'][0])
        df.loc[row_indexes, 'journal_name'] = info['short-container-title'][0]
    except:
        pass
    try:
        #print(info['type'])
        df.loc[row_indexes, 'type_of_paper'] = info['type']
    except:
        pass
    # try:
    #     print(info['created'])
    # except:
    #     pass
    try:
        #print(info['is-referenced-by-count'])
        df.loc[row_indexes, 'citation_count'] = info['is-referenced-by-count']
    except:
        pass
    try:
        #print(info['author'])
        #print(len(info['author']))
        df.loc[row_indexes, 'author_count'] = len(info['author'])
    except:
        pass
    try:
        #print(info['published-online']['date-parts'][0][0])
        df.loc[row_indexes, 'published_year'] = info['published-online']['date-parts'][0][0]
    except:
        pass
    try:
        #print(info['references-count'])
        df.loc[row_indexes, 'references_count'] = info['references-count']
    except:
        pass
    try:
        #print(info['subject'])
        paper_subject_listToStr = ' '.join([str(elem) for elem in info['subject']])
        #print(paper_subject_listToStr)
        df.loc[row_indexes, 'journal_type'] = paper_subject_listToStr
    except:
        pass

    return df


# # for key, value in info.items():
# #     print(key, ' : ', value)