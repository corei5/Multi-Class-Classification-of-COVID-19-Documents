
def connected_matrices(new_ids, matrix_PubTator_pd, matrix_authors_pd, matrix_bow_pd, matrix_tfidf_pd, matrix_authors_tfidf_pd):
    matrix_PubTator_pd = matrix_PubTator_pd.merge(new_ids, how="inner", left_index=True, right_index=True)
    matrix_authors_pd = matrix_authors_pd.merge(new_ids, how="inner", left_index=True, right_index=True)
    matrix_bow_pd = matrix_bow_pd.merge(new_ids, how="inner", left_index=True, right_index=True)
    matrix_tfidf_pd = matrix_tfidf_pd.merge(new_ids, how="inner", left_index=True, right_index=True)

    matrix_authors_tfidf_pd = matrix_authors_tfidf_pd.merge(new_ids, how="inner", left_index=True, right_index=True)

    #bow_plus_bibliometric_features = bibliomet_matrix_randomforest.merge(matrix_bow_pd, how='inner', left_index=True, right_index=True)
    #tfidf_plus_bibliometric_features = bibliomet_matrix_tfidf.merge(matrix_tfidf_pd, how='inner', left_index=True, right_index=True)
    #bow_plus_PubTator_pd = matrix_PubTator_pd.merge(matrix_bow_pd, how='inner', left_index=True, right_index=True)
    #bow_plus_bibliometric_features_rule = bibliomet_matrix_rulemining.merge(matrix_bow_pd, how='inner', left_index=True, right_index=True)
    #bow_plus_bibliometric_features_rule = bow_plus_bibliometric_features_rule.loc[:, bow_plus_bibliometric_features_rule.columns != 'Target']

    return matrix_PubTator_pd, matrix_authors_pd, matrix_bow_pd, matrix_tfidf_pd, matrix_authors_tfidf_pd
