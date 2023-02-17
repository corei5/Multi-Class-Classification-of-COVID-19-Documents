from io import StringIO


def category_id(df):
    df['category_id'] = df['tag_target_class'].factorize()[0]
    category_id_df = df[['tag_target_class', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'tag_target_class']].values)
    return df