#Scispacy imports
    #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz
    #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_bionlp13cg_md-0.3.0.tar.gz
    #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_jnlpba_md-0.3.0.tar.gz
    #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_craft_md-0.3.0.tar.gz

import pandas as pd
from Data_prepossessing import IsNotAscii
import spacy
import scispacy
import en_ner_bc5cdr_md
import en_ner_bionlp13cg_md
import en_ner_jnlpba_md
import en_ner_craft_md

nlp = en_ner_bc5cdr_md.load()
bionlp = en_ner_bionlp13cg_md.load()
jnlpba = en_ner_jnlpba_md.load()
craft = en_ner_craft_md.load()

def binary_document_matrix_with_scispacy_entities(df,col_name,path):
    matrix_scispacy_pd = pd.DataFrame(index=df.index, columns=[])
    for index, df in df.iterrows():
        #print("processing " + str(document.name))
        abstract_lower = df[col_name]
        ent_nlp = nlp(abstract_lower)
        ent_bionlp = bionlp(abstract_lower)
        ent_craft = craft(abstract_lower)
        ent_jnlpba = jnlpba(abstract_lower)
        all_ents = set(ent_nlp.ents + ent_bionlp.ents + ent_craft.ents + ent_jnlpba.ents)

        #print(all_ents)

        # new - lemmatization
        all_ents = map(lambda x: x.lemma_, all_ents)

        # TODO it would be better to sanitize entities with non-ascii char rather than completely remove them
        all_ents = filter(lambda x: not (IsNotAscii.is_not_ascii(str(x))), all_ents)
        for ent in all_ents:
            #print(ent)
            matrix_scispacy_pd.at[df.name, str(ent)] = 1

    #matrix_scispacy_pd.shape
    matrix_scispacy_pd.to_csv(path)

    print("Finish matrix_scispacy_binary")

    return matrix_scispacy_pd