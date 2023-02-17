import pandas as pd
import os
from rdflib import Graph
from Data_prepossessing import IsNotAscii

def binary_document_matrix_with_PubTator_entities(documents):
    FHIR_DATA_PATH = 'inputs/PUBMED/'

    matrix_PubTator = pd.DataFrame(index=documents.index, columns=[])
    entities_doc = set()

    for index, document in documents.iterrows():
        path = FHIR_DATA_PATH + str((document.pmcid)) + ".ttl"
        if os.path.isfile(path):
            # print("processing " + path)
            pubmed_record = open(path, 'r', encoding="utf8")
            g = Graph()
            g.parse(pubmed_record, format='turtle')
            qres = g.query(
                """SELECT DISTINCT ?text
                   WHERE {
                      ?a pmc:text ?text .
                   }""")

            # print(qres)

            for row in qres:
                # print(row)
                extractedEntity = str(row.asdict()['text'].toPython())

                # print(extractedEntity)

                if len(extractedEntity) < 40 and not (IsNotAscii.is_not_ascii(str(extractedEntity))):
                    matrix_PubTator.at[document.name, extractedEntity] = 1

    matrix_PubTator.to_csv('cache/Entity_matrix/matrix_fhir_binary.csv')

    print("Finish matrix_PubTator")

    return matrix_PubTator