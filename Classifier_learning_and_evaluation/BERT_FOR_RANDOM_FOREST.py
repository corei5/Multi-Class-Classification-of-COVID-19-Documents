import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve,classification_report,confusion_matrix

#BERT random forest
import sys, setuptools, tokenize
import torch
import tensorflow
from tensorflow import keras
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import IntProgress

def bert_for_random_forest(df):
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    #BERT tokenizer

    documents_bert_1 = df
    documents_bert_1['abstract'] = documents_bert_1['abstract'].str[:512]
    tokenized = documents_bert_1['abstract'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    padded = torch.tensor(padded).to(torch.int64)

    #BERT model (BERT embeddings)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    labels = df['tag_target_class']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)

    #Random forest with BERT

    rf = RandomForestClassifier(bootstrap=True, max_depth=500, max_features=30, min_samples_leaf=10,
                                min_samples_split=2, n_estimators=100)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    clsf_report = pd.DataFrame(classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)).transpose()
    clsf_report.to_csv('results/randomforest/results_bert_randomforest.csv', index=True)
    print(clsf_report)