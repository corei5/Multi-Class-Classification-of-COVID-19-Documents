import pandas as pd
import numpy as np
import re
import numpy
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve,classification_report,confusion_matrix
#BERT random forest
import sys, setuptools, tokenize
import warnings
warnings.filterwarnings('ignore')
# BERT neural network
# need to make sure that you are running TensorFlow 2.0. Google Colab, by default, doesn't run your script on TensorFlow 2.0.
# try:
#     %tensorflow_version 2.x
# except Exception:
#     pass
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import math
import random

from rpy2.robjects import pandas2ri, packages
pandas2ri.activate()
# from __future__ import print_function




class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output

def bert_tokenizer_for_neural_network(df):
    # if you get module 'bert' has no attribute 'bert_tokenization
    # pip install bert-tensorflow
    # pip install bert-for-tf2

    documents_bert_1 = df
    documents_bert_1['abstract'] = documents_bert_1['abstract'].str[:512]
    tokenized = documents_bert_1['abstract'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    # watch out we override tokenizer that has been created before with a BERT tokenizer
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    def preprocess_text(sen):
        # Removing html tags
        sentence = remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence

    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    bert_texts = []
    sentences = list(documents_bert_1['abstract'])
    for sen in sentences:
        bert_texts.append(preprocess_text(sen))

    def tokenize_texts(text_reviews):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

    tokenized_texts = [tokenize_texts(text) for text in bert_texts]

    # Neural Network modelling
    # Preprocess data:

    y = documents_bert_1['tag_target_class']
    y = np.array(list(map(lambda x: 1 if x == "high" else 0, y)))

    add_id = documents_bert_1.reset_index()['doi']
    add_id = np.array(list(add_id))

    reviews_with_len = [[review, y[i], len(review), add_id[i]]
                        for i, review in enumerate(tokenized_texts)]
    for_max_size = [len(review)
                    for i, review in enumerate(tokenized_texts)]

    max_size = max(for_max_size)

    reviews_with_len = [
        [[review[cnt] if cnt < len(review) else 0 for cnt in range(max_size)], y[i], len(review), add_id[i]]
        for i, review in enumerate(tokenized_texts)]

    random.shuffle(reviews_with_len)

    sorted_reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews_with_len]
    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))

    BATCH_SIZE = 32
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), ()))

    next(iter(batched_dataset))

    TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
    # TEST_BATCHES = TOTAL_BATCHES // 10

    # I want to have 30% for testing , not 10%
    TEST_BATCHES = TOTAL_BATCHES // 30

    # batched_dataset.shuffle(TOTAL_BATCHES)
    test_data = batched_dataset.take(TEST_BATCHES)
    train_data = batched_dataset.skip(TEST_BATCHES)

    param1 = {'EMB_DIM': 200,
              'CNN_FILTERS': 100}
    param2 = {'EMB_DIM': 1400,
              'CNN_FILTERS': 130}
    param3 = {'EMB_DIM': 500,
              'CNN_FILTERS': 200}
    param4 = {'EMB_DIM': 1300,
              'CNN_FILTERS': 50}

    OUTPUT_CLASSES = 3

    params_mat = [('param1', param1),
                  ('param2', param2),
                  ('param3', param3),
                  ('param4', param4)]

    for names, params in params_mat:
        print(params.get('EMB_DIM'))
        print(params.get('CNN_FILTERS'))
        text_model = TEXT_MODEL(vocabulary_size=len(tokenizer.vocab),
                                embedding_dimensions=params.get('EMB_DIM'),
                                cnn_filters=params.get('CNN_FILTERS'),
                                dnn_units=256,
                                model_output_classes=3,
                                dropout_rate=0.2)

        if OUTPUT_CLASSES == 2:
            text_model.compile(loss="binary_crossentropy",
                               optimizer="adam",
                               metrics=["accuracy"])
        else:
            text_model.compile(loss="sparse_categorical_crossentropy",
                               optimizer="adam",
                               metrics=["sparse_categorical_accuracy"])

        traning = text_model.fit(train_data, epochs=5)
        text_model.summary()
        results = text_model.evaluate(test_data)
        print(results)
        text_model.predict(test_data) > 0.5
        numpy.set_printoptions(threshold=sys.maxsize)
        y_test_bert = np.concatenate([y for x, y in test_data], axis=0)
        x = documents_bert_1["abstract"]
        X_test_bert = np.concatenate([x for x, x in test_data], axis=0)
        y_prob = text_model.predict(test_data)
        y_classes = y_prob.argmax(axis=-1)
        predicted = np.argmax(y_prob, axis=1)
        print(predicted.size)
        report = pd.DataFrame(classification_report(y_test_bert, predicted, output_dict=True)).transpose()
        print(report)
        report.to_csv('results/neural_network/results_bow_bert_neural_network_' + str(names) + '.csv', index=True)