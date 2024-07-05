import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, BertModel
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import pickle
from kobert_tokenizer import KoBERTTokenizer
import csv
import datetime
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from collections import Counter

data = pd.read_csv('preprocessedCsv.csv')
data.columns = ['combined_column', 'result']

##    불용어 처리 및 단어길이 1 처리
# with open('korean_stopwords.txt', 'r',encoding='utf-8') as f:
#   list_file = f.readlines() 
# stopwords = [word.strip() for word in list_file]

# okt = Okt()
# stopWordList = []
# for sentence in tqdm(data['combined_column']):
#     tokenized_sentence = okt.morphs(sentence, stem=True)
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords and len(word) > 1]
#     joined_sentence = ' '.join(stopwords_removed_sentence)
#     stopWordList.append(joined_sentence)
# with open('stopWordList.pkl', 'wb') as f:
#     pickle.dump(stopWordList, f)
with open('stopWordList.pkl', 'rb') as f:
    stopWordList = pickle.load(f)

##  빈도 수 낮은 단어 제외
allWords = ' '.join(stopWordList).split()
word_counts = Counter(allWords)
threshold = 5
# totalWordCount = sum(word_counts.values())
# threshWord = sum(count for count in word_counts.values() if count<= threshold)
# threshWordRatio = threshWord / totalWordCount
# print(f"전체 단어 수: {totalWordCount}")
# print(f"등장 빈도가 {threshold} 이하인 단어 수: {threshWord}")
# print(f"등장 빈도가 {threshold} 이하인 단어의 비율: {threshWordRatio:.2%}")
filtered_words = {word for word, count in word_counts.items() if count > threshold}
def filter_sentence(sentence):
    return ' '.join([word for word in sentence.split() if word in filtered_words])
thresholdWords = [filter_sentence(sentence) for sentence in stopWordList]
data['preprocess'] = thresholdWords

##  길이 분포 확인하기
# data['length'] = data['preprocess'].apply(len)
# plt.figure(figsize=(10, 6))
# plt.hist(data['length'], bins=30, edgecolor='black', alpha=0.7)
# plt.title('Distribution of Sentence Lengths in combined_column')
# plt.xlabel('Sentence Length')
# plt.ylabel('Frequency')
# plt.show()

data.dropna(inplace=True)
train_data, test_data = train_test_split(data[['preprocess', 'result']], test_size=0.2, random_state=42)

##  토큰화
# tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
# tokenizer.save_pretrained("tokenizer")
tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
max_seq_len = 180

# def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

#     input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

#     for example, label in tqdm(zip(examples, labels), total=len(examples)):
#         input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)
#         padding_count = input_id.count(tokenizer.pad_token_id)
#         attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
#         token_type_id = [0] * max_seq_len

#         assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
#         assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
#         assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

#         input_ids.append(input_id)
#         attention_masks.append(attention_mask)
#         token_type_ids.append(token_type_id)
#         data_labels.append(label)

#     input_ids = np.array(input_ids, dtype=int)
#     attention_masks = np.array(attention_masks, dtype=int)
#     token_type_ids = np.array(token_type_ids, dtype=int)

#     data_labels = np.asarray(data_labels, dtype=np.int32)

#     return (input_ids, attention_masks, token_type_ids), data_labels

# train_X, train_y = convert_examples_to_features(train_data['preprocess'], train_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)
# test_X, test_y = convert_examples_to_features(test_data['preprocess'], test_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)

# directory = 'pickle'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# with open('pickle/train_data.pkl', 'wb') as f:
#     pickle.dump((train_X, train_y), f)

# with open('pickle/test_data.pkl', 'wb') as f:
#     pickle.dump((test_X, test_y), f)

with open('pickle/train_data.pkl', 'rb') as f:
    train_X, train_y = pickle.load(f)

with open('pickle/test_data.pkl', 'rb') as f:
    test_X, test_y = pickle.load(f)
print(train_X[:1])

input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

max_seq_len = 256
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        # BERT 모델 로드
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, training=False)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        prediction = self.classifier(x)
        return prediction

# 콜백 정의
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model/model_checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
    verbose=1
)

# 모델 컴파일 및 학습
model = TFBertForSequenceClassification("skt/kobert-base-v1")
for layer in model.layers:
    layer.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 학습
model.fit(train_X, train_y, epochs=10, batch_size=4, validation_split=0.2, verbose=1,
          callbacks=[checkpoint_callback, early_stopping_callback])
model.load_weights('model/model_checkpoint')
results = model.evaluate(test_X, test_y, batch_size=2)
print("test loss, test acc: ", results)