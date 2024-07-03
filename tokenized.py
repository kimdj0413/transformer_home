import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

data = pd.read_csv('preprocessedCsv.csv')
data.columns = ['combined_column', 'result']
train_data, test_data = train_test_split(data[['combined_column', 'result']], test_size=0.2, random_state=42)
# with open('korean_stopwords.txt', 'r',encoding='utf-8') as f:
#   list_file = f.readlines() 
# stopwords = [word.strip() for word in list_file]

okt = Okt()
X_train = []
X_test = []
# for sentence in tqdm(train_data['combined_column']):
#     tokenized_sentence = okt.morphs(sentence, stem=True)
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
#     X_train.append(stopwords_removed_sentence)
# with open('X_train.pkl', 'wb') as f:
#     pickle.dump(X_train, f)
# for sentence in tqdm(test_data['combined_column']):
#     tokenized_sentence = okt.morphs(sentence, stem=True)
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
#     X_test.append(stopwords_removed_sentence)
# with open('X_test.pkl', 'wb') as f:
#     pickle.dump(X_test, f)

with open('X_train_okt.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('X_test_okt.pkl', 'rb') as f:
    X_test = pickle.load(f)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# print(tokenizer.word_index)

threshold = 4
total_cnt = len(tokenizer.word_index)
rare_cnt = 0 
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 1
print(vocab_size)
tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['result'])
y_test = np.array(test_data['result'])

np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
y_train= np.load('y_train.npy')
y_test= np.load('y_test.npy')
# print('기사의 최대 길이 :',max(len(article) for article in X_train))
# print('기사의 평균 길이 :',sum(map(len, X_train))/len(X_train))
# plt.hist([len(review) for review in X_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

# def below_threshold_len(max_len, nested_list):
#   count = 0
#   for sentence in nested_list:
#     if(len(sentence) <= max_len):
#         count = count + 1
#   print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 65
# below_threshold_len(max_len, X_train)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
print(X_train[:1])

# with open('X_train.pkl', 'wb') as f:
#     pickle.dump(X_train, f)
# with open('X_test.pkl', 'wb') as f:
#     pickle.dump(X_test, f)
# with open('X_train.pkl', 'rb') as f:
#     X_train = pickle.load(f)
# with open('X_test.pkl', 'rb') as f:
#     X_test = pickle.load(f)
# print(X_train[:1])
# print(X_test[:1])
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


embedding_dim = 100
hidden_units = 128
vocab_size = 33646

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))