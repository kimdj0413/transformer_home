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
with open('korean_stopwords.txt', 'r',encoding='utf-8') as f:
  list_file = f.readlines() 
stopwords = [word.strip() for word in list_file]

okt = Okt()
X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)
