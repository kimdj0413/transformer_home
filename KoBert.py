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

##  csv 불러오기
data = pd.read_csv('naverNews.csv')
data.columns = ['date','day','media','title','main']
data.drop_duplicates(subset=['title'], inplace=True)
data.drop_duplicates(subset=['main'], inplace=True)
data.dropna(inplace=True)
data = data.sample(frac=1).reset_index(drop=True)
data = data.iloc[:150000]

##  등락 추가하기
def datePlus(date):
  date = (datetime.datetime.strptime(date, "%Y.%m.%d")+ datetime.timedelta(days=1)).strftime("%Y.%m.%d")
  return date
def resultBinary(date, dataDict, raiseList):
  if '-' in dataDict[date]:
    raiseList.append(0)
  else:
    raiseList.append(1)
  return raiseList
dataDict = {}
with open('삼성전자주가.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        key = row[0]
        keyYear = key[:4]
        keyMonth = key[4:6]
        keyDay = key[6:]
        value = row[2]
        key = f"{keyYear}.{keyMonth}.{keyDay}"
        dataDict[key] = value

raiseList = []
for date in data['date']:
  while True:
    try:
      resultBinary(date, dataDict, raiseList)
      break
    except KeyError:
      date = datePlus(date)  
data['result'] = raiseList

##  중복 처리
data['titleDup'] = data.duplicated(subset=['title'], keep=False).astype(int)
data['mainDup'] = data.duplicated(subset=['main'], keep=False).astype(int)
data['duplicated'] = ((data['titleDup'] == 1) | (data['mainDup'] == 1)).astype(int)
data.drop_duplicates(subset=['title'], inplace=True)
data.drop_duplicates(subset=['main'], inplace=True)

##  null 값 처리
data.dropna(inplace=True)

##  요일을 숫자에서 글자로
def numToDay(num):
  if num == 0:
    return '월요일'
  elif num == 1:
    return '화요일'
  elif num == 2:
    return '수요일'
  elif num == 3:
    return '목요일'
  elif num == 4:
    return '금요일'
  elif num == 5:
    return '토요일'
  elif num == 6:
    return '일요일'
data['dayWord'] = data['day'].apply(numToDay)

##  본문 내용 정규표현식
def regulationText(text):
    if not isinstance(text, str):
      text = str(text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
data['preprocessedMain'] = data['main'].apply(regulationText)

##  제목, 본문, 요일 합치기
data['combined_column'] = data['title'].astype(str)+ ' ' + data['preprocessedMain'].astype(str)  +' '+ data['dayWord'].astype(str) #+ ' ' + data['duplicated'].astype(str)
data.reset_index(drop=True, inplace=True)

##  result의 0과 1 비율 맞추기
countDown = data[data['result'] == 0].shape[0]
countUp = data[data['result'] == 1].shape[0]
countMin = min(countDown, countUp)
dataDown = data[data['result'] == 0].sample(n=countMin, random_state=42)
dataUp = data[data['result'] == 1].sample(n=countMin, random_state=42)
balancedData = pd.concat([dataDown, dataUp])

##  전처리 완료된 csv 새로 저장 후 불러오기
selectedData = balancedData[['combined_column', 'result']]
selectedData.to_csv('preprocessedCsv.csv', index=False)
"""
data = pd.read_csv('preprocessedCsv.csv')
data.columns = ['combined_column', 'result']
data['length'] = data['combined_column'].apply(len)

# # 그래프 설정
# plt.figure(figsize=(10, 6))

# # 히스토그램 그리기
# plt.hist(data['length'], bins=30, edgecolor='black', alpha=0.7)
# plt.title('Distribution of Sentence Lengths in combined_column')
# plt.xlabel('Sentence Length')
# plt.ylabel('Frequency')
# plt.show()

##    트레인, 테스트 셋 나누기
train_data, test_data = train_test_split(data[['combined_column', 'result']], test_size=0.2, random_state=42)

##  토큰화
# tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
# tokenizer.save_pretrained("tokenizer")
tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
max_seq_len = 256

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
        token_type_id = [0] * max_seq_len

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels

# train_X, train_y = convert_examples_to_features(train_data['combined_column'], train_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)
# test_X, test_y = convert_examples_to_features(test_data['combined_column'], test_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)

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

model = TFBertModel.from_pretrained("skt/kobert-base-v1", from_pt=True)
max_seq_len = 256
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        # BERT 모델 로드
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        
        # BERT 모델의 일부 층 고정
        for layer in self.bert.layers[:8]:  # 상위 8개 층만 학습하지 않도록 고정
            layer.trainable = False
        
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs.last_hidden_state[:, 0, :]  # BERT의 [CLS] 토큰
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
    patience=3,  # early stopping patience 감소
    mode='min',
    verbose=1
)

# 모델 컴파일 및 학습
model = TFBertForSequenceClassification("skt/kobert-base-v1")
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)  # 학습률 조정
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 학습
model.fit(train_X, train_y, epochs=10, batch_size=16, validation_split=0.2, verbose=1,
          callbacks=[checkpoint_callback, early_stopping_callback])
model.load_weights('model/model_checkpoint')
results = model.evaluate(test_X, test_y, batch_size=2)
print("test loss, test acc: ", results)

# def scheduler(epoch, lr):
#     if epoch < 3:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

# lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

##  전처리 셔플X, 10000개 - val_loss: 0.2909 - val_accuracy: 0.8789
"""