import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
# import tensorflow_datasets as tfds
from transformers import BertTokenizer, TFBertModel
import csv
import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

##  csv 불러오기
data = pd.read_csv('naverNews.csv')
data.columns = ['date','day','media','title','main']
data.drop_duplicates(subset=['title'], inplace=True)
data.drop_duplicates(subset=['main'], inplace=True)
data = data.sample(frac=1).reset_index(drop=True)
data = data.iloc[:200000]

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
  # try:
  #   resultBinary(date, dataDict, raiseList)
  # except KeyError:
  #   date = datePlus(date)
  #   try:
  #     resultBinary(date, dataDict, raiseList)
  #   except KeyError:
  #     date = datePlus(date)
  #     resultBinary(date, dataDict, raiseList)
data['result'] = raiseList
# filteredData = data[data['date'] == '2024.06.21']
# print(filteredData[['date', 'result']])

##  중복 처리
# 제목 중복 : 19027 -> 13990
# 본문 중복 : 19027 -> 14196
# 총합 중복 : 19027 -> 13386
data['titleDup'] = data.duplicated(subset=['title'], keep=False).astype(int)
data['mainDup'] = data.duplicated(subset=['main'], keep=False).astype(int)
data['duplicated'] = ((data['titleDup'] == 1) | (data['mainDup'] == 1)).astype(int)
# print(data[['title', 'duplicated']][50:60])
# print(len(data))
data.drop_duplicates(subset=['title'], inplace=True)
data.drop_duplicates(subset=['main'], inplace=True)
# print(len(data))
# total_duplicates = data['duplicated'].sum()
# print(f"총 중복된 갯수: {total_duplicates}")

##  null 값 처리
# print(data.isnull().sum(),"\n")
# print(data.isnull().values.any(),"\n")
data.dropna(inplace=True)
# print(data.isnull().sum(),"\n")
# print(data.isnull().values.any(),"\n")

##  요일을 숫자에서 글자로
def numToDay(num):
  if num == 0:
    return '월'
  elif num == 1:
    return '화'
  elif num == 2:
    return '수'
  elif num == 3:
    return '목'
  elif num == 4:
    return '금'
  elif num == 5:
    return '토'
  elif num == 6:
    return '일'
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
data['combined_column'] = data['title'].astype(str) + ' ' + data['preprocessedMain'].astype(str) + ' ' + data['dayWord'].astype(str) #+ ' ' + data['duplicated'].astype(str)
# randomNum = random.sample(range(19028), 3)
# for i in randomNum:
#   print("- "+data['combined_column'][i]+"\n")
data.reset_index(drop=True, inplace=True)
# print(data['combined_column'][3])

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
data = pd.read_csv('preprocessedCsv.csv')
data.columns = ['combined_column', 'result']

##    트레인, 테스트 셋 나누기
train_data, test_data = train_test_split(data[['combined_column', 'result']], test_size=0.2, random_state=42)
# print(len(train_data))
# print(train_data[:2])
# train_data['result'].value_counts().plot(kind = 'bar')

##  토큰화
# tokenizer = BertTokenizer.from_pretrained("klue/bert-base")   #bert-base-multilingual-cased
# tokenizer.save_pretrained("tokenizer")
tokenizer = BertTokenizer.from_pretrained("tokenizer")
# print('평균길이 : ',sum(map(len, train_data['combined_column']))/ len(train_data))
# print('최대길이 : ',max(len(script) for script in train_data['combined_column']))
# train_data['combined_column']의 길이 계산
# lengths = [len(script) for script in train_data['combined_column']]
# length_counts = pd.Series(lengths).value_counts().sort_index()

# plt.figure(figsize=(10, 6))
# plt.bar(length_counts.index, length_counts.values, color='skyblue')
# plt.xlabel('legth')
# plt.ylabel('data')
# plt.title('length per data')
# plt.show()
# print(tokenizer.encode("내 말 찰떡같이 알아듣는 AI 비서 꼼꼼한 대화분석으로 만든다 한국어 학습 위한 챗봇 구축에 기여 연구팀은 현재까지 200여 명이 참여한 WoZ 실험을 통해 총 2만 5천 건의 말차례가 담긴 데이터를 수집했고 삼성전자 빅스비와 사람의 대화에서 7천여 건의 말차례를 수집했다 이를 토대로 총 45개의 행위와 52개의 개체명을 추출 데이터를 태깅했다 이 태깅 화"))
# print(tokenizer.tokenize("내 말 찰떡같이 알아듣는 AI 비서 꼼꼼한 대화분석으로 만든다 한국어 학습 위한 챗봇 구축에 기여 연구팀은 현재까지 200여 명이 참여한 WoZ 실험을 통해 총 2만 5천 건의 말차례가 담긴 데이터를 수집했고 삼성전자 빅스비와 사람의 대화에서 7천여 건의 말차례를 수집했다 이를 토대로 총 45개의 행위와 52개의 개체명을 추출 데이터를 태깅했다 이 태깅 화"))
# print(tokenizer.encode("장중시황 코스피 외국인 순매도에 2660선 후퇴 삼성전자 SK하이닉스 LG에너지솔루션 현대차 삼성바이오로직스 삼성전자우 기아 셀트리온 KB금융 POSCO홀딩스 같은 시각 코스닥지수는 전일 대비 026 상승한 84059를 기록중이다 목"))
# print(tokenizer.tokenize("장중시황 코스피 외국인 순매도에 2660선 후퇴 삼성전자 SK하이닉스 LG에너지솔루션 현대차 삼성바이오로직스 삼성전자우 기아 셀트리온 KB금융 POSCO홀딩스 같은 시각 코스닥지수는 전일 대비 026 상승한 84059를 기록중이다 목"))
max_seq_len = 282
# encoded_result = tokenizer.encode("장중시황 코스피 외국인 순매도에 2660선 후퇴 삼성전자 SK하이닉스 LG에너지솔루션 현대차 삼성바이오로직스 삼성전자우 기아 셀트리온 KB금융 POSCO홀딩스 같은 시각 코스닥지수는 전일 대비 026 상승한 84059를 기록중이다 목", truncation=True, max_length=max_seq_len, padding='max_length')
# print(encoded_result)

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

train_X, train_y = convert_examples_to_features(train_data['combined_column'], train_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)
test_X, test_y = convert_examples_to_features(test_data['combined_column'], test_data['result'], max_seq_len=max_seq_len, tokenizer=tokenizer)

input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
max_seq_len = 282
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction
    
    # def save_model(self, save_path):
    #     self.save_weights(save_path)

    # def load_model(self, load_path):
    #     self.load_weights(load_path)
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
model = TFBertForSequenceClassification("klue/bert-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
model.fit(train_X, train_y, epochs=20, batch_size=16, validation_split=0.2, verbose=1, callbacks=[checkpoint_callback, early_stopping_callback]) # ,callbacks=[early_stopping, checkpoint]
# model.save_model('model_weights')
# model.load_model('model_weights')
model.load_weights('model/model_checkpoint')
results = model.evaluate(test_X, test_y, batch_size=16)
print("test loss, test acc: ", results)
