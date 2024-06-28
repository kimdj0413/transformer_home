import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel
import csv
import datetime
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

##  csv 불러오기
data = pd.read_csv('/content/drive/MyDrive/백업/bertmodel/ValNews.csv')
data.columns = ['date','day','media','title','main','merge']

##  모델 불러오기
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
model = TFBertForSequenceClassification("klue/bert-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
model.load_weights('/content/drive/MyDrive/백업/bertmodel/model/model_checkpoint')

##  검증
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/백업/bertmodel/tokenizer")

def financial_predict(sentence):
  input_id = tokenizer.encode(sentence, max_length=max_seq_len, pad_to_max_length=True)

  padding_count = input_id.count(tokenizer.pad_token_id)
  attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
  token_type_id = [0] * max_seq_len

  input_ids = np.array([input_id])
  attention_masks = np.array([attention_mask])
  token_type_ids = np.array([token_type_id])

  encoded_input = [input_ids, attention_masks, token_type_ids]
  score = model.predict(encoded_input)[0][0]

  return score

dateList = data['date'].unique()
startNum=0
endNum=0
date_counts = data['date'].value_counts()
scoreList=[]
countList=[]

cnt=0
# date=['2013.10.22','2013.10.23''2013.10.24']
date = '2014.04.07'
for index, row in data.iterrows():
    if row['date'] == date:
        scoreList.append(financial_predict(row['merge']))
        cnt+=1
        if cnt == 30:
            break
  # for date in dateList:
  #     subList = []
  #     count = date_counts.get(date, 0)
  #     # print(f"{date}: {count}개")
  #     countList.append(count)
  #     endNum=count
  #     for i in range(startNum,startNum+10):
  #         score = financial_predict(data['merge'][i])
  #         subList.append(score)
  #     scoreList.append(subList)
  #     startNum=endNum

for i in range(0,len(scoreList)):
  positiveScore = 0
  negativeScore = 0
  for score in scoreList:
    if score > 0.7:
      positiveScore += 1
    else:
      negativeScore += 1
  # date = dateList[i]
  average = sum(scoreList)/len(scoreList) if len(scoreList) else 0
  sumPosNeg = positiveScore + negativeScore
  positiveRate = positiveScore / sumPosNeg if sumPosNeg > 0 else 0
  negativeRate = negativeScore / sumPosNeg if sumPosNeg > 0 else 0
if(average > 0.7):
  print(f"날짜 : {date}, 평균 확률: {(average*100):.2f}%, 상승장입니다. 상승 : {(positiveRate*100):.2f}% ({positiveScore}개), 하락 : {(negativeRate*100):.2f}% ({negativeScore}개)")
else:
  print(f"날짜 : {date}, 평균 확률: {((1-average)*100):.2f}%, 하락장입니다. 상승 : {(positiveRate*100):.2f}% ({positiveScore}개), 하락 : {(negativeRate*100):.2f}% ({negativeScore}개)")

