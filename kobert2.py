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
with open('X_train.pkl', 'rb') as f:
    train_X = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    test_X = pickle.load(f)
train_y= np.load('y_train.npy')
test_y= np.load('y_test.npy')

input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

model = TFBertModel.from_pretrained("skt/kobert-base-v1", from_pt=True)
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
