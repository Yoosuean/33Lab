#!/usr/bin/env python
# coding: utf-8

# ##### 컨볼루션 신경망은 입력된 이미지에서 다시 한번 특징을 추출하기 위해 마스크(필터, 원도 또는 커널이라고도 함)를 도입하는 기법이다.  

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf


# In[4]:


# seed 값 설정
seed = 0
numpy.random.seed(seed)

# 데이터 불러오기

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)


# In[5]:


# 컨볼루션 신경망의 설정
model = Sequential()
#컨볼루션 층 추가 : 32개의 필터, 3x3, 맨 처음 층이 입력되는 값/이미지가 색상이면 3 흑백이면 1, 활성화 함수 렐루~!~! 
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
#컨볼루션 층 추가 : 64개의 필터 3x3 활성화 함수 렐루~~
model.add(Conv2D(64, (3, 3), activation='relu'))
# 맥스풀링 : 풀링 기업 중 가장 많이 사용되는 방법으로 정해진 구역 안에서 가장 큰 값만 다음 층으로 넘기고 나머지는 버린다.
# 여기서 pool_size는 풀링 창의 크기를 정하는 것으로, 2로 정하면 전체 크기가 절반으로 줄어듬.
model.add(MaxPooling2D(pool_size=2))
# 과적합을 막기 위한 "드롭아웃" : 은닉층에 배치된 노드 중 일부를 임의로 꺼줌. 밑에서는 25% 노드를 랜덤하게 꺼준다.
model.add(Dropout(0.25))
# 컨볼루션 층이나 맥스 풀링은 주어진 이미지를 2차원 배열인 채로 다룬다. 이를 1차원으로 바꿔주는 함수 Flatten()
model.add(Flatten())
model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[6]:


# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


# In[7]:


# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])


# In[8]:


# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


# ## 아까보다 정확도 업업 !!!!!!!!!!

# In[9]:


# 테스트 셋의 오차
y_vloss = history.history['val_loss']


# In[10]:


# 학습셋의 오차
y_loss = history.history['loss']


# In[11]:


# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

