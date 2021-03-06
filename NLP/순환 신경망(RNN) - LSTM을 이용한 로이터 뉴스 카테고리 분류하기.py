#!/usr/bin/env python
# coding: utf-8

# 로이터 뉴스 데이터셋 불러오기 from keras.datasets import reuters  
# 로이터 뉴스 데이터는, 총 11,258개의 뉴스 기사가 46개의 카테고리로 나누어진 대용량 텍스트 데이터

# In[1]:


# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

# 로이터 뉴스 데이터셋 불러오기
from keras.datasets import reuters

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessi ng import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt


# In[5]:


# seed 값 설정
seed = 0
numpy.random.seed(seed)


# reuter.lead_data() 함수를 이용해 기사를 불러온 후,  
# test_split 인자를 통해 20%를 테스트셋으로 사용했다.

# In[6]:


# 불러온 데이터를 학습셋, 테스트셋으로 나누기
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)


# np.max() 함수로 y_train의 종류를 구하니 46개의 카테고리로 구분되어 있음을 알 수 있다.  
# (0부터 세기 때문에 1을 더해서 출력)  
# 이 중 8982개는 학습용으로, 2246개는 테스트용으로 준비되어 있다.  
#   
# ###### print(X_train[0])의 결과가 단어가 아닌 숫자가 나오는 이유는?  
# 딥러닝은 단어를 그대로 사용하지 않고 숫자로 변환한 다음에 학습한다.  
# 여기서는 데이터 안에서 해당 단어가 몇 번이나 나타나는지 세어 빈도에 따라 번호를 붙였다.
# 예를 들어, 3이라고 하면 세 번째로 빈도가 높은 단어라는 뜻.  
# 이러한 작업을 위해서 tokenizer() 같은 함수를 사용하는데, 케라스는 이 작업을 이미 마친 데이터를 불러올 수 있다.  
#   
# 기사 안의 단어 중에는 거의 사용되지 않는 것들도 있는데, 모든 단어를 다 사용하는 것은 비효율적이므로 빈도가 높은 단어만 불러와 사용  
# <b>이 때 사용하는 인자가 위의 테스트셋과 학습셋으로 나눌 때 함께 적용했던 num_words=1000 </b>  
# 빈도가 1~1000에 해당하는 단어만 선택해서 불러오는 것.
# 
# *각 기사의 단어 수가 제각각 다르므로 단어의 숫자를 맟춰야 함. 이때는 데이터 전처리 함수 sequence를 이용
# 

# In[7]:


# 데이터 확인하기
category = numpy.max(Y_train) + 1
print(category, '카테고리')
print(len(X_train), '학습용 뉴스 기사')
print(len(X_test), '테스트용 뉴스 기사')
print(X_train[0])


# maxlen=100은 단어 수를 100개로 맞추라는 뜻.  
# 만일 입력된 기사의 단어 수가 100보다 크면 100개째 단어만 선택하고 나머지는 버린다.  
# 100에서 모자랄 때는 모자라는 부분을 모두 0으로 채운다.  
#   
# 그 후, y 데이터에 원-핫 인코딩 처리를 하여 데이터 전처리 과정을 마친다.

# In[8]:


# 데이터 전처리
x_train = sequence.pad_sequences(X_train, maxlen=100)
x_test = sequence.pad_sequences(X_test, maxlen=100)
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)


# Embedding 층은 데이터 전처리 과정을 통해 입력된 값을 받아 다음 층이 알아들 수 있는 형태로 변환하는 역할  
# Embedding('불러온 단어의 총 개수', '기사당 단어 수') 형식으로 사용하며, 모델 설정 부분의 맨 처음에 있어야 한다.  
#   
# LSTM은 RNN에서 기억 값에 대한 가중치를 제어한다. LSTM(기사당 단어 수, 기타 옵션)의 형태로 적용  
# LSTM의 활성화 함수로는 Tanh를 사용.

# In[9]:


# 모델의 설정
model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))


# In[10]:


# 모델의 컴파일
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])


# In[11]:


# 모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))


# In[12]:


# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))


# In[13]:


# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']


# In[14]:


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


# 테스트 오차가 상승하기전까지의 학습이 과적합 직전의 최적 학습 시간.
