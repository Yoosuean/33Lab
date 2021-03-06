#!/usr/bin/env python
# coding: utf-8

# 인터넷 영화 데이터 베이스(Internet Movie Database, IMDB)는 영화와 관련된 정보와 출연진 정보, 개봉 정보, 영화 후기, 평점에 이르기까지 매우 폭넓은 데이터가 저장된 자료.  
# 영화에 관해 남긴 2만 5000여 개의 영화 리뷰가 담겨 있으며, 해당 영화를 긍정적으로 평가했는지 혹은 부정적으로 평가했는지도 담겨 있다.  
# 로이터 뉴스 데이터와 마찬가지로 각 단어에 대한 전처리를 마친 상태로, 데이터셋에서 나타나는 빈도에 따라 번호가 정해지므로 빈도가 높은 데이터를 불러와 학습시킬 수 있다.  
# 데이터 전처리 과정은 로이터 뉴스 데이터와 거의 같지만 클래스가 긍정 또는 부정 두 가지뿐이라 원-핫 인코딩 과정은 없다.

# In[1]:


# -*- coding: utf-8 -*-
# 코드 내부에 한글을 사용가능 하게 해주는 부분입니다.

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

import numpy
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


# seed 값 설정
seed = 0
numpy.random.seed(seed)


# In[3]:


# 학습셋, 테스트셋 지정하기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)


# In[4]:


# 데이터 전처리
x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)


# 마지막에 model.summary() 함수를 넣으면 현재 설정된 모델의 구조를 한눈에 확인할 수 있다.  
#   
# 저번주에 "MNIST 손글씨 인식"을 다룰때는 Conv2D와 MaxPooling2를 사용했다.  
# 하지만, 2차원 배열을 가진 이미지와 다르게 지금 다루고 있는 데이터는 배열 형태로 이루어진 1차원이라는 차이가 있다.  
# <b>Conv1D는 Conv2D의 개념을 1차원으로 옮긴 것 ! !  컨볼루션 층이 1차원이고 이동하는 배열도 1차원</b>  
#   
#  MaxPooling1D 역시 마찬가지로 2차원 배열이 1차원으로 바뀌어 정해진 구역 안에서 가장 큰 값을 다음 층으로 넘기고 나머지는 버림.

# In[5]:


# 모델의 설정
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=4)) # 전체 크기가 1/4 로 줄어듬
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()


# In[6]:


# 모델의 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[7]:


# 모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=5, validation_data=(x_test, y_test))


# In[8]:


# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))


# In[9]:


# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습셋의 오차
y_loss = history.history['loss']


# In[10]:


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

케라스의 개발자인 구글의 프랑수아 콜렛이 운영하는 깃허브를 방문하여  
autoencoder, GAN, Deep dream, Neural style transfer 같은 예제를 직접 테스트해봐라~!~!