#!/usr/bin/env python
# coding: utf-8

# In[19]:


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping
#학습의 자동 중단 함수

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf


# In[38]:


# seed 값 설정
seed = 0
numpy.random.seed(seed)


# In[26]:


# MNIST 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)


# ## 딥러닝 실행을 위한 프레임 설정!
# 총 784개의 속성이 있고 10개의 클래스가 있다.
# 입력 값은 784개, 은닉층은 512개 그리고 출력이 10개인 모델
# 활성화함수로 은닉층에서는 relu를, 출력층에서는 softmax를 사용한다.

# In[27]:


# 모델 프레임 설정
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 그리고 딥러닝 실행 환경을 위해 오차함수로 categorical_crossentropy,  
# 최적화 함수로 adam을 사용 ! !

# In[28]:


# 모델 실행 환경 설정
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 모델의 실행에 앞서 모델의 성과를 저장하고  
# 모델의 최적화 단계에서 학습을 자동 중단하게끔 설정

# In[29]:


# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
# 10회 이상 모델의 성과 향상이 없으면 자동으로 학습을 중단한다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


# In[30]:


# 모델의 실행
# 샘플 200개를 모두 30번 실행하게끔 설정. 그리고 테스트셋으로 최종 모델의 성과를 측정하여 그 값을 출력.
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])


# In[31]:


# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))


# 22번째 실행에서 멈춘 것을 확인할 수 있다. 베스트 모델은 12번째 에포크이며, 이 테스트셋에 대한 정확도는 98.39%이다.

# In[32]:


# 테스트 셋의 오차
y_vloss = history.history['val_loss']


# In[33]:


# 학습셋의 오차
# 학습셋의 정확도 대신 오차를 그래프로 표현. 학습셋의 오차 = 1 - 학습셋의 정확도
y_loss = history.history['loss']


# In[37]:


# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시

plt.legend(loc='upper right')
# plt.axis([0, 20, 0, 0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# #### 학습셋의 오차와 테스트셋의 오차를 비교한 이유는?
# 학습셋의 정확도는 1.00에 가깝고 테스트셋의 오차는 0.00에 가까우므로 두 개를 함께 비교하기가 어렵기 때문이다.

# ## 위의 98.39%의 정확도를 보인 딥러닝 프레임은 하나의 은닉층을 둔 아주 단순한 모델!

# ## 컨볼루션 신경망으로 넘어가보자 ! !
