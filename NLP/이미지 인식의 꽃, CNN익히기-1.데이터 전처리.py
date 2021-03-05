#!/usr/bin/env python
# coding: utf-8

# #### mnist 데이터 셋 불러오기 (70000개의 글자 이미지에 각각 0부터 9까지 이름표를 붙인 데이터 셋)

# In[16]:


from keras.datasets import mnist
from keras.utils import np_utils

import numpy
import sys
import tensorflow as tf


# In[33]:


#seed 값 설정
seed=0
numpy.random.seed(seed)
tf.random.set_seed(3)


# ##### 불러온 이미지 데이터 X, 이 이미지에 0~9까지 붙인 이름표 Y_class
# 학습에 사용될 부분 : X_train, Y_class_train  
# 테스트에 사용될 부분 : X_test, Y_class_test

# In[34]:


#MNIST 데이터셋 불러오기
(X_train, Y_class_train), (X_test, Y_class_test)=mnist.load_data()


# In[35]:


#케라스의 MNIST 데이터는 총 70,000개의 이미지 중 60,000개를 학습용으로, 10,000개를 테스트용으로 미리 구분해 놓고 있음
print("학습셋 이미지 수: %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))


# In[36]:


#X_train[0]을 이용해 첫번째 이미지 출력, cmap='Greys' 옵션을 지정해 흑백으로 출력
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='Greys')
plt.show()


# In[37]:


# 코드로 확인
for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')


# In[41]:


# 차원 변환 과정
# 2차원배열을 1차원배열로 전환할 때 사용하는 함수 reshape() -> reshape(총 샘플 수, 1차원 속성의 수)
X_train = X_train.reshape(X_train.shape[0], 784)

# 케라스 데이터는 0~1 사이의 값으로 변환한 다음 구동할 때 최적의 성능을 보인다.
# 그렇기에 0~255 사이 값으로 이루어진 값을 바꿔야 함 ! ! 각 값을 255로 나눈다 ! ! !
# 이렇게 데이터 폭이 클 때 적절한 값으로 분산의 정도를 바꾸는 과정을 데이터 정규화(normalization)이라고 함.
X_train = X_train.astype('float64')
# 정규화를 위해 255로 나누어 주려면 먼저 이 값을 실수형으로 바꿔야함.
X_train = X_train / 255


# In[39]:


# X_test에도 마찬가지로 이 작업을 적용함.
X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255


# In[42]:


# 클래스 값 확인
# 실제로 이 숫자의 레이블이 어떤지를 불러오는 것 = 5 ! !
print("class : %d " % (Y_class_train[0]))
  


# In[45]:


# 바이너리화 과정
# 딥러닝의 분류 문제를 해결하려면 원-핫 인코딩 방식을 적용해야 함.
# 즉, 0~9까지의 정수형 값을 갖는 현재 형태에서 0 또는 1로만 이루어진 벡터로 값을 수정해야 함.
# 예를 들어 class가 '3'이라면, [3]을 [0,0,1,0,0,0,0,0,0,0]로 바꿔주어야 함.
# 이를 가능하게 해 주는 함수 => np_utils.to_categrical(클래스, 클래스의 개수)
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)
 
print(Y_train[0])


# ## 놀랍게도 아직 딥러닝 한 거 아님 ^___^ 
# 데이터 전처리임..
# 챕터 2에서 봅시다 . . 더보기
