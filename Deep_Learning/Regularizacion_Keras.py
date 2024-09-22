#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(5,5)
from keras.datasets  import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[2]:


x_train.shape


# In[3]:


x_train[0].shape


# In[4]:


from ipywidgets import interact,IntSlider

@interact(i=IntSlider(min=0,max=50,step=1,value=1))
def dibujar_numero(i):
    plt.imshow(x_train[i],cmap='gray')
    plt.title('Numero {}'.format(y_train[i]))


# In[5]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
import numpy as np

np.unique(y_train)


# In[6]:


# Como las redes neuroneles esperan densas esperan siempre un array como entrada , hay que aplanar a un array de 1 dimension
x_train_plano=x_train.reshape(x_train.shape[0],28*28)
x_test_plano=x_test.reshape(x_test.shape[0],28*28)
x_test_plano[0]


# In[7]:


from keras.utils.np_utils import to_categorical # Esta sirve para codificar cada valor del array

y_train_one_hot=to_categorical(y_train)
y_test_one_hot=to_categorical(y_test)


# In[8]:


modelo=Sequential()
modelo.add(Dense(50,activation='relu',input_shape=(784,)))
modelo.add(Dense(250,activation='relu'))
modelo.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
modelo.summary()


# In[9]:


modelo.fit(x_train_plano,y_train_one_hot,epochs=50,batch_size=1000,verbose=0);


# In[10]:


resultados={}
modelo.metrics_names


# In[11]:


modelo.metrics_tensors


# In[12]:


evaluacion_train= modelo.evaluate(x_train_plano,y_train_one_hot)
evaluacion_train


# In[13]:


evaluacion_test=modelo.evaluate(x_test_plano,y_test_one_hot)
evaluacion_test


# In[14]:


resultados['sin_regularizacion']=[evaluacion_train[1],evaluacion_test[1]]


# # Regularizando L1 , O L2

# In[15]:


from keras import regularizers

modelo_l2=Sequential()
modelo_l2.add(Dense(50,activation='relu',input_shape=(784,)))
modelo_l2.add(Dense(250,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
modelo_l2.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo_l2.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_l2.summary()


# In[16]:


modelo_l2.fit(x_train_plano,y_train_one_hot,verbose=0,epochs=50,batch_size=1000);


# In[17]:


acc_train=modelo_l2.evaluate(x_train_plano,y_train_one_hot)[1]
acc_train


# In[18]:


acc_test=modelo_l2.evaluate(x_test_plano,y_test_one_hot)[1]
acc_test


# In[19]:


resultados['regularizacion_l2']=[acc_train,acc_test]


# In[20]:


# Regularizacion L1
modelo_l1=Sequential()
modelo_l1.add(Dense(50,activation='relu',input_shape=(784,)))
modelo_l1.add(Dense(250,activation='relu',kernel_regularizer=regularizers.l1(0.01)))
modelo_l1.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo_l1.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_l1.fit(x_train_plano,y_train_one_hot,verbose=0,epochs=50,batch_size=1000)
modelo_l1.summary()


# In[21]:


acc_train=modelo_l1.evaluate(x_train_plano,y_train_one_hot)[1]
acc_test=modelo_l1.evaluate(x_test_plano,y_test_one_hot)[1]
resultados['regularizacion_l1']=[acc_train,acc_test]


# In[22]:


print(acc_train,acc_test)


# # Dropout

# In[23]:


# Dropout ignora un porciento p se las neuronas en cada iteracion , solo a la capa a la que se le ponga a continuacion
modelo_dropout=Sequential()
modelo_dropout.add(Dense(50,activation='relu',input_shape=(784,)))
modelo_dropout.add(Dense(250,activation='relu'))
modelo_dropout.add(Dropout(0.2))
modelo_dropout.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo_dropout.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_dropout.summary()


# In[24]:


modelo_dropout.fit(x_train_plano,y_train_one_hot,verbose=0,epochs=50,batch_size=1000)
acc_train=modelo_dropout.evaluate(x_train_plano,y_train_one_hot)[1]
acc_test=modelo_dropout.evaluate(x_test_plano,y_test_one_hot)[1]
print(acc_train,acc_test)


# In[25]:


resultados['regularizacion_dropout']= [acc_train,acc_test]


# # Normalizacion en bloques 

# In[26]:


from keras.layers import BatchNormalization # Funciona solo sobre la capa anterior

modelo_bnorm=Sequential()
modelo_bnorm.add(Dense(50,activation='relu',input_shape=(784,)))
modelo_bnorm.add(Dense(250,activation='relu'))
modelo_bnorm.add(BatchNormalization())
modelo_bnorm.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo_bnorm.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_bnorm.summary()


# In[27]:


modelo_bnorm.fit(x_train_plano,y_train_one_hot,verbose=0,epochs=50,batch_size=1000)
acc_train=modelo_bnorm.evaluate(x_train_plano,y_train_one_hot)[1]
acc_test=modelo_bnorm.evaluate(x_test_plano,y_test_one_hot)[1]
resultados['bat_normalizacion']=[acc_train,acc_test]
print(acc_train,acc_test)


# # Bacth + Dropout

# In[28]:


modelo_bnorm_drop=Sequential()
modelo_bnorm_drop.add(Dense(50,activation='relu',input_shape=(784,)))
modelo_bnorm_drop.add(Dense(250,activation='relu'))
modelo_bnorm_drop.add(BatchNormalization())
modelo_bnorm_drop.add(Dropout(0.2))
modelo_bnorm_drop.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo_bnorm_drop.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_bnorm_drop.fit(x_train_plano,y_train_one_hot,verbose=0,epochs=50,batch_size=1000)
acc_train=modelo_bnorm_drop.evaluate(x_train_plano,y_train_one_hot)[1]
acc_test=modelo_bnorm_drop.evaluate(x_test_plano,y_test_one_hot)[1]
resultados['batch_normalizacion + dropout']=[acc_train,acc_test]
print(acc_train,acc_test)


# In[30]:


import pandas as pd

resultados_df=pd.DataFrame(resultados).T
resultados_df.columns=['acc_train','acc_test']
resultados_df['pct_diff']=1-(resultados_df.acc_test/resultados_df.acc_train)
resultados_df.sort_values(by='pct_diff')

