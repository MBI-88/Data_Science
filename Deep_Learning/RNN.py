#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.figsize']=(7,7)

pasajero=pd.read_csv('C:/Users\MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/international-airline-passengers.csv',sep=';')
pasajero.head()


# In[8]:


pasajero['Passengers_1']=pasajero['Passengers'].shift(-1)
pasajero.tail()


# In[9]:


pasajero=pasajero.drop(143)
pasajero_x=pasajero['Passengers'].astype(float).values
pasajero_y=pasajero['Passengers_1'].astype(float).values


# In[10]:


n_periodos=len(pasajero)
pct_test=0.2
n_train= int(n_periodos *(1-pct_test))
n_train


# # Estandarizando datos

# In[11]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
scaler.fit(pasajero_x.reshape(-1,1))
pasajero_x_std=scaler.transform(pasajero_x.reshape(-1,1))
pasajero_y_std=scaler.transform(pasajero_y.reshape(-1,1))

x_train=pasajero_x_std[:n_train]
x_test=pasajero_x_std[n_train:]

y_train=pasajero_y_std[:n_train]
y_test=pasajero_y_std[n_train:]

x_train=x_train.reshape(-1,1,1)
x_test=x_test.reshape(-1,1,1)
x_train.shape


# In[12]:


from keras import Sequential
from keras.layers import Dense,LSTM,GRU

modelo_lstm=Sequential()
modelo_lstm.add(GRU(10,input_shape=(1,1)))
modelo_lstm.add(Dense(1))
modelo_lstm.summary()


# In[15]:


modelo_lstm.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])
modelo_lstm.fit(x_train,y_train,epochs=100,batch_size=1,verbose=0);


# In[16]:


from sklearn.metrics import mean_squared_error

train_pred=scaler.inverse_transform(modelo_lstm.predict(x_train))
y_train_original=scaler.inverse_transform(y_train)
error_train=np.sqrt(mean_squared_error(y_train_original,train_pred))
error_train


# In[17]:


test_pred=scaler.inverse_transform(modelo_lstm.predict(x_test))
y_test_original=scaler.inverse_transform(y_test)
error_test=np.sqrt(mean_squared_error(y_test_original,test_pred))
error_test


# In[18]:


test_pred_plot=np.zeros(pasajero_y.shape)
test_pred_plot[-test_pred.shape[0]:]=test_pred[:,0]
test_pred_plot[:-test_pred.shape[0]]=np.nan

plt.plot(pasajero_y)
plt.plot(train_pred,label='prediccion train')
plt.plot(test_pred_plot,label='prediccion test')
plt.title('Numero de pasajeros internacionales')
plt.legend();


# In[ ]:




