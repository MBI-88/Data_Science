#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[12,12]

vino=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/vino.csv')
vino.shape


# In[2]:


vino.head()


# In[4]:


vino.dtypes


# In[5]:


variable_dep='puntuacion'
variable_inde=vino.drop(variable_dep,axis=1).columns

vino_X=vino[variable_inde]
vino_y=vino[variable_dep]


# In[7]:


from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

estimador_rf=RandomForestRegressor()
estimador_ab=AdaBoostRegressor()
np.linspace(0.001,1.,10)


# In[9]:


train_size,train_scores,test_scores=learning_curve(
    estimador_ab,vino_X,vino_y,cv=5,n_jobs=-1,scoring='neg_mean_squared_error',train_sizes=np.linspace(0.001,1.,10))


# In[10]:


train_scores


# In[12]:


train_size


# In[13]:


train_scores_mean=np.mean(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)

plt.plot(train_size,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamineto')
plt.plot(train_size,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curva de Apendizaje: ADABoost')
plt.xlabel('Numero de  muestras de entrenamiento')
plt.ylabel('Error Cuadratico Medio (MSE)')
plt.legend();


# #  Usando el segundo  modelo

# In[37]:


train_size,train_scores,test_scores=learning_curve(
    estimador_rf,vino_X,vino_y,cv=5,n_jobs=-1,train_sizes=np.linspace(0.001,1.,10),scoring='neg_mean_squared_error')
train_scores_mean=np.mean(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)

plt.plot(train_size,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamineto')
plt.plot(train_size,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curva de Apendizaje: Bosques Aleatorios')
plt.xlabel('Numero de  muestras de entrenamiento')
plt.ylabel('Error Cuadratico Medio (MSE)')
plt.legend();


# In[38]:


from sklearn.model_selection import  validation_curve # Devuelve solo los  puntos de train y los test

n_arboles=[2,10,20,50,100,150,200]
train_scores,test_scores=validation_curve(
    estimador_rf,vino_X,vino_y,param_name='n_estimators',param_range=n_arboles,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)


# In[39]:


train_scores_mean=np.mean(train_scores,axis=1)
train_scores_std=np.std(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)
test_scores_std=np.std(test_scores,axis=1)

plt.plot(n_arboles,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamiento')
plt.plot(n_arboles,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curvas de Validacion: Bosques Aleatorios / Numeros de Arboles')
plt.xlabel('Numero de estimadores')
plt.ylabel('Error Cuadratico Medio (MSE)')
plt.legend();


# In[40]:


n_esimadores=[10,20,50,100,200,350,500]
train_scores,test_scores=validation_curve(
    estimador_ab,vino_X,vino_y,param_name='n_estimators',param_range=n_esimadores,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)

train_scores_mean=np.mean(train_scores,axis=1)
train_scores_std=np.std(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)
test_scores_std=np.std(test_scores,axis=1)

plt.plot(n_arboles,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamiento')
plt.plot(n_arboles,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curvas de Validacion: AdaBoost / Numeros de Arboles')
plt.xlabel('Numero de estimadores')
plt.ylabel('Error Cuadratico Medio (MSE)')
plt.legend();


# In[41]:


train_size,train_scores,test_scores=learning_curve(
    RandomForestRegressor(n_estimators=100),vino_X,vino_y,cv=5,scoring='neg_mean_squared_error',n_jobs=-1,
    train_sizes=np.linspace(0.01,1.,10))
train_scores_mean=np.mean(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)

plt.plot(train_size,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamiento')
plt.plot(train_size,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curva de Aprendizaje: Bosques Aleatorios')
plt.xlabel('Numero de muestras')
plt.ylabel('Error Cuadratico Medio (MSE)')
plt.legend();

