#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['font.size']=10


# In[2]:


vehiculos= pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/data_frame_estandarizado.csv',
                          usecols=['consumo','co2','cilindros','desplazamiento'])
litros_port_galon= 3.78541
vehiculos['consumo_litros_milla']=litros_port_galon/vehiculos.consumo
vehiculos.shape


# In[3]:


vehiculos.head()


# In[4]:


porciento_entrenamiento= 0.8
vehiculos_entrenamiento= vehiculos.sample(frac=porciento_entrenamiento)
vehiculos_test=vehiculos[~vehiculos.index.isin(vehiculos_entrenamiento.index)]
print(vehiculos_entrenamiento.shape)
print(vehiculos_test.shape)


# In[7]:


variable_independiente=['desplazamiento','cilindros','consumo','consumo_litros_milla']
variable_dependiente='co2'
X=np.array(vehiculos_entrenamiento[variable_independiente],dtype=float)#Conviete a matriz se numpy
Y=np.array(vehiculos_entrenamiento[variable_dependiente],dtype=float)#Conviete a matriz se numpy
X_T=X.T#Traspuesta de X
X


# In[8]:


Y


# In[9]:


betas=np.linalg.inv(X_T@X)@X_T@Y
betas


# In[10]:


Y.mean()


# In[13]:


alfa=Y.mean()-np.dot(betas,vehiculos_entrenamiento[variable_independiente].mean())
alfa=np.array(alfa,dtype=float)
alfa


# In[14]:


def predecir(r):#Funcion de prediccion de Co2
    return alfa+np.dot(betas,r.values)

vehiculos_entrenamiento['co2_predicho']=vehiculos_entrenamiento[variable_independiente].apply(predecir,axis=1)
vehiculos_test['co2_predicho']=vehiculos_test[variable_independiente].apply(predecir,axis=1)
vehiculos_entrenamiento[['co2','co2_predicho']].head()#Visualizar la prediccion


# In[15]:


modelo_formula='y~{alfa:.3f}+{betas_1:.2f}*desplazamiento+{betas_2:.2f}*cilindros+{betas_3:.3f}*consumo_litro'.format(
    alfa=alfa, betas_1=betas[0], betas_2=betas[1], betas_3=betas[2])
modelo_formula


# In[16]:


plt.scatter(vehiculos_test.consumo_litros_milla,vehiculos_test.co2,alpha=0.5,label='real')
plt.text(0.1, 850, modelo_formula)
plt.plot(vehiculos_test.consumo_litros_milla,vehiculos_test.co2_predicho,c='black',label='prediccion')
plt.xlabel('Consumo combustible (litros/milla)')
plt.ylabel('Emision CO2 (gramos/milla)')
plt.legend();


# In[17]:


def error_cuadratico_medio(y,y_pred):
    return np.sum((y-y_pred)**2/len(y))
error_entrenamiento=error_cuadratico_medio(vehiculos_entrenamiento.co2,vehiculos_entrenamiento.co2_predicho)
error_entrenamiento


# In[18]:


error_test=error_cuadratico_medio(vehiculos_test.co2,vehiculos_test.co2_predicho)
error_test

