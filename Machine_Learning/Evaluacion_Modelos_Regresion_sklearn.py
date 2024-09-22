#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import datasets,metrics
from sklearn.linear_model import LinearRegression
boston= datasets.load_boston()
modelo=LinearRegression()
modelo.fit(X=boston.data,y=boston.target)
y_objetivo=boston.target
y_predi=modelo.predict(boston.data)


# In[8]:


metrics.mean_absolute_error(y_objetivo,y_predi)#Error absoluto de la prediccion es la diferencia
#entre el valor real y el predicho


# In[3]:


import numpy as np
np.sqrt(metrics.mean_squared_error(y_objetivo,y_predi))#raiz cuadrada del herror absoluto medio


# In[4]:


modelo_r2= metrics.r2_score(y_objetivo,y_predi) #coeficiente  de determinacion, mide la porcion  de la verianza de la variable objetivo que se puede explicar por el modelo

modelo_r2


# In[5]:


np.corrcoef(y_objetivo,y_predi)**2 #correlacion al cuadrado


# In[6]:


len(modelo.coef_)


# In[7]:


modelo_r2_ajustado=1-(1-modelo_r2)*(len(boston.target)-1)/(len(boston.target)-boston.data.shape[1]-1)
modelo_r2_ajustado  # El modelo ajustado tiene en consideracion la complejidad del modelo, donde n es el numero de observaciones
# y k es el numero de coeficinetes del modelo(sin contar el termino  independiente)

