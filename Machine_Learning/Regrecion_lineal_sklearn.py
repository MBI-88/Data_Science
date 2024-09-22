#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Modelo de regrecion lineal con  sklearn
from sklearn import datasets
datos= datasets.load_boston()
datos.keys()


# In[36]:


print(datos['filename'])


# In[35]:


print(datos['data'])


# In[3]:


print(datos['DESCR'])


# In[4]:


variable_objetivo= datos['target']
variable_objetivo


# In[8]:


nombre_variable_independientes=datos['feature_names']
nombre_variable_independientes


# In[9]:


variable_independiente=datos['data']
variable_independiente


# In[14]:


from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('pinfo', 'LinearRegression')


# In[15]:


modelo= LinearRegression(n_jobs=1)
modelo.fit(X=variable_independiente,y=variable_objetivo)


# In[16]:


modelo.intercept_ #contiene el termino independiente del modelo lineal


# In[19]:


modelo.coef_ #contiene los coeficientes de Bn


# In[24]:


predicciones=modelo.predict(variable_independiente)


# In[25]:


for y,y_pred in list(zip(variable_objetivo,predicciones)) [:5]:#zip es una funcion de python que devuelve dos listas cada elemento con su correspondiente
    print('Valor Real : {:.3f} Valor Estimado : {:.5f}'.format(y,y_pred))


# In[26]:


import pandas as pd
data=pd.DataFrame(variable_independiente,columns=nombre_variable_independientes)
data.head()


# In[30]:


data['MEDV']=variable_objetivo
data.head()


# In[31]:


modelo.fit(X=data[nombre_variable_independientes],y=data['MEDV'])
data['MEDV_predic']=modelo.predict(data[nombre_variable_independientes])
data.head()


# In[34]:


modelo_normalizado=LinearRegression(normalize=True,n_jobs=-1)
modelo_normalizado.fit(X=data[nombre_variable_independientes],y=data['MEDV'])
data['MEDV_2']=modelo_normalizado.predict(data[nombre_variable_independientes])
data.head()


# In[ ]:




