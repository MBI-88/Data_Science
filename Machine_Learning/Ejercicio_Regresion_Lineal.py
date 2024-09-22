#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
from sklearn.linear_model import LinearRegression

movie=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv')
movie.head()


# In[38]:


variable_objetivo=movie.ventas
variables_numericas=movie.drop(['ventas'],axis=1).select_dtypes(include=['int64','float64'])
variables_numericas.head()


# In[39]:


valores_df= len(variables_numericas)
def valores_Nan(df):
    for col in df:
        print('{} | {} | {}'.format(df[col].name,len(df[df[col].isnull()])/(1.0*valores_df),df[col].dtype))

valores_Nan(variables_numericas)


# In[40]:


variables_numericas['presupuesto']=variables_numericas.presupuesto.fillna(0)
variables_numericas['duracion']=variables_numericas.duracion.fillna(0)


# In[41]:


valores_Nan(variables_numericas)#Comprobando que no hay valores Nan


# In[50]:


variables_numericas.head()


# In[49]:


variable_objetivo[variable_objetivo.isnull()].head()


# In[51]:


variable_objetivo_imp=movie.ventas.fillna(0)


# In[69]:


modelo=LinearRegression()
modelo.fit(X=variables_numericas,y=variable_objetivo_imp)
movie['ventas_pred']=modelo.predict(variables_numericas)
modelo_pred=modelo.predict(variables_numericas)
movie.head()


# In[83]:


movie[['ventas','ventas_pred']].head()


# Validando el modelo

# In[70]:


from sklearn import metrics

metrics.mean_absolute_error(variable_objetivo_imp,modelo_pred)


# In[77]:


m_r2=metrics.median_absolute_error(variable_objetivo_imp,modelo_pred)
m_r2

