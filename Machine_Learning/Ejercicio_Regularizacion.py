#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn import metrics
from sklearn.model_selection import cross_val_score

movie=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv').select_dtypes(np.number).fillna(0)
variable_objetivo='ventas'
movie.head()


# In[66]:


train_data=movie.drop([variable_objetivo],axis=1)
train_data.head()


# In[86]:


def rmse(estimador,X,y):
    preds=estimador.predict(X)
    return np.sqrt(metrics.mean_squared_error(y,preds))

def evaluador(estimador,X,y):
    return cross_val_score(estimador,X,y,scoring=rmse,cv=5).mean()


# In[92]:


modelo_lineal=LinearRegression()
modelo_lineal.fit(train_data,movie[variable_objetivo])
estimado_lineal=modelo_lineal.predict(train_data)

modelo_lasso=Lasso(alpha=1.0)
modelo_lasso.fit(train_data,movie[variable_objetivo])
etimado_lasso=modelo_lasso.predict(train_data)

modelo_ridge=Ridge(alpha=1.0)
modelo_ridge.fit(train_data,movie[variable_objetivo])
estimado_ridge=modelo_ridge.predict(train_data)

modelo_elasticnet=ElasticNet(l1_ratio=0.5)
modelo_elasticnet.fit(train_data,movie[variable_objetivo])
estimado_elasticnet=modelo_elasticnet.predict(train_data)


# In[93]:


Resultado={}
Resultado['evaluaciones']={
    'validacion_lineal':evaluador(modelo_lineal,train_data,movie[variable_objetivo]),
    'validacion_lasso':evaluador(modelo_lasso,train_data,movie[variable_objetivo]),
    'validacion_ridge':evaluador(modelo_ridge,train_data,movie[variable_objetivo]),
    'validacion_elasticnet':evaluador(modelo_elasticnet,train_data,movie[variable_objetivo])
}
pd.set_option('display.float_format',lambda x: str(round(x,3)))
resultado_df=pd.DataFrame(Resultado).T
resultado_df.head()

