#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics,datasets
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
import numpy as np
import pandas as pd

boston= datasets.load_boston()
def rmse(objetivo,estimado):# Raiz del error cuadratico medio
    return np.sqrt(metrics.mean_squared_error(objetivo,estimado))
def adjustado(objetivo,estimado,n,k):# Error ajustado 
    r2=metrics.r2_score(objetivo,estimado)
    return 1-(1-r2)*(n-1)/(n-k-1)

def evaluar_modelo(objetivo,estimado,n,k):
    return{
        'rmse':rmse(objetivo,estimado),
        'mae':metrics.mean_absolute_error(objetivo,estimado),
        'adjusted_r2':adjustado(objetivo,estimado,n,k)
    }
modelo_ols=LinearRegression(n_jobs=1)
modelo_ols.fit(X=boston.data,y=boston.target)
modelo_ols_pred=modelo_ols.predict(boston.data)


# In[12]:


Resultados={}
N=boston.data.shape[0]
Resultados['ols']=evaluar_modelo(
boston.target,modelo_ols_pred,N,len(modelo_ols.coef_))
Resultados


# In[13]:


boston.data.shape


# In[14]:


#Funcion para separar el dataset en datos entrenamiento
X_train,X_test,y_train,y_test= train_test_split(
    boston.data,boston.target,test_size=0.33,random_state=13)
print(X_train.shape,y_train.shape)


# In[15]:


print(X_test.shape,y_train.shape)


# In[16]:


modelo_ols=LinearRegression(n_jobs=1)
modelo_ols.fit(X=X_train,y=y_train)
modelo_ols_train_pred=modelo_ols.predict(X_train)


# In[17]:


Resultados['ols_train']=evaluar_modelo(
y_train,modelo_ols_train_pred,X_train.shape[0],len(modelo_ols.coef_))

modelo_ols_test_pred=modelo_ols.predict(X_test)
Resultados['ols_test']=evaluar_modelo(
y_test,modelo_ols_test_pred,X_test.shape[0],len(modelo_ols.coef_))
dataFrame=pd.DataFrame(Resultados)
dataFrame.head()


# In[18]:


#Funcion para comprobar que semilla funciona mejor
model=LinearRegression()
resultado=[]
def test_semilla(semilla):
    X_train,X_test,y_train,y_test=train_test_split(
    boston.data,boston.target,test_size=0.33,random_state=semilla)
    test_pred=model.fit(X_train,y_train).predict(X_test)
    semilla_rmse=rmse(y_test,test_pred)
    resultado.append([semilla_rmse,semilla])  
for i in range(1000):
    test_semilla(i)
resultado_sorted=sorted(resultado,key=lambda x: x[0],reverse=False)
resultado[:5]


# In[19]:


resultado_sorted[0]#Error minimio


# In[20]:


resultado_sorted[-1]#Error mayor


# In[26]:


#Validacion cruzada(Para evitar cometer error con  la semilla)

get_ipython().run_line_magic('pinfo', 'cross_val_score')


# In[21]:


model_ols=LinearRegression()
X=boston.data
y=boston.target
resultados_validacion_cruzada=cross_val_score(
estimator=model_ols,
    X=X,
    y=y,
    scoring='neg_mean_squared_error',
    cv=10
)
resultados_validacion_cruzada


# In[22]:


resultados_validacion_cruzada.mean()


# In[23]:


def rmse_cross_val(estimador,X,y):#Esta funcion se puede utilizar en cross_val_score en scoring , scoring acepta funciones definidas por el usuario
    y_pred=estimador.predict(X)
    return np.sqrt(metrics.mean_squared_error(y,y_pred))
resultado_cv=[]
for i in range(10,200):
    cv_rmse=cross_val_score(
    estimator=model_ols,
        X=X,
        y=y,
        scoring=rmse_cross_val,
        cv=i
    ).mean()
    resultado_cv.append(cv_rmse)
resultado_cv[:5]


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inLine')
import matplotlib.pyplot as plt
plt.plot(resultado_cv);


# In[26]:


scoring={'mae':'neg_mean_absolute_error','rmse':rmse_cross_val}
estimator=modelo_ols
scores=cross_validate(estimator,
                      boston.data,
                      boston.target,
                      scoring=scoring,cv=100,
                      return_train_score=True)


# In[27]:


pd.DataFrame(scores).mean()#Devuelve la media de los valores de cross_validate

