#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

vehiculos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/data_frame_estandarizado.csv')
datos_entrenamiento= vehiculos[['desplazamiento','cilindros','consumo']]
objetivo=vehiculos.co2


# In[2]:


valores_df=len(vehiculos)
def valores_inexistentes(df):#Muestra valores inexistentes en el data frame
    for columna in df:
        print('{} | {} | {} |'.format(df[columna].name,len(df[df[columna].isnull()])/(1.0*valores_df),df[columna].dtype))
        
        
valores_inexistentes(vehiculos)


# In[3]:


modelo_ols=LinearRegression(n_jobs=1)
modelo_ols.fit(datos_entrenamiento,objetivo)
modelo_ols.coef_


# In[4]:


def norma_l1(coeficientes):#Aplicando la norma L1
    return np.abs(coeficientes).sum()

def norma_l2(coeficientes):#Aplicando la norma L2
    return np.sqrt(np.power(coeficientes,2).sum())

print(norma_l1(modelo_ols.coef_))
print(norma_l2(modelo_ols.coef_))


# In[5]:


#Utilizando la funcion de numpy.linalg.norm que calcula la norma (Hace lo mismo que lo anterior)
def norma_l1_np(coeficiente):
    return np.linalg.norm(coeficiente,ord=1)
def norma_l2_np(coeficiente):
    return np.linalg.norm(coeficiente,ord=2)
print(norma_l1_np(modelo_ols.coef_))
print(norma_l2_np(modelo_ols.coef_))


# In[6]:


def norma_l1_cv(estimador,X,y):
    return norma_l1(estimador.coef_)
def norma_l2_cv(estimador,X,y):
    return norma_l2(estimador.coef_)


# In[7]:


from sklearn.preprocessing import PolynomialFeatures
transformador_polinmial=PolynomialFeatures(5)
transformador_polinmial.fit(datos_entrenamiento)


# In[8]:


variables_polinomiales= transformador_polinmial.transform(datos_entrenamiento)
variables_polinomiales.shape


# In[9]:


datos_entrenamiento.loc[0]


# In[10]:


variables_polinomiales[0]


# In[11]:


#Ahorrando un metodo (Hace lo mismo del anterior pero con un solo metodo)
variables_polinomiales=PolynomialFeatures(5).fit_transform(datos_entrenamiento)
variables_polinomiales.shape


# In[12]:


#Evaluando los distintos tipos de regularizacion
Resultados={}
modelo_ols=LinearRegression(n_jobs=1)
modelo_ols.fit(variables_polinomiales,objetivo)
print(modelo_ols.coef_)

Resultados['ols']={
    'norma_l1':norma_l1(modelo_ols.coef_),
    'norma_l2':norma_l2(modelo_ols.coef_),
}


# In[13]:


modelo_l1=Lasso(alpha=1.0,tol=0.01,max_iter=5000)
modelo_l1.fit(variables_polinomiales,objetivo)
print(modelo_l1.coef_)
Resultados['regularizacion_l1']={
    'norma_l1':norma_l1(modelo_l1.coef_),
    'norma_l2':norma_l2(modelo_l1.coef_),
    
}


# In[14]:


modelo_l2=Ridge(alpha=1.0,tol=0.01,max_iter=5000)
modelo_l2.fit(variables_polinomiales,objetivo)
print(modelo_l2.coef_)
Resultados['regularizacion_l2']={
    'norma_l1':norma_l1(modelo_l2.coef_),
    'norma_l2':norma_l2(modelo_l2.coef_),
    
}


# In[15]:


mode_elasticnet=ElasticNet(l1_ratio=0.5,tol=0.01,max_iter=5000)
mode_elasticnet.fit(variables_polinomiales,objetivo)
print(mode_elasticnet.coef_)
Resultados['regulaciones_elasticnet']={
    'norma_l1':norma_l1(mode_elasticnet.coef_),
    'norma_l2':norma_l2(mode_elasticnet.coef_),  
}


# In[16]:


pd.set_option('display.float_format',lambda x: str(round(x,6)))
resultados_df=pd.DataFrame(Resultados).T
l1_ols=resultados_df.loc['ols','norma_l1']
l2_ols=resultados_df.loc['ols','norma_l2']

resultados_df['pct_reduccion_l1']= 1-resultados_df.norma_l1/l1_ols
resultados_df['pct_resuccion_l2']= 1-resultados_df.norma_l2/l2_ols
resultados_df

