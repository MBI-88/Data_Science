#!/usr/bin/env python
# coding: utf-8

# ### Ejercicio  - Seleccion de Variables

# In[59]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.feature_selection import f_regression,RFE,SelectKBest

pelis=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv').drop(columns='titulo')
pelis.head()


# In[69]:


objetivo='ventas'
pelis=pelis[pelis[objetivo].notnull()]
independientes=pelis.drop(objetivo,axis=1).columns
pelis_nuevo=OneHotEncoder().fit_transform(pelis.drop(objetivo,axis=1))
lista_columnas=[]
lista_columnas.append(pelis_nuevo.columns)


# In[70]:


X=pd.DataFrame(SimpleImputer().fit_transform(pelis_nuevo),columns=lista_columnas)
y=pelis[objetivo]
X.head()


# In[71]:


selector_kbest_5=SelectKBest(f_regression,k=5)
pelis_new_kbest=selector_kbest_5.fit_transform(X,y)
pelis_new_kbest.shape


# In[72]:


from sklearn.ensemble import RandomForestRegressor

selector_rfe_20=RFE(RandomForestRegressor(),n_features_to_select=20)
pelis_new_rfe=selector_rfe_20.fit_transform(X,y)


# In[81]:


col_selec_kbest_5=sorted(
    filter(
    lambda c:  c[2],zip( 
        X.columns,
        selector_kbest_5.scores_,
        selector_kbest_5.get_support() )
    ),key=lambda c:c[1],reverse=True
)
col_selec_kbest_5


# In[80]:


col_selec_rfe_20=sorted(
    filter(
        lambda c: c[2],zip(X.columns,selector_rfe_20.ranking_,selector_rfe_20.get_support())
    ),key=lambda c: c[1],reverse=True
)
col_selec_rfe_20


# ### Variante para mostrar las 20 columnas

# In[84]:


variables_inde=pelis_nuevo.columns
variables_inde[selector_rfe_20.support_]

