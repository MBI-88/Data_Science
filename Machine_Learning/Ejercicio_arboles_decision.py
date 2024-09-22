#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import numpy as np
from category_encoders import OneHotEncoder
import graphviz


movies=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv').drop(columns=['titulo'])
variable_objetivo= 'ventas'
movies=movies[movies[variable_objetivo].notnull()]
variable_indep=movies.drop(variable_objetivo,axis=1).columns

movies=OneHotEncoder().fit_transform(movies)
imputador=preprocessing.Imputer(strategy='mean').fit_transform(movies.drop(variable_objetivo,axis=1))


# In[ ]:


arbol=tree.DecisionTreeRegressor(max_depth=5)
arbol.fit(imputador,movies[variable_objetivo])


# In[3]:


variables_arbol=movies.drop(variable_objetivo,axis=1).columns
tree.export_graphviz(arbol,out_file='arbol.dot',filled=True,feature_names=variables_arbol)


# In[4]:


arbol.feature_importances_


# In[5]:


sorted(zip(arbol.feature_importances_,variables_arbol),reverse=True)


# Variante

# In[6]:


import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import numpy as np

peli=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv').drop(columns=['titulo'])
objetivo='ventas'
peli.head()


# In[7]:


variable_nu=peli.drop(objetivo,axis=1).select_dtypes(np.number)
variable_cate=peli.drop(objetivo,axis=1).select_dtypes(['object'])
imputador=preprocessing.Imputer(strategy='mean')

variable_cate=pd.get_dummies(variable_cate)
df_ajustado=(pd.concat([variable_nu,variable_cate,peli[objetivo]],axis=1))
X=imputador.fit_transform(df_ajustado.drop(objetivo,axis=1))
columna_1=df_ajustado.drop(objetivo,axis=1).columns
df_final=pd.DataFrame(X,columns=columna_1)
df_final[objetivo]=peli[objetivo]
df_final.head()


# In[8]:


df_final=df_final[df_final[objetivo].notnull()]
arbol_nuevo=tree.DecisionTreeRegressor(max_depth=5)
arbol_nuevo.fit(df_final.drop(objetivo,axis=1),df_final[objetivo])


# In[9]:


arbol_nuevo.feature_importances_


# In[10]:


sorted(zip(arbol_nuevo.feature_importances_,peli.drop(objetivo,axis=1).columns),reverse=True)

