#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder

np.random.seed(42)

pelis=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv')
pelis=pelis[pelis['genero'].notnull()]
n_genero=len(pelis.genero.unique())# 17 generos
generos_pelis=pelis.genero.values# lista de generos
pelis=pelis.drop('genero',axis=1)
pelis=OneHotEncoder().fit_transform(pelis)
pelis.head()


# In[2]:


X=SimpleImputer().fit_transform(pelis)
X


# In[8]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

pelis_escalado=MinMaxScaler().fit_transform(X)
pelis_df=pd.DataFrame(pelis_escalado,columns=pelis.columns)

estimador_K=KMeans(n_clusters=n_genero,n_jobs=-1,random_state=42)
cluster=estimador_K.fit_predict(pelis_df)


# # Coeficiente de silueta

# In[9]:


from sklearn.metrics import adjusted_rand_score,silhouette_score

coeficiente_silueta=silhouette_score(pelis_df,labels=cluster,random_state=42)


# In[10]:


coeficiente_silueta


# In[11]:


indice_Rand=adjusted_rand_score(generos_pelis,cluster)
indice_Rand


# In[7]:


print(adjusted_rand_score.__doc__)

