#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib
import matplotlib.pyplot  as plt
from sklearn.datasets import fetch_20newsgroups_vectorized
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[10,10]
np.random.seed(42)

data=fetch_20newsgroups_vectorized()


# In[8]:


noticias=data.data
noticias


# In[9]:


noticias.shape


# In[10]:


data.target_names


# In[11]:


len (data.target_names)


# In[13]:


from sklearn.cluster import MiniBatchKMeans

estimador_M=MiniBatchKMeans(n_clusters=20)
estimador_M.fit(noticias)
etiquetas_pred=estimador_M.labels_
etiquetas_pred


#  # Medidas de evaluacion externas

# In[14]:


from sklearn.metrics import homogeneity_completeness_v_measure,adjusted_rand_score
clases=data.target
homogeneity_completeness_v_measure(clases,etiquetas_pred)


# In[15]:


adjusted_rand_score(clases,etiquetas_pred)


# In[16]:


from sklearn.model_selection import cross_val_score

resultados=cross_val_score(X=noticias,y=clases,estimator=MiniBatchKMeans(),scoring='adjusted_rand_score',cv=10)
resultados.mean()


# # Medidas de evaluacion interna

# In[17]:


from sklearn.metrics import silhouette_score,calinski_harabasz_score

silhouette_score(noticias,etiquetas_pred)


# In[18]:


calinski_harabasz_score(noticias.todense(),etiquetas_pred)

