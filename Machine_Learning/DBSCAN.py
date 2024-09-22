#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[12,12]

K=3
n_muestras=5000
centroides=[(24,30),(52,35),(35,60)]

X,clases_reales=make_blobs(n_samples=n_muestras,n_features=2,cluster_std=5.0,centers=centroides,shuffle=False,random_state=42)


# In[2]:


from matplotlib import cm

plt.scatter(X[:,0],X[:,1],c=clases_reales,s=20,marker='o',cmap=cm.Set3);


# In[3]:


from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score,homogeneity_completeness_v_measure

estimador_K=KMeans(n_clusters=3,random_state=42)
etiquetas_K=estimador_K.fit(X).labels_


# In[4]:


plt.scatter(X[:,0],X[:,1],c=etiquetas_K,s=20,marker='o',cmap=cm.Set3)
plt.title('Cluster producidos con K-Medias');


# # Para evaluar que punto no ha agrupado correctamente

# In[5]:


def jaccard_index(cluster_1,cluster_2):
    intersection=np.intersect1d(cluster_1,cluster_2).shape[0]
    union=np.union1d(cluster_1,cluster_2).shape[0]
    return intersection/union

jaccard_index(X[clases_reales==0],X[etiquetas_K==0])#Va de 0 a 1 y corresponde con la asignacion correcta de clusters a las clasee reales


# In[6]:


from itertools import product

def emparejar_clusters_clases(clusters,clases):
    combinaciones=product(np.unique(clases),np.unique(clusters))
    emparejamiento={}
    class_ids=np.unique(clases)
    cluster_ids=np.unique(clusters)
    for cluster_id in cluster_ids:
        old_jaccard = 0
        for class_id in class_ids:
            new_jaccard=jaccard_index(X[clases==class_id],X[clusters==cluster_id])
            if new_jaccard > old_jaccard:
                old_jaccard =  new_jaccard
                emparejamiento[cluster_id]=class_id
                if new_jaccard >= 0.5:
                    break
    return emparejamiento
emparejar_clusters_clases(etiquetas_K,clases_reales)


# In[7]:


def alinear_clusters(clusters,clases):
    reemplazo=emparejar_clusters_clases(clusters,clases)
    clusters_alineados=[reemplazo[x] for x in clusters]
    return np.array(clusters_alineados)

def  encontrar_errores(clusters,clases):
    etiquetas_clusters_alineados=alinear_clusters(clusters,clases)
    return X[clases != etiquetas_clusters_alineados]


# In[8]:


errores_K = encontrar_errores(etiquetas_K,clases_reales)
plt.scatter(X[:,0],X[:,1],s=20,c=etiquetas_K,marker='o',cmap=cm.Set3)
plt.scatter(errores_K[:,0],errores_K[:,1],s=40,marker='o',color='r')
plt.title('Clusters producidos con KMedias. Mostrando elementos erroneamente clasificados');


# In[9]:


silhouette_score(X,etiquetas_K)


# In[10]:


homogeneity_completeness_v_measure(clases_reales,etiquetas_K)


# In[11]:


estimador_D=DBSCAN(eps=10,min_samples=1000)
etiqueta_D=estimador_D.fit(X).labels_
np.unique(etiqueta_D)


# In[12]:


plt.scatter(X[:,0],X[:,1],s=20,c=etiquetas_K,marker='o',cmap=cm.Set3)
errores_D=encontrar_errores(etiqueta_D,clases_reales)
plt.scatter(errores_D[:,0],errores_D[:,1],s=40,marker='o',color='r')
plt.title('Cluster producidos con DBSCAN. Mostrando elementos erroneamente clasificados');


# In[13]:


silhouette_score(X,etiqueta_D)


# In[14]:


homogeneity_completeness_v_measure(clases_reales,etiqueta_D)


# In[15]:


from sklearn.datasets import make_circles

X,clases_reales=make_circles(n_samples=5000,factor=.3,noise=.05)
plt.scatter(X[:,0],X[:,1],c=clases_reales,s=20,marker='o',cmap=cm.Set3);


# In[16]:


estimador_K=KMeans(n_clusters=2,n_jobs=-1)
etiquetas_K=estimador_K.fit(X).labels_
plt.scatter(X[:,0],X[:,1],s=20,c=etiquetas_K,marker='o',cmap=cm.Set3)
plt.title('Clusters producido con KMedias');


# In[17]:


errores_K=encontrar_errores(clases_reales,etiquetas_K)
plt.scatter(X[:,0],X[:,1],s=20,c=etiquetas_K,marker='o',cmap=cm.Set3)
plt.scatter(errores_K[:,0],errores_K[:,1],s=40,marker='o',color='r')
plt.title('Clusters producidos con KMedias. Mostrando elementos erroneamente clasificados');


# In[18]:


silhouette_score(X,etiquetas_K)


# In[19]:


homogeneity_completeness_v_measure(clases_reales,etiquetas_K)


# In[21]:


estimador_D=DBSCAN(eps=0.05,min_samples=10)
etiquetas_D=estimador_D.fit(X).labels_
np.unique(etiquetas_D)


# In[23]:


plt.scatter(X[:,0],X[:,1],s=20,c=etiquetas_D,marker='o',cmap=cm.Set3)
errores_D=encontrar_errores(etiquetas_D,clases_reales)
plt.scatter(errores_D[:,0],errores_D[:,1],s=40,marker='o',color='r')
plt.title('Clusters producidos con DBSCAN. Mostrando elementos erroneamente clasificados');


# In[24]:


silhouette_score(X,etiquetas_D)


# In[25]:


homogeneity_completeness_v_measure(clases_reales,etiquetas_D)


# # HDBSCAN

# In[26]:


# import sys
#!conda install --yes --prefix {sys.prefix} -c conda-forge hdbscan


# In[27]:


from hdbscan import HDBSCAN

estimador_H=HDBSCAN()
etiquetas_H=estimador_H.fit_predict(X)


# In[30]:


plt.scatter(X[:,0],X[:,1],s=20,c=etiquetas_H,marker='o',cmap=cm.Set3)
errores_H=encontrar_errores(etiquetas_H,clases_reales)
plt.scatter(errores_H[:,0],errores_H[:,1],s=40,marker='o',color='r')
plt.title('Clusters producido con HDBSCAN. Mostrando elementos erroneamente cladificados');


# In[31]:


silhouette_score(X,etiquetas_H)


# In[32]:


homogeneity_completeness_v_measure(etiquetas_H,clases_reales)

