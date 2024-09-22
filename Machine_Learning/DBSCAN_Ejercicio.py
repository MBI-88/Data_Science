#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/CC GENERAL.csv')
df.dtypes


# In[2]:


df=df.drop('CUST_ID',axis=1)
df.head()


# In[3]:


df.shape


# In[4]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df_1=SimpleImputer(strategy='constant',fill_value=0).fit_transform(df)
df_normalizado=pd.DataFrame(StandardScaler().fit_transform(df_1),columns=df.columns)
df_normalizado.head()


# In[5]:


estimador_D=DBSCAN()
etiquetas_D=estimador_D.fit_predict(df_normalizado)
pd.Series(etiquetas_D).value_counts()


# In[6]:


from sklearn.metrics import silhouette_score

silhouette_score(df_normalizado,etiquetas_D)


# In[7]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform

distribucion_param={
    'eps':uniform(0,5),
    'min_samples':sp_randint(2,20),
    'p':sp_randint(1,3),
}


# In[8]:


from sklearn.model_selection import ParameterSampler #Busqueda de parametros optimos

n_muestras=30
n_iteraciones=3
pct_muestras=0.7
resultados_busqueda=[]
lista_param=list(ParameterSampler(distribucion_param,n_iter=n_muestras))

for param in lista_param:
    for iteration in range(n_iteraciones):
        param_resultados=[]
        muestra=df_normalizado.sample(frac=pct_muestras)
        etiquetas_clusters=DBSCAN(n_jobs=-1,**param).fit_predict(muestra)
        try:
            param_resultados.append(silhouette_score(muestra,etiquetas_clusters))
        except ValueError:
            pass
    puntuacion_media=np.mean(param_resultados)
    resultados_busqueda.append([puntuacion_media,param])


# In[9]:


sorted(resultados_busqueda, key=lambda x: x[0],reverse=True)[:5]


# In[10]:


mejores_parametros={'eps': 4.8225111292359975, 'min_samples': 15, 'p': 2}
estimador_D=DBSCAN(n_jobs=-1,**mejores_parametros)
etiquetas_D=estimador_D.fit_predict(df_normalizado)
pd.Series(etiquetas_D).value_counts()


# In[11]:


def resumen_cluster(cluster_id):
    cluster=df[etiquetas_D==cluster_id]
    resumen_cluster=cluster.mean().to_dict()
    resumen_cluster['cluster_id']=cluster_id
    return resumen_cluster

def comparar_cluster(*cluster_ids):
    resumenes=[]
    for cluster_id in cluster_ids:
        resumenes.append(resumen_cluster(cluster_id))
    return pd.DataFrame(resumenes).set_index('cluster_id').T

comparar_cluster(0,-1)

