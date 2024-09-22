#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[14,14]
np.random.seed(42)


# In[2]:


vehiculos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/vehiculos_procesado_con_grupos.csv').drop(
    ['fabricante','modelo','transmision','traccion','clase','combustible','consumo'],axis=1)
vehiculos.head()


# In[3]:


datos_numericos=vehiculos.select_dtypes(np.number)
datos_categoricos=vehiculos.select_dtypes(include=['category',object])

for col in datos_numericos.columns:
    datos_numericos[col].fillna(datos_numericos[col].mean(),inplace=True)

datos_numericos.isnull()


# In[4]:


from sklearn.preprocessing import MinMaxScaler #con K-medias se necesita escalar los datos

datos_numericos_normalizados=MinMaxScaler().fit_transform(datos_numericos)
datos_numericos_normalizados=pd.DataFrame(datos_numericos,columns=datos_numericos.columns)
datos_categoricos_codificados=pd.get_dummies(datos_categoricos,drop_first=True)
vehiculos_procesados=pd.concat([datos_numericos_normalizados,datos_categoricos_codificados],axis=1)
vehiculos_procesados.head()


# In[5]:


from sklearn.cluster import KMeans

estimador_k=KMeans(random_state=42,n_clusters=8,n_jobs=-1)
estimador_k.fit(vehiculos_procesados)


# In[6]:


clusters=estimador_k.labels_
clusters


# In[7]:


estimador_k.predict(vehiculos_procesados)


# In[8]:


centroides=estimador_k.cluster_centers_
centroides


# In[9]:


centroides.shape


# In[10]:


estimador_k.inertia_


# In[11]:


print(KMeans.__doc__)


# In[12]:


from sklearn.metrics import euclidean_distances

distancia_centroide=euclidean_distances(centroides)
distancia_centroide


# # Buscando los cluster mas distintos por medio de la  distancia

# In[13]:


list(zip(np.argmax(distancia_centroide,axis=1),np.max(distancia_centroide,axis=1)))


# In[14]:


def resumen_cluster(cluster_id):
    cluster=vehiculos[clusters==cluster_id]
    resumen_cluster=cluster[datos_categoricos.columns].mode().to_dict(orient='records')[0]
    resumen_cluster.update(cluster.mean().to_dict())
    resumen_cluster['cluster_id']=cluster_id
    return resumen_cluster

def comparar_clusters(*cluster_ids):
    resumenes=[]
    for cluster_id in cluster_ids:
        resumenes.append(resumen_cluster(cluster_id))
    return pd.DataFrame(resumenes).set_index('cluster_id').T


# In[15]:


resumen_cluster(0)


# In[16]:


comparar_clusters(0,5)


# In[17]:


comparar_clusters(*np.unique(clusters))


# # Variante de forma mas grafica

# In[18]:


def Kmeans_cluster(df,n_clusters=2):
    model=KMeans(n_clusters=n_clusters,random_state=42)
    clusters=model.fit_predict(df)
    clusters_results=df.copy()
    clusters_results['Cluster']=clusters
    return clusters_results

def resumen_grafico_clustering(results):
    cluster_size=results.groupby(['Cluster']).size().reset_index()
    cluster_size.columns=['Cluster','Count']
    cluster_means=results.groupby(['Cluster'],as_index=False).mean()
    cluster_summary=pd.merge(cluster_size,cluster_means,on='Cluster')
    cluster_summary=cluster_summary.drop(['Count'],axis=1).set_index('Cluster')
    return cluster_summary[sorted(cluster_summary.columns)]

clusters_results=Kmeans_cluster(vehiculos_procesados,8)
cluster_summary=resumen_grafico_clustering(clusters_results)


# In[19]:


cluster_summary


# In[20]:


import seaborn as sns

sns.heatmap(cluster_summary.transpose(),annot=True);


# In[21]:


from sklearn.cluster import MiniBatchKMeans


estimador_k2=MiniBatchKMeans(random_state=42,n_clusters=8).fit(vehiculos_procesados)


# In[22]:


estimador_k2.inertia_# comprobando la inercia


# In[23]:


estimador_k=KMeans(random_state=42,n_clusters=8,n_jobs=-1).fit(vehiculos_procesados)
estimador_k.inertia_


# # Como elegir K

# In[24]:


# Metodo del codo
from scipy.spatial.distance import cdist
print(cdist.__doc__)


# In[25]:


varianza_total=cdist(XA=vehiculos_procesados,XB=np.array([vehiculos_procesados.mean()]))
suma_varianza_total=varianza_total.sum()
suma_varianza_total


# In[26]:


def varianza_cluster(cluster_id,centroide_cluster,etiquetas_clusters):
    elementos_cluster=vehiculos_procesados[etiquetas_clusters==cluster_id]
    return cdist(XA=elementos_cluster,XB=np.array([centroide_cluster])).sum()

def medida_vairanza(estimador_kmedia,suma_varianza_total):
    etiquetas_clusters = estimador_kmedia.labels_
    wss=0
    for i,cluster_id in enumerate(np.unique(etiquetas_clusters)):
        centroide_cluster=estimador_kmedia.cluster_centers_[i]
        wss +=varianza_cluster(cluster_id,centroide_cluster,etiquetas_clusters)
        return (suma_varianza_total-wss)/suma_varianza_total
    


# In[35]:


def medida_inercia(estimdor_kmedias):
    return estimdor_kmedias.inertia_

def evaluar_K_Kmedias(k,medida,**kwargs):
    if medida=='inercia':
        funcion_medida=medida_inercia
    elif medida=='varianza':
        funcion_medida=medida_vairanza
        
    estimdor_kmedias=KMeans(random_state=42,n_clusters=k)
    estimdor_kmedias.fit(vehiculos_procesados)
    return funcion_medida(estimdor_kmedias,**kwargs)


# In[36]:


resultados_k={}
rango_k=[5,10,20,30,50,75,100,200,300]
for k in rango_k:
    resultados_k[k]=evaluar_K_Kmedias(k,'inercia'), evaluar_K_Kmedias(k,'varianza',suma_varianza_total=suma_varianza_total)

resultados_k


# In[40]:


fig, ax1=plt.subplots()

ax1.plot(
    [c[0]  for c in resultados_k.items()],
    [c[1][0] for  c in resultados_k.items()],label='inercia',color='red'
)
ax1.set_ylabel('inercia')

ax2=ax1.twinx()

ax2.plot(
    [c[0] for c in resultados_k.items()],
    [c[1][1] for c in resultados_k.items()],label='varianza'
)
ax2.set_ylabel('varianza explicada')
plt.xlabel('K')
plt.legend()
plt.title('Variacion de inercia/varianza respecto a K');

