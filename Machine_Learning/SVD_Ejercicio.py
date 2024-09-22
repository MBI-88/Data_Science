#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df_ventas=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/retail.csv')
df_ventas.head()


# In[2]:


y_test=dict(zip(df_ventas.StockCode,df_ventas.Description))


# In[3]:


ventas_matrix=df_ventas.pivot_table(values='Quantity',index='CustomerID',columns='StockCode',fill_value=0)
ventas_matrix.head()


# In[4]:


lista_clientes=np.array(ventas_matrix.index.tolist())
item_id_lista=np.array(ventas_matrix.columns.tolist())


# In[5]:


ventas_matrix_sp=np.array(ventas_matrix,dtype=float)
ventas_matrix_sp


# In[6]:


from sklearn.decomposition import TruncatedSVD

truncador=TruncatedSVD(n_components=10,n_iter=10,random_state=42,algorithm="arpack")
ventas_svd=truncador.fit_transform(ventas_matrix_sp)


# In[7]:


ventas_svd


# In[8]:


ventas_svd.shape


# In[9]:


def recomendar(id_cliente,num_recomendaciones=5):
    identificador=np.where(lista_clientes==id_cliente)[0][0]
    index_sort=ventas_svd[identificador,:].argsort()[::-1]
    no_comprados=ventas_matrix_sp[identificador,:][index_sort]==0
    rec_index=index_sort[no_comprados]
    rec_ids=item_id_lista[rec_index]
    recomendaciones=rec_ids[:num_recomendaciones]
    return recomendaciones


# In[10]:


recomendacion=recomendar(12352,)
sugeridos=[y_test[x] for x in recomendacion]


# # Comprobando el resultado

# In[11]:


cliente_id=12352
ventas_matrix.head()


# In[12]:


d=ventas_matrix.loc[cliente_id]
comprados=[y_test[x] for x in d[d.values!=0].index]


# In[13]:


lista=list(zip(sugeridos,comprados))
lista


# In[14]:


sugeridos in comprados # comprobando que los sugeridos no se hallan comprados


# In[42]:


sugeridos # Este es el envio al cliente para sugerirle nuevas compras

