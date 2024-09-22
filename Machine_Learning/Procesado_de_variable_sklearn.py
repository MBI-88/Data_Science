#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

dataset=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/datos_procesamiento.csv')
dataset.head()


# In[2]:


dataset.dtypes


# In[13]:


variable_objetivo=['objetivo']
variable_numerica=dataset.drop(variable_objetivo,axis=1).select_dtypes(include=['int64','float64'])
variable_numerica.columns


# In[14]:


variable_numerica.head()


# In[15]:


variable_numerica[variable_numerica.isnull().any(axis=1)].shape


# In[16]:


variable_numerica[variable_numerica.isnull().any(axis=1)].head()


# In[17]:


imputador=preprocessing.Imputer(strategy='mean')
variable_numerica_imputados=imputador.fit_transform(variable_numerica)


# In[18]:


variable_numerica_imputados


# In[19]:


variable_numerica_imputados_df=pd.DataFrame(variable_numerica_imputados,index=variable_numerica.
                                           index, columns=variable_numerica.columns)
variable_numerica_imputados_df.head()


# In[20]:


variable_numerica_imputados_df[variable_numerica_imputados_df.isnull().any(axis=1)].head()


# Estandarizacion

# In[21]:


variable_numerica.columns


# In[22]:


variable_numerica_imputados_df.mean()


# In[23]:


variable_numerica_imputados_df.std()


# In[24]:


estandarizador = preprocessing.StandardScaler()
variable_numerica_imputados_estandarizado=estandarizador.fit_transform(variable_numerica_imputados_df)   


# In[25]:


estandarizador.mean_


# In[26]:


variable_numerica_imputados_estandarizado.mean(axis=0)


# In[27]:


variable_numerica_imputados_estandarizado.std(axis=0)


# In[28]:


variable_numerica_imputados_estandarizado[0]


# In[29]:


estandarizador_robusto=preprocessing.RobustScaler()
variable_numerica_imputados_estandarizado_robusto=estandarizador_robusto.fit_transform(variable_numerica_imputados_df)


# In[30]:


variable_numerica_imputados_estandarizado_robusto.mean(axis=0)


# In[31]:


variable_numerica_imputados_estandarizado_robusto.std(axis=0)


# Escalando a un rango especifico en donde se necesita valores entre [-1,1] MinMaxScaler valores entre [0,1] MaxAbscaler

# In[32]:


variable_numerica_imputados_df.min()


# In[33]:


variable_numerica_imputados_df.max()


# In[34]:


escalador_min_max=preprocessing.MinMaxScaler()
variable_numerica_imputados_df_escalado_min_max=escalador_min_max.fit_transform(variable_numerica_imputados_df)


# In[35]:


variable_numerica_imputados_df_escalado_min_max.max()


# In[36]:


variable_numerica_imputados_df_escalado_min_max.min()


# In[37]:


escalador_maxabs=preprocessing.MaxAbsScaler()
variable_numerica_imputados_df_escalado_maxabs=escalador_maxabs.fit_transform(variable_numerica_imputados_df)


# In[38]:


variable_numerica_imputados_df_escalado_maxabs.max()


# In[39]:


variable_numerica_imputados_df_escalado_maxabs.min()


#  Para casos en los que se necesita tener obervaciones con norma unitaria (norma L2 o euclidiana)

# In[40]:


normalizador=preprocessing.Normalizer()
variable_numerica_imputados_df_normalizadas=normalizador.fit_transform(variable_numerica_imputados_df)


# In[41]:


variable_numerica_imputados_df_normalizadas[1,:]


# In[42]:


np.linalg.norm(variable_numerica_imputados_df_normalizadas[1,:])


# Otras transformaciones para casos de desviacion estandar muy variada

# In[45]:


sns.kdeplot(dataset.col3);# col3 no tiene una distribucion normal


# Una practica es usar logaritmos para convertirlas a variables con distribucion normal

# In[46]:


transformer=preprocessing.FunctionTransformer(np.log1p)
cilindro_transformados=transformer.transform(variable_numerica_imputados_df[['col3']])
cilindro_transformados=cilindro_transformados.reshape(cilindro_transformados.shape[0],)
sns.kdeplot(cilindro_transformados);


# In[47]:


df_numerico_procesado=pd.DataFrame(variable_numerica_imputados_df_normalizadas,columns=variable_numerica_imputados_df.columns)
df_numerico_procesado.head()


# Procesado de variable parte 2

# Procesado de variables categoricas

# In[53]:


variables_categoricas=dataset[['col_categorica','col_ordinal']]
variables_categoricas.head()


# In[54]:


label_encode_ordinal=preprocessing.LabelEncoder()
label_encode_ordinal.fit(variables_categoricas.col_ordinal)


# In[55]:


label_encode_ordinal.classes_


# In[56]:


label_encode_ordinal.transform(['bien', 'mal', 'muy bien', 'muy mal', 'regular'])


# In[57]:


#Variante

label_encode_ordinal.inverse_transform([0,0,1,2])


# In[58]:


label_encode_categorico=preprocessing.LabelEncoder()
label_encode_categorico.fit_transform(variables_categoricas.col_categorica)[:15]


# In[59]:


label_encode_categorico.classes_


# Utilizando One-hot encoding

# In[60]:


oh_codificador=preprocessing.OneHotEncoder()
categorias_codificadas=label_encode_categorico.transform(variables_categoricas.col_categorica)


# In[62]:


categorias_codificadas


# In[64]:


categorias_codificadas_oh=oh_codificador.fit_transform(categorias_codificadas.reshape(1000,1))
categorias_codificadas_oh


# In[65]:


categorias_codificadas_oh.toarray()#convirtiendo a array de numpy


# In[66]:


import sys # Para ver el tamaño en bytes  que ocupa la matriz
sys.getsizeof(categorias_codificadas_oh)


# In[67]:


sys.getsizeof(categorias_codificadas_oh.toarray())


# In[68]:


#Para usar array tipo numpy
oh_codificador=preprocessing.OneHotEncoder(sparse=False)
categorias_codificadas_oh=oh_codificador.fit_transform(categorias_codificadas.reshape(1000,1))
categorias_codificadas_oh


# In[69]:


oh_codificador.feature_indices_


# Haciendo lo mismo pero con Pandas

# In[74]:


df_categoricos_procesados=pd.get_dummies(variables_categoricas.col_categorica)
df_categoricos_procesados.head()


# Para procesado de texto

# In[77]:


from sklearn import feature_extraction

dataset.col_texto.values[:10]


# In[80]:


vectorizador_count=feature_extraction.text.CountVectorizer()#Tiene un error cuenta las palabras repetidas
# sin tener en cuenta el valor que tiene en la oracion
X=vectorizador_count.fit_transform(dataset.col_texto)
pd.DataFrame(X.toarray(),columns=vectorizador_count.get_feature_names())


# In[83]:


#Variante para  evitar ese error es usar TF-IDF
vectorizador_tfidf= feature_extraction.text.TfidfVectorizer()
X=vectorizador_tfidf.fit_transform(dataset.col_texto)
df_texto_procesado=pd.DataFrame(X.toarray(),columns=vectorizador_tfidf.get_feature_names())


# Para hacer concatenacion de lo antes realizado

# In[85]:


datos_procesados=pd.concat([df_numerico_procesado,df_categoricos_procesados,df_texto_procesado],axis=1)
datos_procesados['col_ordinal']=label_encode_ordinal.fit_transform(variables_categoricas.col_ordinal)
datos_procesados.head(10)


# In[87]:


datos_procesados.to_csv('datos_procesados_parciales_1.csv',index=False)

