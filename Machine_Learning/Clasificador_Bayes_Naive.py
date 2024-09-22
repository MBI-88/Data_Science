#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

noticias=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/noticias.csv')
noticias.head()


# In[11]:


noticias.shape


# In[12]:


noticias.categoria.value_counts()


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
import json
with open('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/stopwords-es.json')as fname:
    stopwords_es= json.load(fname)


# In[14]:


stopwords_es[:30]


# In[15]:


vectorizador=TfidfVectorizer(strip_accents='unicode',stop_words=stopwords_es)
vectorizador.fit_transform(noticias.descripcion)


# In[16]:


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import issparse
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB


class DenseTransformer(BaseEstimator):
    def __init__(self,return_copy=True):
        self.return_copy=return_copy
        self.is_fitted=False
    def transform(self,X,y=None):
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X
    def fit(self,X,y=None):
        self.is_fitted=True
        return self
    def fit_transform(self,X,y=None):
        return self.transform(X=X,y=y)


# In[17]:


pipeline_gaussiano = make_pipeline(
    vectorizador,
    DenseTransformer(),
    GaussianNB()
)
pipeline_gaussiano.fit(X=noticias.descripcion,y=noticias.categoria)


# In[18]:


pipeline_gaussiano.predict(noticias.descripcion)


# In[19]:


cross_val_score(pipeline_gaussiano,noticias.descripcion,noticias.categoria,scoring='f1')
#Este recurso falla por una seleccion multiclase y tener una media binaria


# Variante para solucionar este problema

# In[20]:


from sklearn.metrics import f1_score

def f1_multietiqueta(estimador,X,y):
    preds=estimador.predict(X)
    return f1_score(y,preds,average='micro')#macro se utiliza cuando las clases estan valanciadas

cross_val_score(pipeline_gaussiano,noticias.descripcion,noticias.categoria,scoring=f1_multietiqueta)


# Variante para que tarde menos

# In[21]:


pipeline_gaussiano = make_pipeline(
    TfidfVectorizer(strip_accents='unicode',stop_words=stopwords_es,max_features=1000),
    DenseTransformer(),
    GaussianNB()
)
cross_val_score(pipeline_gaussiano,noticias.descripcion,noticias.categoria,scoring=f1_multietiqueta)
#max_feature se disminuye las palabras parabuscar


# In[22]:


pipeline_multinominal=make_pipeline(
    TfidfVectorizer(strip_accents='unicode',stop_words=stopwords_es,max_features=500),
    DenseTransformer(),
    MultinomialNB()
)
cross_val_score(pipeline_multinominal,noticias.descripcion,noticias.categoria,scoring=f1_multietiqueta)


# Nota: Se nota una mejor aproxomacion que la que dio el Gaussiano

# In[23]:


#Para utilizar el clasificador de BermoulliNB se vectoriza de forma binaria
from sklearn.feature_extraction.text import CountVectorizer

vectorizador_count=CountVectorizer(stop_words=stopwords_es,binary=True,strip_accents='unicode',
                                  max_features=1000)


# In[24]:


vectorizador_count.fit(noticias.descripcion)
vectorizador_count.vocabulary_


# In[25]:


pipeline_bermoulli=make_pipeline(
    vectorizador_count,
    DenseTransformer(),
    BernoulliNB()
)
cross_val_score(pipeline_bermoulli,noticias.descripcion,noticias.categoria,scoring=f1_multietiqueta)

