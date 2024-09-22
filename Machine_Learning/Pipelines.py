#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing,feature_extraction
from sklearn.impute import SimpleImputer

data=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/datos_procesamiento.csv')

class BinarizadorCategorico(preprocessing.LabelBinarizer):
    def fit(self,X,y=None):
        super(BinarizadorCategorico,self).transform(X)
    
    def transform(self,X,y=None):
        return super(BinarizadorCategorico,self).transform(X)
    
    def fit_transform(self,X,y=None):
        return super(BinarizadorCategorico,self).fit(X).transform(X)
    

class CodificadorCategorico(preprocessing.LabelEncoder):
    def fit(self,X,y=None):
        super(CodificadorCategorico,self).fit(X)
    
    def transform(self,X,y=None):
        return super(CodificadorCategorico,self).transform(X)
    
    def fit_transform(self,X,y=None):
        return super(CodificadorCategorico,self).fit(X).transform(X)
    
data.head()


# In[2]:


from sklearn.linear_model import LogisticRegression

col_numerica=['col_inexistente1','col2','col3','col_outliers','col_outliers2']
col_categorica=['col_categorica']
col_ordinal=['col_ordinal']
col_texto=['col_texto']


# In[4]:


imputador = SimpleImputer(missing_values=np.nan,strategy='mean')
escalador = preprocessing.StandardScaler()

transformador_ordinal=CodificadorCategorico()
transformador_categorico=BinarizadorCategorico()
transformador_texto=feature_extraction.text.TfidfVectorizer()

estimador=LogisticRegression()


# In[5]:


transformador_categorico.fit_transform(data.col_categorica)


# In[6]:


preprocessing.OneHotEncoder().fit_transform(transformador_ordinal.fit_transform(data[col_categorica]).reshape(1000,1)).toarray()


# In[7]:


data.col_categorica


# In[8]:


from sklearn.pipeline import Pipeline

transformador_numerico=Pipeline(
    [('imputador',imputador),('escalador',escalador)])

transformador_numerico.fit_transform(data[col_numerica])


# Para eliminar  el problema de no poder pasar un dataFrame en el transformador se crea un transformador customizado

# In[12]:


from sklearn.base import TransformerMixin,BaseEstimator
from scipy.sparse import issparse

class Transformador_Base(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X

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
        self.is_fitted = True
        return self
    
    def fit_transform(self,X,y=None):
        return self.transform(X=X,y=y)

class ColumExtractor(TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
    
    def transform(self,X,**transform_params):
        return X[self.columns]
    
    def fit(self,X,y=None,**fit_params):
        return self


# In[13]:


cext=ColumExtractor(columns=col_numerica)


# In[14]:


cext.fit_transform(data)


# In[15]:


pipeline_numerico=Pipeline([
    ['selector_numerico',ColumExtractor(columns=col_numerica)],
    ['transformador_numerico',transformador_numerico]
])
pipeline_numerico.fit_transform(data)[:5]


# In[16]:


pipeline_texto=Pipeline([
    ['selector_texto',ColumExtractor(columns=col_texto)],
    ['transformador_dim1',preprocessing.FunctionTransformer(lambda x: x[:,0],validate=False)],
    ['transformador_texto',transformador_texto],
    ['text_array',DenseTransformer()]
])
pipeline_texto.fit_transform(data)[:5]


# In[14]:


pipeline_categorico=Pipeline([
    ['selector_categorico',ColumExtractor(columns=col_categorica)],
    ['transformador_categorico',transformador_categorico]
])
pipeline_categorico.fit_transform(data)[:5]


# In[15]:


pipeline_ordinal=Pipeline([
    ['selector_ordinal',ColumExtractor(columns=col_ordinal)],
    ['transformador_dism1',preprocessing.FunctionTransformer(lambda x: x[:,0],validate=False)],
    ['transformador_ordinal',transformador_ordinal],
    ['transformador_dism2',preprocessing.FunctionTransformer(lambda x: np.vstack(x[:]),validate=False)],
])
pipeline_ordinal.fit_transform(data)[:5]


# Para ajuntar todo los pipeline se hace lo siguiente

# In[16]:


from sklearn.pipeline import FeatureUnion

pipeline_procesado=FeatureUnion([
    ('variables_numericas',pipeline_numerico),
    ('variables_ordinales',pipeline_ordinal),
    ('variable_categoricas',pipeline_categorico),
    ('variable_texto',pipeline_texto),
])
pipeline_procesado.fit_transform(data)


# Para estimar se hace lo siguiente

# In[17]:


pipeline_estimador=Pipeline([
    ('procesado',pipeline_procesado),
    ('estimador',estimador)
])
pipeline_estimador.fit(data,data.objetivo)


# In[18]:


pipeline_estimador.predict(data)[:5]


# Haciendo validacion cruzada

# In[19]:


from sklearn.model_selection import cross_val_score

cross_val_score(pipeline_estimador,X=data.drop('objetivo',axis=1),y=data.objetivo, scoring='roc_auc',cv=5)


# In[20]:


from sklearn.pipeline import make_pipeline,make_union

pipeline_simple = make_pipeline(
    make_union(
    pipeline_numerico,
    pipeline_ordinal,
    pipeline_categorico,
    ),
    estimador
)
pipeline_simple


# In[20]:


cross_val_score(pipeline_simple,X=data.drop('objetivo',axis=1),y=data.objetivo,scoring='roc_auc',cv=5)


# Biblioteca de sklearn unida a pandas hace todo lo anterior masrapido y facil
# %pip install sklearn-pandas
