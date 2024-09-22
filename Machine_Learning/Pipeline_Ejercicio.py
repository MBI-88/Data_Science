#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import feature_extraction,preprocessing,metrics,impute
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

movie=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/movies_New1.csv')

class ColumExtractor(TransformerMixin,BaseEstimator):
    def __init__(self,columns,output_type='matrix'):
        self.columns=columns
        self.output_type=output_type
    
    def transform(self,X,**transform_params):
        if isinstance(X,list):
            X=pd.DataFrame.from_dict(X)
        if self.output_type=='matrix':
            return X[self.columns].values
        
        elif self.output_type=='dataframe':
            return X[self.columns]
        
        raise Exception('output_type tiene que ser matrix o dataframe')
    
    def fit(self,X,y=None,**fit_params):
        return self
    
variable_objetivo="ventas"
col_num=movie.drop(columns=variable_objetivo).select_dtypes(np.number).columns
estimador = LinearRegression()
imputador = impute.SimpleImputer(missing_values=np.nan,strategy='mean')
escalador=preprocessing.StandardScaler()

pipeline_processado=make_pipeline(
    ColumExtractor(columns=col_num),
    imputador,
    escalador
)
valor_procesado=pipeline_processado.fit_transform(movie)


# In[2]:


movie_ventas=movie[movie[variable_objetivo].notnull()]
pipeline_estimador=make_pipeline(
    pipeline_processado,
    estimador
)
pipeline_estimador.fit(movie_ventas.drop(columns=variable_objetivo),movie_ventas[variable_objetivo])
pred=pipeline_estimador.predict(movie_ventas.drop(columns=variable_objetivo))
evaluacion=cross_val_score(pipeline_estimador,movie_ventas,movie_ventas[variable_objetivo],scoring='neg_mean_absolute_error',cv=3)


# In[3]:


df_resultado_movie=movie_ventas
df_resultado_movie['ventas_pred']=pred
df_resultado_movie['MAE_media']=evaluacion.mean()
df_resultado_movie[['ventas','ventas_pred','MAE_media']].head()

