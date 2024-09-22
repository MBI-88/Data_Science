#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[12,12]
np.random.seed(42)

datos_df=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/salario_censo.csv')
datos_df.head()


# In[2]:


datos_df.shape


# # Pipeline de procesamiento  de datos

# In[3]:


variable_obj= 'objetivo'
variable_inde= datos_df.drop(variable_obj,axis=1).columns
censo_X=datos_df[variable_inde]
censo_y=datos_df[variable_obj]
censo_y.unique()


# # Convertimos las variables objetivos que son texto a numericas 

# In[4]:


censo_y=censo_y.replace({' <=50K':0,' >50K':1})
censo_y.unique()


# In[5]:


datos_numericos=censo_X.select_dtypes(np.number)
col_numericas=datos_numericos.columns

datos_categoricos=censo_X.select_dtypes(include=['object'])
col_categoricas=datos_categoricos.columns


# In[6]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MultiLabelBinarizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion

class ColumExtractor(TransformerMixin):
    def __init__(self,columns):
        self.columns=columns
    
    def transform(self,X,**transform_params):
        return X[self.columns].as_matrix()
    
    def fit(self,X,y=None,**fit_params):
        return self

class BinarizadorMultipleCategoria(MultiLabelBinarizer):
    def fit(self,X,y=None):
        super(BinarizadorMultipleCategoria,self).fit(X)
        
    def transform(self,X,y=None):
        return super(BinarizadorMultipleCategoria,self).transform(X)
    
    def fit_transform(self,X,y=None):
        return super(BinarizadorMultipleCategoria,self).fit(X).transform(X)
    
pipeline_numerico=Pipeline([
    ('selector_numerico',ColumExtractor(columns=col_numericas)),
    ('imputador',SimpleImputer()),
    ('codificador_numerico',StandardScaler()),
])

pipeline_categorico=Pipeline([
    ('selector_categorico',ColumExtractor(columns=col_categoricas)),
    ('codificador_categorico',BinarizadorMultipleCategoria()),
])


# In[7]:


pipeline_categorico.fit_transform(censo_X).shape


# In[8]:


pipeline_numerico.fit_transform(censo_X).shape


# In[9]:


pipeline_procesado=FeatureUnion([
    ('transformador_numerico',pipeline_numerico),
    ('transformador_categorico',pipeline_categorico),
],n_jobs=-1)

censo_X_procesado=pipeline_procesado.fit_transform(censo_X)
censo_X_procesado.shape


# # Para ver el valance de el  dataset

# In[10]:


censo_y.value_counts(True)# se usa el roc_auc que le importa poco que este invalanciado


# In[11]:


from sklearn.model_selection import cross_validate

resultados={}
def evaluar_modelo(estimador,X,y):
    resultados_estimador=cross_validate(estimador,X,y,scoring='roc_auc',n_jobs=-1,cv=5,return_train_score=True)
    return resultados_estimador

def ver_resultados():
    resultados_df=pd.DataFrame(resultados).T
    for col in resultados_df:
        resultados_df[col]=resultados_df[col].apply(np.mean)
        resultados_df[col+'_idx']=resultados_df[col]/resultados_df[col].max()
        return resultados_df


# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

resultados['reg_logistica']=evaluar_modelo(LogisticRegression(),censo_X_procesado,censo_y)
resultados['naive_bayes']=evaluar_modelo(GaussianNB(),censo_X_procesado,censo_y)
resultados['rf']=evaluar_modelo(RandomForestClassifier(),censo_X_procesado,censo_y)
resultados['SVC']=evaluar_modelo(SVC(),censo_X_procesado,censo_y)
ver_resultados()


# In[13]:


estimador_rf=RandomForestClassifier()#el que mejor funciona es Svc pero el termino medio seria rf por tardar menos


# In[14]:


get_ipython().run_cell_magic('timeit', '', 'import time\ndef foo():\n    time.sleep(1)')


# In[15]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 1 # se usa para ejecutar un numero de veces la celda en que esta', 'def foo():\n    time.sleep(1)')


# In[16]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV#generalmente se usa el segundo,en datos pequeños el 1
print(estimador_rf.__doc__)


# In[17]:


estimador_rf.get_params()# muestra los valores for defecto que se pueden cambiar


# In[18]:


#Para variar parapetros se procese de esta manera
parametros_busqueda={
    'criterion':['gini','entropy'],
    'n_estimators':np.linspace(10,1000,10).astype(int),
    'class_weight':[None,'balanced']
}


# In[19]:


grid=GridSearchCV(estimator=estimador_rf,param_grid=parametros_busqueda,scoring='roc_auc',n_jobs=-1)


# In[20]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'grid.fit(censo_X_procesado,censo_y)')


# In[21]:


print(grid.best_score_)
print(grid.best_estimator_)


# In[22]:


pd.DataFrame(grid.cv_results_,).sort_values(by='rank_test_score')


# In[23]:


resultados['rf_gridsea']=evaluar_modelo(grid.best_estimator_,censo_X_procesado,censo_y)


# # Usando el segundo metodo de budqueda

# In[24]:


busqueda_random=RandomizedSearchCV(estimator=estimador_rf,param_distributions=parametros_busqueda,scoring='roc_auc',n_jobs=-1,n_iter=10)


# In[25]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'busqueda_random.fit(censo_X_procesado,censo_y)')


# In[26]:


print(busqueda_random.best_score_)
print(busqueda_random.best_estimator_)


# In[27]:


resultados['rf_randomizedsearch']=evaluar_modelo(busqueda_random.best_estimator_,censo_X_procesado,censo_y)


# In[28]:


#para la busquesa RandomizedSearchCV se usa scipy.stats randint que da una distribucion al azar para variables int
from scipy.stats import randint as sp_randint

param_dist_random={
    'max_depth':[3,None],
    'max_features':sp_randint(1,11),
    'min_samples_split':sp_randint(2,11),
    'min_samples_leaf':sp_randint(1,11),
    'bootstrap':[True,False],
    'criterion':['gini','entropy'],
    'n_estimators':np.linspace(10,1000,10).astype(int),
}


# In[29]:


busqueda_random_100=RandomizedSearchCV(estimator=estimador_rf,param_distributions=param_dist_random,scoring='roc_auc',n_jobs=-1,n_iter=100)


# In[30]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'busqueda_random_100.fit(censo_X_procesado,censo_y)')


# In[31]:


print(busqueda_random_100.best_score_)
print(busqueda_random_100.best_estimator_)


# In[33]:


resultados['rf_randomizedsearch_100']=evaluar_modelo(busqueda_random_100.best_estimator_,censo_X_procesado,censo_y)


# # Optimizacion de parametros en Pipeline

# In[36]:


busqueda_random_10=RandomizedSearchCV(estimator=estimador_rf,param_distributions=param_dist_random,scoring='roc_auc',n_jobs=-1,n_iter=10)

pipeline_estimador=Pipeline([
    ('procesado',pipeline_procesado),
    ('estimador',busqueda_random_10)
])
pipeline_estimador.fit(censo_X,censo_y)
pipeline_estimador.predict(censo_X)


# # Bibliotecas externas de optimizacion

# In[69]:


from skopt import gp_minimize,space,BayesSearchCV

param_epacio_skopt=[
    space.Integer(3,10),#max_depth
    space.Integer(1,11),#max_features
    (0.001,0.99),#min_sample_split
    (0.001, 0.5),#min_sample_leaf
    space.Integer(1,1000),#n_estimators
    space.Categorical(['gini','entropy']),#criterion
    space.Categorical([True,False])#bootstrap
]


# In[72]:


from sklearn.model_selection import cross_val_score

estimador_rf=RandomForestClassifier()
param_espacio_skop_bayesCV={
    'max_depth':space.Integer(3,10),
    'max_features':space.Integer(1,11),
    'min_samples_split':space.Real(0.001,0.99,'uniform'),
    'min_samples_leaf':space.Real(0.001,0.5,'uniform'),
    'n_estimators':space.Integer(1,1000),
    'criterion':space.Categorical(['gini','entropy']),
    'bootstrap':space.Categorical([True,False])
}

busqueda_bayesiano_skopt_100=BayesSearchCV(
    estimator=estimador_rf,
    search_spaces=param_espacio_skop_bayesCV,
    scoring='roc_auc',
    n_iter=100,n_jobs=-1,
    random_state=42
)


# In[73]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 1', 'busqueda_bayesiano_skopt_100.fit(censo_X_procesado,censo_y)')


# In[87]:


busqueda_bayesiano_skopt_100.best_estimator_.get_params()


# In[76]:


resultados['bayesiano_skopt_100']=evaluar_modelo(busqueda_bayesiano_skopt_100.best_estimator_,censo_X,censo_y)


# In[79]:


param_espacio_skop=[
    space.Integer(3,10),
    space.Integer(1,11),
    space.Real(0.001,0.99,'uniform'),
    space.Real(0.001,0.5,'uniform'),
    space.Integer(1,1000),
    space.Categorical(['gini','entropy']),
    space.Categorical([True,False])
]

def funcion_skopt(params):
    max_depth,max_features,min_sample_split,min_sample_leaf,n_estimators,criterion,bootstrap=params
    
    estimador_rf.set_params(
    max_depth=max_depth,
    max_features=max_features,
    min_samples_split=min_sample_split,
    min_samples_leaf=min_sample_leaf,
    n_estimators=n_estimators,
    criterion=criterion,
    bootstrap=bootstrap
    )
    return -np.mean(cross_val_score(estimador_rf,censo_X_procesado,censo_y,cv=5,n_jobs=-1,scoring='roc_auc'))


# In[80]:


resultado_gp=gp_minimize(funcion_skopt,param_espacio_skop,n_calls=100,random_state=42)


# In[81]:


estimador_skopt_gp_100=RandomForestClassifier(
    max_depth=resultado_gp.x[0],
    max_features=resultado_gp.x[1],
    min_samples_split=resultado_gp.x[2],
    min_samples_leaf=resultado_gp.x[3],
    n_estimators=resultado_gp.x[4],
    criterion=resultado_gp.x[5],
    bootstrap=resultado_gp.x[6]
)


# In[84]:


resultados['gp_skopt']=evaluar_modelo(estimador_skopt_gp_100,censo_X,censo_y)


# In[85]:


ver_resultados()

