#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=[10,10]
np.random.seed(42)

ames=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/ames.csv').drop('id_parcela',axis=1)
ames.shape


# In[2]:


ames.head()


# In[3]:


variable_ind=ames.drop('precio_venta',axis=1).columns
variable_obj=['precio_venta']
datos_num=ames[variable_ind].select_dtypes(np.number)
col_no_num=ames[variable_ind].select_dtypes(include=['object']).columns
col_numericas=datos_num.columns

dict_var_ordinales={
     "calidad_cocinas": ["Po", "Fa", "TA", "Gd", "Ex"],
    "funcionalidad":["Sal", "Sev", "Maj2", "Maj1", "Min2", "Min1","Typ"],
    "calidad_chimeneas":["NA","Po","Fa","TA","Gd","Ex"],
    "acabado_garaje":["NA","Unf","RFn","Fin"],
    "calidad_garaje":["NA","Po","Fa","TA","Gd","Ex"],
    "condicion_garaje":["NA","Po","Fa","TA","Gd","Ex"],
    "acceso_garaje_pavimentado":["N", "P", "Y"],
    "calidad_piscina":["NA","Fa","TA","Gd","Ex"],
    "calidad_valla":["NA","MnWw","GdWo","MnPrv","GdPrv"],
    "forma_parcela":["IR3", "IR2", "IR1","Reg"],
    "tipo_instalaciones":["ELO","NoSeWa","NoSewr","AllPub"],
    "pendiente_parcela":["Sev", "Mod", "Gtl"],
    "calidad_material_exterior":["Po","Fa","TA","Gd","Ex"],
    "condicion_material_exterior":["Po","Fa","TA","Gd","Ex"],
    "altura_sotano":["NA","Po","Fa","TA","Gd","Ex"],
    "condicion_sotano":["NA","Po","Fa","TA","Gd","Ex"],
    "sotano_exterior":["NA","No","Mn","Av","Gd"],
    "calidad_sotano_habitable1":["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "calidad_sotano_habitable2":["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
     "calidad_calefaccion":["Po","Fa","TA","Gd","Ex"],
}
col_ordinales=list(dict_var_ordinales.keys())
datos_ordinales=ames[col_ordinales]
col_categoricas=list(set(col_no_num) - set(col_ordinales))
datos_categoricos=ames[col_categoricas]


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
import category_encoders 

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns, output_type="matrix"):
        self.columns = columns
        self.output_type = output_type

    def transform(self, X, **transform_params):
        if isinstance(X, list):
            X = pd.DataFrame.from_dict(X)
        if self.output_type == "matrix":
            return X[self.columns].as_matrix()
        elif self.output_type == "dataframe":
            return X[self.columns]
        raise Exception("output_type tiene que ser matrix o dataframe")
        
    def fit(self, X, y=None, **fit_params):
        return self
        
    def fit(self, X, y=None, **fit_params):
        return self


# In[5]:


pipeline_categorico=Pipeline([
    ('selector_categorias',ColumnExtractor(columns=col_categoricas, output_type='dataframe')),
    ('transformador_categorico',category_encoders.OneHotEncoder(cols=col_categoricas,handle_unknown='value')),
])


# In[6]:


pipeline_categorico.fit(ames.head())
pipeline_categorico.transform(ames.head(10))


# In[7]:


pipeline_numerico=Pipeline([
    ['selector_numerico',ColumnExtractor(columns=col_numericas)],
    ['transformador_numerico',Pipeline([
        ('imputador_numerico',SimpleImputer(missing_values=np.nan,strategy='mean')),
        ('scalador_numerico',StandardScaler())]),
    ]
])


# In[8]:


niveles_ordinales =[]
for col, levels in dict_var_ordinales.items():
    niveles_ordinales.append({"col": col, "mapping": dict(zip(levels, range(len(levels))))})
niveles_ordinales


# In[9]:


pipeline_ordinal =Pipeline(
    [
      ("selector_ordinal", ColumnExtractor(columns=col_ordinales, output_type="dataframe")),
      ('transformador_ordinal',category_encoders.ordinal.OrdinalEncoder(mapping=niveles_ordinales,handle_unknown='value')),
])


# In[10]:


pipeline_ordinal.fit_transform(ames.head())


# In[11]:


pipeline_procesado=FeatureUnion(
    [('variables_numericas',pipeline_numerico),
    ('variables_ordinales',pipeline_ordinal),
    ('variables_categoricas',pipeline_categorico)]
)


# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA,TruncatedSVD

pipeline_estimador=Pipeline([
    ('procesado_variables',pipeline_procesado),
    ('reducir_dim',PCA()),
    ('estimador',RandomForestRegressor())
])


# In[13]:


pipeline_estimador.fit(X=ames,y=ames[variable_obj])


# In[14]:


pipeline_estimador.predict(X=ames.head())


# In[15]:


pipeline_estimador.get_params().keys()


# ### Haciendo una busqueda especificada de iperparametros

# In[16]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

param_dist_random = {
    "estimador__max_depth": [3, None],
    "estimador__max_features": sp_randint(1, 11),
    "estimador__min_samples_split": sp_randint(2, 11),
    "estimador__min_samples_leaf": sp_randint(1, 11),
    "estimador__bootstrap": [True, False],
    "estimador__n_estimators": np.linspace(10,1000,10).astype(int),
    "estimador__min_impurity_decrease": np.logspace(-1, 0.0),
    "reducir_dim": [PCA(), TruncatedSVD()],
    "reducir_dim__n_components": sp_randint(10, 50)
}

busqueda_random = RandomizedSearchCV(
    estimator=pipeline_estimador, 
    param_distributions=param_dist_random,
    cv=5, 
    refit=True,
    scoring="neg_mean_squared_error", n_jobs=-1,
    return_train_score=True,
    n_iter=100)


# In[17]:


import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore',category=DataConversionWarning)


# In[18]:


busqueda_random.fit(X=ames.drop(variable_obj,axis=1),y=ames[variable_obj])


# In[19]:


busqueda_random.predict(ames.head())


# In[20]:


busqueda_random.best_estimator_.steps


# In[21]:


np.min(np.sqrt(np.abs(busqueda_random.cv_results_['mean_test_score'])))


# In[22]:


from sklearn.externals import joblib

joblib.dump(busqueda_random,'pipeline_ames.pkl')


# In[23]:


clf=joblib.load('pipeline_ames.pkl')


# In[24]:


clf.predict(ames.head())


# ### Guardando las columnas del data set de entrenamiento

# In[25]:


import json

with open('columnas_ames.json','w') as fname:
    ames_columnas=ames.columns.tolist()
    json.dump(ames_columnas,fname)


# ### Guardando las variables del data set de entrenamiento

# In[26]:


ames_dtypes= ames.dtypes
ames_dtypes={col:ames[col].dtype for col in ames.columns}
joblib.dump(ames_dtypes,'dtypes_ames.pkl')


# ### Ejemplo de una observacion como prueba

# In[27]:


nueva_observacion=ames.to_dict(orient='record')[0]
nueva_observacion


# In[28]:


obs={
    'precio_venta':215000,
    'sistema_electrico':'SBrkr',
    'sotano_exterior':'Gd',
    'tipo_acceso':'Pave',
    'tipo_andamios':'CBlock',
    'tipo_calefaccion':'GasA',
    'tipo_casa':'1Story',
    'tipo_construccion':20,
    'tipo_edificio':'1Fam',
    'tipo_garaje':'Attchd',
    'tipo_instalaciones':'AllPub',
    'tipo_revestimiento':'Stone',
    'tipo_tejado':'Hip',
    'tipo_venta':'WD',
    'tipo_zona':'RL',
    'valor_atributo_miscelaneo':0
}


# In[36]:


def dict_a_df(obs,columnas,dtypes):
    obs_df=pd.DataFrame([obs])
    for col,dtype in dtypes.items():
        if col in obs_df.columns:
            obs_df[col]=obs_df[col].astype(dtype)
        else:
            obs_df[col]=None
    return obs_df

obs_df=dict_a_df(obs,ames_columnas,ames_dtypes)


# In[37]:


obs_df


# In[38]:


pipeline_estimador.predict(obs_df)

