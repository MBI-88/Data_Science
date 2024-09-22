#!/usr/bin/env python
# coding: utf-8

# ### Seleccion de variables

# In[1]:


from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[10,10]

np.random.seed(42)
ames=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/ames.csv').drop(['id_parcela'],axis=1)
ames.head()


# In[2]:


ames.dtypes


# In[3]:


ames.sample(1).T


# In[4]:


ames[ames.duplicated()].shape


# In[5]:


ames.precio_venta.describe()


# In[6]:


ames.to_csv('ames_modificado_1.csv',index=False)


# In[7]:


ames=pd.read_csv('ames_modificado_1.csv')
ames.head()


# In[8]:


ames.shape


# ### Procesado de datos

# In[9]:


variable_ind=ames.drop('precio_venta',axis=1).columns
variable_obj=['precio_venta']
datos_numericos=ames[variable_ind].select_dtypes(np.number)
col_no_num=ames[variable_ind].select_dtypes(include=['object']).columns
col_numericas=datos_numericos.columns

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


# In[10]:


datos_categoricos.head()


# In[11]:


datos_ordinales.head()


# In[12]:


datos_numericos.head()


# ### Para asegurar que todas las columnas estan clasificadas como categoricas o numericas

# In[13]:


[col for col in ames.columns if col not in datos_numericos.columns and col not in datos_categoricos.columns and col not in datos_ordinales.columns]


# In[39]:


from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
datos_numericos_imputados_normalizados=pd.DataFrame(
    normalize(SimpleImputer(missing_values=np.nan,strategy='median').fit_transform(datos_numericos)),
    columns=col_numericas
)


# In[40]:


datos_categoricos_dummy=pd.get_dummies(datos_categoricos,drop_first=True)
datos_categoricos_dummy.shape


# In[16]:


for columna_ordinal,valores in dict_var_ordinales.items():
    datos_ordinales[columna_ordinal]=(
    datos_ordinales[columna_ordinal].astype('category').cat.set_categories(valores).cat.codes
    )
datos_ordinales.head()


# In[17]:


ames_procesado=pd.concat(
    [datos_numericos_imputados_normalizados,datos_categoricos_dummy,datos_ordinales
        
    ],axis=1
)
ames_procesado.head()


# In[18]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ames_X = ames_procesado
ames_Y = ames[variable_obj]

def rmse(y_real,y_pred):
    return np.sqrt(mean_squared_error(y_real,y_pred))

def rmse_cv(estimator,X,Y):
    y_pred=estimator.predict(X)
    return rmse(Y,y_pred)

res=cross_validate(LinearRegression(),ames_procesado,ames[variable_obj],scoring=rmse_cv,n_jobs=-1,cv=10)

res


# In[19]:


def evaluacion_modelo(estimator,X,Y):
    resultados_estimador=cross_validate(estimator,X,Y,scoring=rmse_cv,n_jobs=-1,cv=10,return_train_score=True)
    return resultados_estimador
resultados={}

def ver_resultados():
    resultados_df=pd.DataFrame(resultados).T
    resultados_cols=resultados_df.columns
    for col in resultados_df:
        resultados_df[col]=resultados_df[col].apply(np.mean)
        resultados_df[col + '_indx']=resultados_df[col]/resultados_df[col].min()
    return resultados_df

resultados["reg_lineal_sin_seleccion"]=evaluacion_modelo(LinearRegression(),ames_X,ames_Y)
resultados["svr_sin_seleccion"]=evaluacion_modelo(SVR(),ames_X,ames_Y)
resultados["rf_sin_seleccion"]=evaluacion_modelo(RandomForestRegressor(),ames_X,ames_Y)
ver_resultados()


# ### Metodos de filtrado

# In[20]:


from sklearn.feature_selection import SelectKBest, f_regression
selector_kbest10=SelectKBest(f_regression,k=10,)
ames_X_kbest10=selector_kbest10.fit_transform(ames_X,ames_Y)
ames_X_kbest10.shape


# In[21]:


selector_kbest10.get_support()


# ### Seleccion de las columnas mas importantes port el selector kbest

# In[22]:


columnas_seleccion_kbest10=ames_X.loc[:,selector_kbest10.get_support()].columns
columnas_seleccion_kbest10


# In[23]:


selector_kbest10.scores_[:10]


# In[24]:


puntuciones_selector_kbest10=zip(ames_X.columns,selector_kbest10.scores_,selector_kbest10.get_support())
evaluacion_kbest10=sorted(filter(lambda c: c[2],puntuciones_selector_kbest10),
                         key=lambda c: c[1],reverse=True)

list(evaluacion_kbest10)


# In[25]:


resultados['reg_lineal_kbest10']=evaluacion_modelo(LinearRegression(),ames_X_kbest10,ames_Y)
resultados['rf_kbest10']=evaluacion_modelo(RandomForestRegressor(),ames_X_kbest10,ames_Y)
resultados['svr_kbest10']=evaluacion_modelo(SVR(),ames_X_kbest10,ames_Y)


# In[26]:


ver_resultados()


# ### Usando 50 variables

# In[27]:


selector_kbest50=SelectKBest(f_regression,k=50)
ames_X_kbest50=selector_kbest50.fit_transform(ames_X,ames_Y)
resultados['reg_lineal_kbest50']=evaluacion_modelo(LinearRegression(),ames_X_kbest50,ames_Y)
resultados['rf_kbest50']=evaluacion_modelo(RandomForestRegressor(),ames_X_kbest50,ames_Y)
resultados['svr_kbest50']=evaluacion_modelo(SVR(),ames_X_kbest50,ames_Y)
ver_resultados()


# In[28]:


from sklearn.feature_selection import SelectPercentile
selector_pct10=SelectPercentile(f_regression,percentile=10)
ames_X_pct10=selector_pct10.fit_transform(ames_X,ames_Y)
ames_X_pct10.shape


# ### Metodos Envolventes (wrapper methods)

# In[29]:


from sklearn.feature_selection import RFE
estimador_seleccion=RandomForestRegressor()
selector_rfe10=RFE(estimador_seleccion,n_features_to_select=10)
ames_X_rfe10_rf=selector_rfe10.fit_transform(ames_X,ames_Y)
ames_X_rfe10_rf.shape


# In[30]:


evaluacion_rfe10_rf=sorted(
    filter(lambda c: c[2],zip(
        ames_X.columns,
        selector_rfe10.ranking_,
        selector_rfe10.get_support()
    )
          ),key=lambda c: c[1],reverse=True
)
evaluacion_rfe10_rf


# In[31]:


resultados['reg_lineal_rfe10']=evaluacion_modelo(LinearRegression(),ames_X_rfe10_rf,ames_Y)
resultados['rf_rfe10_rf']=evaluacion_modelo(RandomForestRegressor(),ames_X_rfe10_rf,ames_Y)
resultados['svr_rfe10_rf']=evaluacion_modelo(SVR(),ames_X_rfe10_rf,ames_Y)


# In[32]:


estimador_selector=LinearRegression()
selector_rfe10_lineal=RFE(estimador_seleccion,n_features_to_select=10)
ames_X_rfe10_lineal=selector_rfe10_lineal.fit_transform(ames_X,ames_Y)


# In[33]:


evaluacion_rfe10_lineal=sorted(
    filter(lambda c:c[2],zip(
        ames_X.columns,
        selector_rfe10_lineal.ranking_,
        selector_rfe10_lineal.get_support()
    ))
)
evaluacion_rfe10_lineal


# In[34]:


set(evaluacion_rfe10_rf).intersection(set(evaluacion_rfe10_lineal))


# In[35]:


resultados['reg_lineal_rfe10_lineal']=evaluacion_modelo(LinearRegression(),ames_X_rfe10_lineal,ames_Y)
resultados['rf_rfe10_lineal']=evaluacion_modelo(RandomForestRegressor(),ames_X_rfe10_lineal,ames_Y)
resultados['svr_rfe10_lineal']=evaluacion_modelo(SVR(),ames_X_rfe10_lineal,ames_Y)
ver_resultados()


# In[36]:


from sklearn.feature_selection import RFECV
estimador_selector=RandomForestRegressor()
selector_rfecv=RFECV(estimador_selector,cv=5,n_jobs=-1)
ames_X_rfcev=selector_rfecv.fit_transform(ames_X,ames_Y)
ames_X_rfcev.shape


# In[38]:


resultados['reg_lineal_rfecv_rf']=evaluacion_modelo(LinearRegression(),ames_X_rfcev,ames_Y)
resultados['rf_rfecv']=evaluacion_modelo(RandomForestRegressor(),ames_X_rfcev,ames_Y)
resultados['svr_rfecv_rf']=evaluacion_modelo(SVR(),ames_X_rfcev,ames_Y)
ver_resultados()


# In[ ]:




