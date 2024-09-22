#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Parte 1 Organizativa
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize']=(12,12)
vehiculos= pd.read_csv("C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/071 vehiculos-original.csv")


# In[2]:


vehiculos.head()


# In[3]:


vehiculos=vehiculos.rename(columns={
    "cylinders":"cilindros",
    "trany":"transmision",
    "make":"fabricante",
    "model":"modelo",
    "displ":"desplazaminto",
    "drive":"traccion",
    "VClass":"calse",
    "fuelType":"combustible",
    "comb08":"consumo",
    "co2TailpipeGpm":"co2"
}

)
vehiculos.head()


# In[4]:


# Descripcion de entidaes
vehiculos.to_csv("vehiculos-1.071 vehiculos-original.csv",index=False)


# In[5]:


#Parte 2 Deteccion de errores
#Proceso de deteccion de errores
vehiculos=pd.read_csv("vehiculos-1.071 vehiculos-original.csv")
vehiculos.shape


# In[6]:


vehiculos["modelo_unico"]=vehiculos.fabricante.str.cat([vehiculos.modelo,vehiculos.year.apply(str)],sep='-')
vehiculos.modelo_unico.value_counts()


# In[7]:


vehiculos[vehiculos.modelo_unico=='Chevrolet-C10 Pickup 2WD-1984'].head()


# In[8]:


vehiculos[vehiculos.duplicated()].shape


# In[9]:


vehiculos=vehiculos.drop_duplicates()
vehiculos.shape


# In[10]:


del vehiculos['modelo_unico']


# In[11]:


n_valores= len(vehiculos)
def valores_duplicados(df):
    for columna in df:
        valores_columna= df[columna].value_counts()
        mas_comun= valores_columna.iloc[0]
        menos_comun= valores_columna.iloc[-1]
        print("{} | {}-{} | {}".format(df[columna].name,
        round(mas_comun/(1.0*n_valores),3),
        round(menos_comun/(1.0*n_valores),3),df[columna].dtype))
        
        
valores_duplicados(vehiculos)


# In[12]:


vehiculos.traccion.value_counts(normalize=True)


# In[13]:


vehiculos.transmision.value_counts(normalize=True).plot.barh();


# In[14]:


vehiculos.cilindros.value_counts(normalize=True).plot.hist();


# In[15]:


vehiculos.combustible.value_counts(normalize=True).plot.barh();


# In[ ]:


valore=len(vehiculos)
def valores_inexistente_col(df):
    for columna in df:
        print("{} | {} | {} ".format(df[columna].name,len(df[df[columna].isnull()])/(1.0*valore),
                                     df[columna].dtype))

valores_inexistente_col(vehiculos)


# In[17]:


#Parte 3 Agrupacion de variables
import scipy.stats as sc
import numpy as np
def outliers_col(df):
    for columna in df:
        if vehiculos[columna].dtype != np.object:
            n_outliers= len(vehiculos[np.abs(sc.zscore(vehiculos[columna]))>3])
            print("{} | {} | {} ".format(df[columna].name,n_outliers,df[columna].dtype))

outliers_col(vehiculos)


# In[18]:


vehiculos.boxplot(column='consumo');


# In[19]:


vehiculos.boxplot(column='co2');


# In[20]:


vehiculos[vehiculos.co2==0].combustible.unique()


# In[21]:


vehiculos.combustible.unique()


# In[22]:


vehiculos_no_electricos= vehiculos[vehiculos.co2>0]


# In[23]:


outliers_col(vehiculos_no_electricos)


# In[24]:


valores_duplicados(vehiculos_no_electricos)


# In[25]:


valores_inexistente_col(vehiculos)


# In[26]:


vehiculos_no_electricos.to_excel("vehiculos2.Limpio_analisis.xls",index=False)


# In[27]:


vehiculos=pd.read_excel("vehiculos2.Limpio_analisis.xls")


# In[28]:


vehiculos.head()


# In[29]:


def valores_columna_unicos(df):
    for columna in df:
        print("{} | {} | {}".format(df[columna].name,len(df[columna].unique()),df[columna].dtype))
    

valores_columna_unicos(vehiculos)


# In[30]:


vehiculos.calse.unique()


# In[31]:


pequeño=["Compact Cars","Subcompact Cars","Two Seaters","Minicompact Cars"]
mediano=['Midsize Cars']
grandes=['Large Cars']
vehiculos.loc[vehiculos['calse'].isin(pequeño),'clase_tipo']='Coches Pequeños'
vehiculos.loc[vehiculos['calse'].isin(mediano),'clase_tipo']='Coches Medianos'
vehiculos.loc[vehiculos['calse'].isin(grandes),'clase_tipo']='Coches Grandes'
vehiculos.loc[vehiculos['calse'].str.contains('Truck'),'clase_tipo']='Camionetas'
vehiculos.loc[vehiculos['calse'].str.contains('Special Purpose'),'clase_tipo']='Vehiculos Especiales'
vehiculos.loc[vehiculos['calse'].str.contains('Sport Utility'),'clase_tipo']='Deportivos'
vehiculos.loc[vehiculos['calse'].str.contains('Station'),'clase_tipo']='Coche Familiar'
vehiculos.loc[(vehiculos['calse'].str.lower().str.contains('van')),'clase_tipo']='Furgoneta'


# In[32]:


vehiculos.clase_tipo=vehiculos.clase_tipo.astype('category')


# In[33]:


vehiculos.clase_tipo.value_counts()


# In[34]:


vehiculos.traccion.unique()


# In[35]:


vehiculos['traccion_tipo']='dos'
vehiculos.loc[vehiculos.traccion.isin(['4-Wheel or All-Wheel Drive','All-Wheel Drive','4-Wheel Drive','Part-time 4-Wheel Drive']),'traccion_tipo']='cuaatro'
vehiculos.traccion_tipo=vehiculos.traccion_tipo.astype('category')
vehiculos.traccion_tipo.value_counts()


# In[36]:


vehiculos.transmision.unique()


# In[37]:


vehiculos['transmision_tipo']="Automatico"
vehiculos['transmision_tipo'][(vehiculos['transmision'].notnull())&(vehiculos['transmision'].str.startswith('M'))]="Manual"
vehiculos.transmision_tipo=vehiculos.transmision_tipo.astype('category')
vehiculos.transmision_tipo.value_counts()


# In[38]:


vehiculos.combustible.value_counts()


# In[39]:


vehiculos['combustible_tipo']='Otros tipos de combustible'
vehiculos.loc[vehiculos['combustible']=='Regular','combustible_tipo']='Normal'
vehiculos.loc[vehiculos['combustible']=='Premium','combustible_tipo']='Premium'
vehiculos.loc[vehiculos['combustible'].str.contains('Electricity'),'combustible_tipo']='Hibrido'
vehiculos.combustible_tipo=vehiculos.combustible_tipo.astype('category')
vehiculos.combustible_tipo.value_counts()


# In[40]:


vehiculos.head(20)


# In[41]:


#Asignacion para variables continuas(se utiliza los quintiles para el trabajo)

tipos_tamaño_motor=['muy pequeño','pequeño','mediano','grande','muy grande']
vehiculos['tamaño_motor_tipo']=pd.qcut(vehiculos['desplazaminto'],5,tipos_tamaño_motor)
tipos_consumo=['muy bajo','bajo','medio','alto','muy alto']
vehiculos['tipos_consumo']=pd.qcut(vehiculos['consumo'],5,tipos_consumo)
tipos_co2=['muy bajo','bajo','medio','alto','muy alto']
vehiculos['tipos_co2']=pd.qcut(vehiculos['consumo'],5,tipos_co2)


# In[42]:


vehiculos.plot.scatter(x='consumo',y='co2');


# In[43]:


litros_por_galon=3.785
vehiculos['consumo_litros_milla']=litros_por_galon/vehiculos.consumo
vehiculos.plot.scatter(x='consumo_litros_milla',y='co2');


# In[44]:


tipo_consumo=['muy bajo','bajo','moderado','alto','muy alto']
vehiculos['consumo_tipo']=pd.qcut(vehiculos['consumo_litros_milla'],5,tipo_consumo)
vehiculos.consumo_tipo.head()


# In[45]:


vehiculos.head()


# In[46]:


vehiculos.dtypes


# In[47]:


#Parte 4 Distribucuion de variables 

vehiculos.to_pickle("vehiculos3.variable_agregada.pkl")
vehiculos =pd.read_pickle('vehiculos3.variable_agregada.pkl')
vehiculos.dtypes


# In[48]:


vehiculos['co2'].plot.hist();


# In[49]:


vehiculos['co2'].plot.kde();


# In[50]:


def distribucicon_variable_numerica(df,col):
    df[col].plot.kde()
    plt.xlabel('Distribucion  de variable {}'.format(col))
    plt.show()

columnas_numericas=vehiculos.select_dtypes(['int64','float64']).columns
from ipywidgets import interact, fixed
interact(distribucicon_variable_numerica,col=columnas_numericas,df=fixed(vehiculos));


# In[51]:


vehiculos['cilindros'].value_counts(normalize=True)


# In[52]:


from scipy import stats
def normalidad_variable_numerica(col):
    stats.probplot(vehiculos[col],plot=plt)
    plt.xlabel('Diagrama de Probabilidad(normal)de la  variable {}'.format(col))
    plt.show()

interact(normalidad_variable_numerica,col=columnas_numericas);
##nota: si la variable sigue la linea recta significa que sigue una distribucion normal 


# In[53]:


for num_col in columnas_numericas:
   _, pvl=stats.normaltest(vehiculos[num_col])
   if (pvl < 0.05):
        print('Columna {} no sigue una distribucion normal'.format(num_col))
#nota la funcion stats.normaltest(vehiculos[num_col]) devuelve las probabilidades de que 
# de que una  variable  arbitraria provenga de una distribucion normal se compara con 0.05
# para sefinir si el menor no proviene , caso contrario proviene


# In[54]:


# Distribucion de variables categoricas
def distribucion_categorica(col):
    vehiculos[col].value_counts(ascending=True,normalize=True).tail(20).plot.barh()
    plt.show()

columnas_categoricas= vehiculos.select_dtypes(['object','category']).columns
interact(distribucion_categorica,col=columnas_categoricas);


# In[55]:


# Parte 5 Comparaciones
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(13,13)
vehiculos=pd.read_pickle('vehiculos3.variable_agregada.pkl')
vehiculos.head()


# In[56]:


vehiculos.dtypes


# In[57]:


def pivot_recuento(df,rows,columns,calc_field):
    df_pivot=df.pivot_table(values=calc_field,index=rows,columns=columns,aggfunc=np.size).dropna(axis=0,how='all')
    return df_pivot

consumo_combustible= pivot_recuento(vehiculos,'combustible_tipo','consumo_tipo','year')
consumo_combustible


# In[58]:


def head_recuento_tipo(df,col1,col2):
    pivot_table=pivot_recuento(df,col1,col2,'year')
    sns.heatmap(pivot_table,annot=True,fmt='g')
    sns.plt.ylabel(col1)
    sns.plt.xlabel(col2)
    plt.show()


# In[59]:


interact(head_recuento_tipo,col1=vehiculos.columns,col2=vehiculos.columns,df=fixed(vehiculos))


# In[60]:


def media_por_categoria(col_grupo,col_calculo):
    vehiculos.groupby(col_grupo)[col_calculo].mean().plot.barh()
    plt.ylabel(col_grupo)
    plt.xlabel('Valores medios de {}'.format(col_calculo))
    plt.show()

columnas_numericas=vehiculos.select_dtypes(['int64','float64']).columns
columnas_categoricas=vehiculos.select_dtypes(['object','category']).columns
columnas_tipo=[col for col in vehiculos.columns if col.endswith('_tipo')]
interact(media_por_categoria,col_grupo=columnas_categoricas,col_calculo=columnas_numericas);


# In[61]:


def pivot_media(rows,columns,calc_field):
    df_pivot=vehiculos.pivot_table(values=calc_field,index=rows,columns=columns,aggfunc=np.mean).dropna(axis=0,how='all')
    return df_pivot

pivot_media('combustible_tipo','consumo_tipo','co2')


# In[62]:


def heatmap_medias_tipo(col1,col2,col3):
    pivot_table=pivot_media(col1,col2,col3)
    sns.heatmap(pivot_table,annot=True,fmt='g')
    sns.plt.ylabel(col1)
    sns.plt.xlabel(col2)
    plt.show()
    
interact(heatmap_medias_tipo,col1=vehiculos.columns,col2=vehiculos.columns,col3=columnas_numericas);


# In[63]:


vehiculos_pre_2017=vehiculos.query('year<2017')
vehiculos_pre_2017.groupby('year')['co2'].mean().plot();


# In[64]:


def evolucion_media(col_calculo):
    vehiculos_pre_2017.groupby('year')[col_calculo].mean().plot()
    plt.ylabel(col_calculo)
    plt.show()

interact(evolucion_media,col_calculo=columnas_numericas);


# In[65]:


#Util para ver tendencias
def evolucion_recuento(col_calculo):
    for categoria in vehiculos_pre_2017[col_calculo].unique():
        n_vehiculos_categoria_año=vehiculos_pre_2017[vehiculos_pre_2017[col_calculo]==categoria].groupby('year').apply(np.size)
        plt.plot(n_vehiculos_categoria_año.index,
        n_vehiculos_categoria_año,
        label=categoria
    )     
    plt.legend()

interact(evolucion_recuento,col_calculo=columnas_categoricas);


# In[66]:


"""Parte 6 Herramientas adicionales 
   Necesista instalar pandas_profiling con conda install -y pandas-profiling
   Necesita instalar missigno para analizar datos faltantes"""


# In[67]:


#Parte 7 procesamiento de datos
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(12,12)
vehiculos=(pd.read_csv('071 vehiculos-original.csv').rename(columns={
    'cylinders':'cilindros',
    'trany':'transmision',
    'make':'fabricante',
    'model':'modelo',
    'displ':'desplazamiento',
    'drive':'traccion',
    'VClass':'clase',
    'fuelType':'combustible',
    'comb08':'consumo',
    'co2TailpipeGpm':'co2',
}).query('co2>0'))
vehiculos.shape


# In[ ]:


vehiculos=vehiculos.drop_duplicates()
vehiculos.shape


# In[ ]:


from scipy import stats
import numpy as np
def valores_extremos(df):#visualiza si hay valores extremos
    for columna in df:
        if df[columna].dtype !=np.object:
            n_valores= len(df[np.abs(stats.zscore(df[columna]))>3])
            print('{} | {} | {}'.format(df[columna].name,n_valores,df[columna].dtype))

valores_extremos(vehiculos)


# In[ ]:


#Para eliminar valores extremos
vehiculos.describe()


# In[ ]:


consumo_min=vehiculos.consumo.mean()-3*vehiculos.consumo.std()
consumo_max=vehiculos.consumo.mean()+3*vehiculos.consumo.std()
print(consumo_min,consumo_max)

co2_min=vehiculos.co2.mean()-3*vehiculos.co2.std()
co2_max=vehiculos.co2.mean()+3*vehiculos.co2.std()
print(co2_min,co2_max)


# In[ ]:


vehiculos=vehiculos[(np.abs(stats.zscore(vehiculos.consumo))<3)&                   (np.abs(stats.zscore(vehiculos.co2))<3)
                   ]
vehiculos.shape


# In[ ]:


vehiculos.describe()


# In[ ]:


vehiculos=vehiculos[(np.abs(stats.zscore(vehiculos.consumo))<3)&                   (np.abs(stats.zscore(vehiculos.co2))<3)]
valores_extremos(vehiculos)
vehiculos.shape


# In[ ]:


#Valores inexistentes
valores_df=len(vehiculos)
def valores_inexistentes(df):#Muestra valores inexistentes en el data frame
    for columna in df:
        print('{} | {} | {} |'.format(df[columna].name,len(df[df[columna].isnull()])/(1.0*valores_df),df[columna].dtype))
        
        
valores_inexistentes(vehiculos)


# In[ ]:


#Eliminando valores nulos / no recomendable
vehiculos_sin_null=vehiculos.dropna(subset=['transmision','desplazamiento','cilindros'])
vehiculos.shape


# In[ ]:


#Forma recomendable
vehiculos['transmision_imp']=vehiculos.transmision.fillna('sin transmision')#fillna rellena los valores nulos
vehiculos['desplazamiento_imp']=vehiculos.desplazamiento.fillna(0)
vehiculos['cilindros_imp']=vehiculos.cilindros.fillna(0)
vehiculos.shape
vehiculos.head()


# In[ ]:


#Otra forma
transmision_moda=vehiculos.transmision.mode().values(0)
cilindros_moda=vehiculos.cilindros.mode().values(0)
desplazamineto_mediana=vehiculos.desplazamiento.median()

vehiculos.transmision=vheiculos.transmision.fillna(transmision_moda)
vehiculos.desplazamiento=vehiculos.desplazamiento.fillna(desplazamineto_mediana)
vehiculos.cilindros=vehiculos.cilindros.fillna(cilindros_moda)

vehiculos[vehiculos.transmision_imp=='sin transmision']


# In[ ]:


vehiculos=vehiculos.drop(['transmision_moda','cilindros_moda','desplazamineto_moda'],axis=1)


# In[ ]:


vehiculos.describe()


# In[ ]:


#Ejemplo de donde la variacion de datos es muy grnade
vehiculos.desplazamiento.plot.kde();


# In[ ]:


stats.skew(vehiculos.desplazamiento)


# In[ ]:


#Normalizacion de variables
desplazamiento_x_min=vehiculos.desplazamiento.min()
desplazamiento_x_max=vehiculos.desplazamiento.max()

desplazamiento_original=vehiculos.desplazamiento
desplazamiento_normalizado= desplazamiento_original.apply(lambda x:(x-desplazamiento_x_min)/(desplazamiento_x_max-desplazamiento_x_min))

desplazamiento_normalizado.plot.hist(label='normalizado')
desplazamiento_normalizado.plot.hist(label='original')
plt.legend();


# In[ ]:


#Estandarizacion
desplazamiento_mu=vehiculos.desplazamiento.mean()
desplazamiento_sigma=vehiculos.desplazamiento.std()
desplazamiento_estandarizado=desplazamiento_original.apply(lambda x:(x-desplazamiento_mu)/desplazamiento_sigma)

desplazamiento_estandarizado.plot.hist(label='estandarizado',alpha=0.5)
desplazamiento_original.plot.hist(label='original',alpha=0.5)
plt.legend();


# In[ ]:


vehiculos['desplazamiento_std']=desplazamiento_estandarizado
vehiculos.head(8)


# In[ ]:


vehiculos.to_pickle('data_frame_estandarizado.pkl')


# In[ ]:


import pandas as pd
vehiculos=pd.read_pickle('data_frame_estandarizado.pkl')
vehiculos.head()


# In[ ]:


vehiculos_prueba=vehiculos.dropna(subset=['desplazamiento','cilindros','consumo'])
vehiculos_prueba.to_csv('data_frame_estandarizado.csv')


# In[ ]:




