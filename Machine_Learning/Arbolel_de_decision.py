#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
datos=pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/titanic.csv')
datos.head()


# In[2]:


arboles=tree.DecisionTreeClassifier()
columnas_categoricas=['genero','puerto_salida']
codificador_binario=preprocessing.LabelBinarizer()#se puede usar este o dummies de pandas hacen lo mismo
datos_categoricos=pd.get_dummies(datos[columnas_categoricas])
pasajeros=(
    pd.concat([
        datos.drop(columnas_categoricas,axis=1),
        datos_categoricos
    ],axis=1
    )
)
pasajeros.edad=pasajeros.edad.fillna(pasajeros.edad.mean())
pasajeros.head()


# In[3]:


arboles.fit(pasajeros.drop('superviviente',axis=1),pasajeros.superviviente)


# In[4]:


cross_val_score(arboles,pasajeros.drop('superviviente',axis=1),pasajeros.superviviente,scoring='roc_auc',cv=10).mean()


# ### Para visualizar el arbol

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import graphviz
def dibujar_arbol(arboles):
    dot_data=tree.export_graphviz(arboles,out_file=None,
                                 feature_names=pasajeros.drop('superviviente',axis=1).columns,
                                 filled=True,
                                 impurity=False,
                                 rounded=True,
                                 special_characters=True)
    graph=graphviz.Source(dot_data)
    graph.format='png'
    graph.render('arbol',view=True)
dibujar_arbol(arboles)


# In[6]:


#Para evaluar la importancia de las ramas del arbol
dict(zip(
    pasajeros.drop('superviviente',axis=1),
    arboles.feature_importances_
))


# In[7]:


arbol_simple=tree.DecisionTreeClassifier(max_depth=3)
arbol_simple.fit(pasajeros.drop('superviviente',axis=1),pasajeros.superviviente)


# In[8]:


cross_val_score(arbol_simple,pasajeros.drop('superviviente',axis=1),pasajeros.superviviente,scoring='roc_auc',cv=10).mean()


# In[9]:


arbol_balanceado=tree.DecisionTreeClassifier(max_depth=3,class_weight='balanced')


# In[10]:


arbol_balanceado.fit(pasajeros.drop('superviviente',axis=1),pasajeros.superviviente)


# In[11]:


cross_val_score(arbol_balanceado,pasajeros.drop('superviviente',axis=1),pasajeros.superviviente,scoring='roc_auc',cv=10).mean()


# In[12]:


cross_val_score(arbol_balanceado,pasajeros.drop('superviviente',axis=1),pasajeros.superviviente,scoring='precision',cv=10).mean()


# Otra clase de arboles que tiene es  arboles extremadamente balanceado

# In[13]:


arbol_aleatorios=tree.ExtraTreeClassifier(max_features=1)
arbol_aleatorios.fit(pasajeros.drop('superviviente',axis=1),pasajeros.superviviente)


# In[14]:


cross_val_score(arbol_aleatorios,pasajeros.drop('superviviente',axis=1),pasajeros.superviviente,scoring='roc_auc',cv=10).mean()

