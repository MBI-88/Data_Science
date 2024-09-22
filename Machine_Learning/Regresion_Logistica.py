#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets

cancer_dataset= datasets.load_breast_cancer()
cancer_dataset.keys()


# In[2]:


cancer_dataset.target[:20]


# In[3]:


cancer_dataset.target_names


# In[3]:


cancer_df=pd.DataFrame(cancer_dataset.data,columns=cancer_dataset.feature_names)
cancer_df['objetivo']=cancer_dataset.target
cancer_df.shape


# In[5]:


cancer_df.head()


# In[23]:


cancer_df.objetivo.value_counts(normalize=True)


# In[4]:


#Usando un modelo de regresion lineal para  ver la comparacion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train_df,test_df= train_test_split(cancer_df,test_size=0.4)
variable_entrenamiento=cancer_dataset.feature_names
variable_objetivo='objetivo'
columna_entrenamiento='worst area'
plt.plot(train_df[columna_entrenamiento],train_df.objetivo, '.r')
plt.xlabel('Peor area encontrada en los nucleos de la imagen')
plt.ylabel('Diagnostico (Maligno | Beningno)');


# In[5]:


modelo_ols=LinearRegression()
modelo_ols.fit(train_df[[columna_entrenamiento]],train_df[variable_objetivo])
predicciones=modelo_ols.predict(test_df[[columna_entrenamiento]])
predicciones[:10]


# In[13]:


plt.plot(test_df[columna_entrenamiento],test_df.objetivo,'.r')
plt.plot(test_df[columna_entrenamiento],predicciones,'.b')
plt.xlabel('Peor area encontrada en los nucleos de la imagen')
plt.ylabel('Diagnostico (Maligno | Beningno)');


# In[6]:


# Teoria de la representacion de la funcion logistica
from ipywidgets import interact
def funcion_logistica(x,L=1,k=1,x0=0):
    return L/(1 + np.exp(-k*(x-x0)))
@interact(L=range(1,10),k=range(-5,5),x0=range(0,10))
def plot_funcion_logistica(L,k,x0):
    x=np.linspace(-5*k,5*k,5000)
    y=funcion_logistica(x,L=L,k=k,x0=x0)
    plt.figure(1)
    plt.plot(x,y)
    plt.show()
    


# In[16]:


# Teoria de la representacion de la funcion logistica
predicciones_probabilidades=list(map(funcion_logistica,predicciones))
plt.plot(test_df[columna_entrenamiento],test_df.objetivo,'.r')
plt.plot(test_df[columna_entrenamiento],predicciones,'.b')
plt.plot(test_df[columna_entrenamiento],predicciones_probabilidades,'.g')
plt.xlabel('Peor area encontrada en los nucleos de la imagen')
plt.ylabel('Diagnostico (Maligno | Beningno)');


# In[7]:


# Teoria de la representacion de la funcion logistica
from functools import partial
funcion_logistica_k5=partial(funcion_logistica,k=5)
predicciones_probabilidades=list(map(funcion_logistica_k5,predicciones))
plt.plot(test_df[columna_entrenamiento],test_df.objetivo,'.r')
plt.plot(test_df[columna_entrenamiento],predicciones,'.b')
plt.plot(test_df[columna_entrenamiento],predicciones_probabilidades,'.g')
plt.xlabel('Peor area encontrada en los nucleos de la imagen')
plt.ylabel('Diagnostico (Maligno | Beningno)');


# In[9]:


#Utilizando sklearn
from sklearn.linear_model import LogisticRegression
X=cancer_df[variable_entrenamiento]
y=cancer_df[variable_objetivo]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=LogisticRegression()
clf.fit(X_train,y_train)
predicciones=clf.predict(X_test)
predicciones[:10]


# In[24]:


predicciones_probabilidades=clf.predict_proba(X_test)
predicciones_probabilidades[:10]


# In[25]:


plt.hist(predicciones_probabilidades);


# In[29]:


probs_df= pd.DataFrame(predicciones_probabilidades)
X=X_test.reset_index().copy()
X['objetivo']=y_test.to_list()
X['prediccion']=predicciones
X=pd.concat([X,probs_df],axis=1)
X[['objetivo','prediccion',0,1]].head(20)

