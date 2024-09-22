#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pickle as  pck
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC

with open('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/mnist.pkl','rb') as fname:
    mnist=pck.load(fname)

mnist_data=mnist['training_images']
mnist_target=mnist['training_labels']
mnist_data.shape


# In[2]:


import numpy as np

sample_size=10000
np.random.seed(42)
indice_sample=np.random.randint(0,mnist_data.shape[0],sample_size)
indice_sample


# In[3]:


mnist_muestra_train=mnist_data[indice_sample]
mnist_muestra_clase=mnist_target[indice_sample]

pca=PCA(n_components=2)
mnist_pca=pca.fit_transform(mnist_muestra_train)


# In[93]:


mnist_pca.shape


# In[4]:


pca.components_


# In[5]:


pca.explained_variance_ratio_


# In[90]:


clasificador=OneVsRestClassifier(estimator=SVC(random_state=42))
pred=clasificador.fit(mnist_pca,mnist_muestra_clase).predict(mnist_pca)


# In[91]:


pred


# In[89]:


mnist_muestra_clase


# In[92]:


score=f1_score(mnist_muestra_clase,pred,average='micro')
score


#  Nota: La primera variante es pobre por eleligir 2 componentes para la reduccion, se pierde mucha informacion del dataset
#         la mejor forma es hacer siempre una optimizacion de hiperparametros para buscar la mejor relacion posible para el 
#         dataset de entrenamiento . Por tanto la variante correcta es la 2 para el uso de PCA

# # Variante

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV,space
from scipy.stats import  randint
np.random.seed(42)
pca=PCA()
pca.get_params()


# In[36]:


n_comp={'copy': space.Categorical([True,False]),
 'iterated_power': space.Categorical(['auto']),
 'n_components': space.Real(0.1,0.99,'uniform'),
 'random_state': space.Categorical([42,None]),
 'svd_solver': space.Categorical(['auto']),
 'tol': space.Real(0.1,0.99,'uniform'),
 'whiten': space.Categorical([False,True])}
  

buscador=BayesSearchCV(estimator=pca,search_spaces=n_comp,random_state=42,n_iter=10,n_jobs=-1)
buscador.fit(mnist_muestra_train,mnist_muestra_clase)


# In[87]:


buscador.best_params_


# In[45]:


pca=PCA(copy=True, iterated_power='auto', n_components=0.9302525189745722,
    random_state=42, svd_solver='auto', tol=0.46856558291212924, whiten=False)
mnist_pca_1=pca.fit_transform(mnist_muestra_train)


# In[94]:


mnist_pca_1.shape


# In[59]:


k_neibor=KNeighborsClassifier(n_jobs=-1)
k_neibor.get_params()


# In[74]:


busqueda_k_neibor={'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
 'leaf_size': randint(30,150),
 'n_neighbors': randint(5,15),
 'p': randint(2,10),
  'weights': ['uniform','distance']
 }


# In[75]:


print(k_neibor.__doc__)


# In[77]:


validando_k_neibor=RandomizedSearchCV(estimator=k_neibor,param_distributions=busqueda_k_neibor,scoring='f1_micro',random_state=42,
                                     cv=3,n_iter=10)
validando_k_neibor.fit(mnist_pca_1,mnist_muestra_clase)


# In[78]:


validando_k_neibor.best_params_


# In[88]:


validando_k_neibor.best_score_


# In[83]:


mejores_param={}
mejores_param=validando_k_neibor.best_params_
mejores_param

k_neibor=KNeighborsClassifier(**mejores_param)
k_neibor.fit(mnist_pca_1,mnist_muestra_clase)


# In[84]:


k_neibor.predict(mnist_pca_1)


# In[71]:


mnist_muestra_clase

