#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from sklearn.datasets import load_iris
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[10,10]
datos=load_iris()
print(datos.DESCR)


# In[2]:


datos.target_names


# In[4]:


iris=pd.DataFrame(datos.data,columns=datos.feature_names)
iris['objetivo']=datos.target
iris.head()


# In[6]:


variables_inde=iris.drop('objetivo',axis=1).columns
iris_X=iris[variables_inde]
iris_y=iris.objetivo


# In[7]:


from sklearn.model_selection import train_test_split
iris_X_train,iris_X_test,iris_y_train,iris_y_test=train_test_split(iris_X,iris_y,test_size=0.2)


# In[8]:


from sklearn.svm import SVC,SVR
estimador_svm=SVC()
estimador_svm.fit(iris_X_train,iris_y_train)


# In[9]:


estimador_svm.predict(iris_X_test)[:10]


# In[10]:


estimador_svm.support_vectors_ 


# In[11]:


estimador_svm.n_support_   # nos dice cuantos puntos del hiperplano estan tocando el margen de desicion


# ## Kernel

# In[12]:


from mlxtend.plotting import plot_decision_regions
X=datos.data[:,:2]
y=datos.target
estimador_svm_lineal=SVC(kernel='linear')
estimador_svm_lineal.fit(X,y)
plot_decision_regions(X,y,clf=estimador_svm_lineal);


# In[13]:


estimador_svm_plinonmial=SVC(kernel='poly')
estimador_svm_plinonmial.fit(X,y)
plot_decision_regions(X,y,clf=estimador_svm_plinonmial);


# In[14]:


estimador_svm_plinonmial=SVC(kernel='poly',degree=2)
estimador_svm_plinonmial.fit(X,y)
plot_decision_regions(X,y,clf=estimador_svm_plinonmial);


# In[15]:


estimador_svm_plinonmial=SVC(kernel='poly',degree=6)
estimador_svm_plinonmial.fit(X,y)
plot_decision_regions(X,y,clf=estimador_svm_plinonmial);


# In[16]:


estimador_svm_rbf=SVC(kernel='rbf',gamma=0.1).fit(X,y)
plot_decision_regions(X,y,clf=estimador_svm_rbf);


# In[17]:


estimador_svm_rbf=SVC(kernel='rbf',gamma=10).fit(X,y)
plot_decision_regions(X,y,clf=estimador_svm_rbf);


# In[18]:


estimador_svm_rbf=SVC(kernel='rbf',gamma=100).fit(X,y) # sobre ajusta los datos
plot_decision_regions(X,y,clf=estimador_svm_rbf);


# ## Parametro de coste C

# In[20]:


from sklearn.model_selection import validation_curve
rango_c=np.linspace(0.01,50,50)
train_scores,test_scores=validation_curve(estimador_svm,iris_X,iris_y,param_name='C',
                                          param_range=rango_c,scoring='f1_weighted')
train_scores_mean=np.mean(train_scores,axis=1)
test_scores_mean=np.mean(test_scores,axis=1)
plt.plot(rango_c,train_scores_mean,'o-',color='r',label='Funcionamiento datos_entrenamienetos')
plt.plot(rango_c,test_scores_mean,'o-',color='g',label='Funcionamiento Validacion Cruzada')
plt.title('Curva Validacion : SVM con  kernel rbf')
plt.xlabel('Constante de regularizacion C')
plt.ylabel('Puntuacion F1')
plt.legend();# se nota el sobreajuste porque la f de datos sube y la de validacion baja


# # Probabilidades

# In[21]:


estimador_svm=SVC()
estimador_svm.fit(iris_X_train,iris_y_train)
estimador_svm.predict(iris_X_test)[:10]


# In[22]:


estimador_svm.decision_function(iris_X_test)[:10]


# In[23]:


estimador_svm_prob=SVC(probability=True)
estimador_svm_prob.fit(iris_X_train,iris_y_train)
estimador_svm_prob.predict_proba(iris_X_test)[:10]


# In[ ]:




