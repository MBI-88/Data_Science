#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(5,5)


f= lambda x: x**2-2*x+4
f(3)


# In[2]:


f_primera=lambda x: 2*x-2
f_primera(3)


# # Radio de aprendisaje

# In[3]:


step_size=0.02
def  descenso_gradiente(x):
    return x-step_size*(f_primera(x))

x0=3 # Minimo al azar
descenso_gradiente(x0)


# In[4]:


minimo_iteraciones=[]                # Algoritmo iterativo para ejemplificar el descenso del gradiente
n_iteraciones=100
x=3
for i in range(n_iteraciones):
    minimo_iteraciones.append(x)
    x=descenso_gradiente(x)


# In[5]:


plt.plot(minimo_iteraciones);


# # Batch Gradient Descent

# In[6]:


from sklearn.datasets import make_regression

n_muestras =1000
n_variables=2

x,y,coeficiente_objetivo=make_regression(n_features=n_variables,n_samples=n_muestras,coef=True)


# In[7]:


coeficiente_objetivo


# In[8]:


def predecir_bacht(coeficiente,x):
    return coeficiente @ x.T

y_pred=predecir_bacht(coeficiente_objetivo,x)
y_pred[:10]


# In[9]:


y_pred[:10]-y[:10] # Comprobando que las predicciones son iguales a la v ariable objetivo


# In[10]:


def error_bacht(y_pred,y_true): # Funcion de error (en este caso es el error cuadratico medio)
    m=y_pred.shape[0]
    return (np.sum(y_pred-y_true)**2)/2*m

def derivada_error_bacht(y_pred,y_true,x): # Derivada del error bacht
    m=y_pred.shape[0]
    return np.sum((y_pred-y_true)*x/m)

coeficientes=np.random.random((x.shape[1],))
coeficientes


# In[11]:


error_bacht(y_pred,y)


# In[12]:


derivada_error_bacht(y_pred,y,x[:,0])


# In[13]:


def descenso_gradiente_bacht(coef,x,y):
    y_predicciones=predecir_bacht(coef,x)
    for i in range(coef.shape[0]):
        coef[i]=coef[i]-step_size*derivada_error_bacht(y_predicciones,y,x[:,i])
    error=error_bacht(y_predicciones,y)
    return coef,error

coeficientes_iteraciones=[]
error_iteraciones=[]
n_iteraciones=200
coeficientes=np.random.random((x.shape[1],))
error=error_bacht(coeficientes,x)
for i in range(n_iteraciones):
    coeficientes_iteraciones.append(coeficientes.copy())
    error_iteraciones.append(error)
    coeficientes,error=descenso_gradiente_bacht(coeficientes,x,y)

coeficientes_iteraciones=np.array(coeficientes_iteraciones)


# In[14]:


coeficientes


# In[15]:


coeficiente_objetivo


# In[16]:


plt.plot(error_iteraciones)
plt.title('Evolucion del error con el numero de iteraciones');


# In[17]:


plt.plot(coeficientes_iteraciones[:,0],color='red')
plt.axhline(coeficiente_objetivo[0],color='red',linestyle='dashed')

plt.plot(coeficientes_iteraciones[:,1],color='blue')
plt.axhline(coeficiente_objetivo[1],color='blue',linestyle='dashed')

plt.xlabel('Numero de iteraciones')
plt.ylabel('Valor del coeficiente')
plt.title('Evolucion de coeficiente con el numero de iteraciones');


# #  Descenso de gradiente estocastico (SGD)

# In[18]:


def predecir_observacion(coeficientes,x):
    return coeficientes @ x.T

predecir_observacion(coeficientes,x[0])


# In[19]:


def derivada_error_observaciones(y_pred,y_true,x):
    return (y_pred-y_true)*x

derivada_error_observaciones(predecir_observacion(coeficientes,x[0]),y[0],x[0,0])


# In[20]:


def descenso_gradiente_estocastico(coef,x,y):
    y_predic=predecir_observacion(coef,x)
    for i in range(coef.shape[0]):
        coef[i]=coef[i]-step_size*derivada_error_observaciones(y_predic,y,x[i])
    return coef

coeficientes_iteraciones=[]
error_iteraciones=[]
coeficientes=np.random.random((x.shape[1],))
error=error_bacht(coeficientes,x)

indice_aleatorio=np.random.permutation(x.shape[0])
for i in indice_aleatorio:
    error_iteraciones.append(error)
    x_iteraciones=x[i]
    y_iteraciones=y[i]
    coeficientes_iteraciones.append(coeficientes.copy())
    coeficientes=descenso_gradiente_estocastico(coeficientes,x_iteraciones,y_iteraciones)
    y_predic=predecir_bacht(coeficientes,x)
    error=error_bacht(y_predic,y)

coeficientes_iteraciones=np.array(coeficientes_iteraciones)
coeficientes


# In[21]:


plt.plot(error_iteraciones)
plt.title('Evolucion del error con el numero de iteraciones');


# In[22]:


plt.plot(coeficientes_iteraciones[:,0],color='red')
plt.axhline(coeficiente_objetivo[0],color='red',linestyle='dashed')

plt.plot(coeficientes_iteraciones[:,1],color='blue')
plt.axhline(coeficiente_objetivo[1],color='blue',linestyle='dashed')

plt.xlabel('Numero de observaciones')
plt.ylabel('Valor del coeficiente')
plt.title('Evolucion de coeficiente den SGD con el numero de observaciones iteradas');


# # SGD en scikit-learn

# In[23]:


from sklearn.linear_model import SGDRegressor

estimador_sgd=SGDRegressor(max_iter=10)
estimador_sgd.fit(x,y)


# In[24]:


estimador_sgd.predict(x)[:10]


# In[25]:


estimador_sgd.coef_


# In[26]:


from sklearn.model_selection import cross_val_score

cross_val_score(SGDRegressor(max_iter=10),x,y,scoring='neg_mean_squared_error')

