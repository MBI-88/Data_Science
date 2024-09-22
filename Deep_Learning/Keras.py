#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(8.5,8.5)

data=load_breast_cancer()
X,y=data.data,data.target
X


# In[18]:


y.shape


# In[19]:


X=data.data[:,:4]
X


# In[20]:


X_std=StandardScaler().fit_transform(X)
y=y.reshape(569,1)
y.shape


# In[21]:


y[:10]


# In[4]:


from keras.models import Sequential
from keras.layers import Dense

modelo=Sequential()
modelo.add(Dense(units=5,activation='sigmoid',input_shape=(4,)))
modelo.add(Dense(units=1,activation='sigmoid',))


# In[5]:


modelo.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[6]:


from keras.optimizers import SGD

sgd=SGD(lr=0.01)
modelo.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
modelo.summary()


# In[8]:


historial=modelo.fit(X_std,y,epochs=100,verbose=1)


# In[9]:


plt.plot(historial.history['acc'])
plt.title('Precision vs epocas de entrenamiento');


# In[10]:


modelo.predict(X_std)[:5]


# In[11]:


modelo.predict_classes(X_std)[:5]


# In[12]:


scores=modelo.evaluate(X_std,y)
scores


# In[13]:


modelo.metrics_names # los nombres hacen referencia a los valores de arriba


# # Callbacks

# In[17]:


from keras.callbacks import Callback
from sklearn.metrics import f1_score,precision_score,recall_score
# Ejemplo de calcular una metrica por epoca
class MetricasEpoca(Callback):
    def on_train_begin(self,logs={}):
        self.f1_epoca=[]
        self.recall_epoca=[]
        self.precision_epoca=[]
        
    def on_epoch_end(self,epoch,logs={}):
        val_predict=self.model.predict_classes(self.validation_data[0])
        val_targ=self.validation_data[1]
        f1=f1_score(val_targ,val_predict)
        recall=recall_score(val_targ,val_predict)
        precision=precision_score(val_targ,val_predict)
        self.f1_epoca.append(f1)
        self.recall_epoca.append(recall)
        self.precision_epoca.append(precision)
        

modelo=Sequential([
    Dense(units=5,activation='sigmoid',input_dim=4),
    Dense(units=1,activation='sigmoid')
])
modelo.compile(loss='binary_crossentropy',optimizer=sgd)
metricas_epocas=MetricasEpoca()
modelo.fit(X_std,y,validation_data=(X_std,y),epochs=100,verbose=0,callbacks=[metricas_epocas])


# In[18]:


plt.plot(metricas_epocas.f1_epoca)
plt.title('Metrica F1 vs numero de epocas');


# # Early Stopping

# In[19]:


from keras.callbacks import EarlyStopping

earlystop=EarlyStopping(monitor='acc',min_delta=0.00001,patience=10,verbose=1,mode='auto')
modelo=Sequential([
    Dense(units=5,activation='sigmoid',input_dim=4),
    Dense(units=1,activation='sigmoid')
])
modelo.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['acc'])
modelo.fit(X_std,y,epochs=100,verbose=0,callbacks=[earlystop]);


# # Guardando modelos en Keras

# In[20]:


from keras.callbacks import ModelCheckpoint  # Otra variante es usar modelo.save que lo guarda despues de entrenado

checkpoint=ModelCheckpoint(filepath='modelo.hdf5',verbose=0,period=10)
modelo=Sequential([
    Dense(units=5,activation='sigmoid',input_dim=4),
    Dense(units=1,activation='sigmoid')
])

modelo.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['acc'])
modelo.fit(X_std,y,epochs=100,verbose=0,callbacks=[checkpoint]);


# In[21]:


get_ipython().system('dir')


# # Cargando el modelo

# In[22]:


from keras.models import load_model

modelo_cargado=load_model('modelo.hdf5')
modelo_cargado.predict(X_std)[:5]


# # Validacion Cruzada

# In[25]:


from sklearn.model_selection import StratifiedKFold,RepeatedKFold

def generar_modelo():
    modelo=Sequential()
    modelo.add(Dense(units=5,activation='sigmoid',input_dim=4))
    modelo.add(Dense(units=1,activation='sigmoid'))
    sgd=SGD(lr=0.01)
    modelo.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return modelo

kfold=StratifiedKFold()
cvscores=[]
for train,test in kfold.split(X_std,y):
    modelo=generar_modelo()
    modelo.fit(X_std[train],y[train],epochs=100,verbose=0)
    scores=modelo.evaluate(X_std[test],y[test],verbose=0)
    cvscores.append(scores[1])

np.mean(cvscores)


# # Optimizacion de hiperparametros

# In[26]:


from keras.wrappers.scikit_learn import KerasClassifier

def generar_modelo(n_ocultas=5,activacion='sigmoide'):
    modelo=Sequential()
    modelo.add(Dense(units=n_ocultas,activation=activacion,input_dim=4))
    modelo.add(Dense(units=1,activation=activacion))
    sgd=SGD(lr=0.0001)
    modelo.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return modelo
modelo=KerasClassifier(build_fn=generar_modelo,verbose=0)


# In[27]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

param_grid={
    'epochs':[10,30],
    'n_ocultas':[5,20],
    'activacion':['sigmoid','relu']
}

grid=GridSearchCV(estimator=modelo,param_grid=param_grid,n_jobs=-1,scoring='accuracy')
grid_result=grid.fit(X_std,y)


# In[29]:


print('Mejor estimador (error {:.5f}): {}'.format(grid_result.best_score_,grid_result.best_params_))

