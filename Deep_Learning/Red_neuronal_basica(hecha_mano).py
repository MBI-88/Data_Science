#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data=load_breast_cancer()
X,y=data.data,data.target
X=data.data[:,:4]

x_standarizador=StandardScaler()
X_std=x_standarizador.fit_transform(X)

# Generando una capa (layer)

class Layer:
    def __init__(self,dim_input,dim_output,fn_activacion,nombre):
        self.dim_input=dim_input
        self.dim_output=dim_output
        self.generar_pesos((dim_output,dim_input))
        self.generar_bias(dim_output)
        self.fn_activacion=fn_activacion
        self.nombre=nombre
    
    def __repr__(self):
        return """
        Capa {}. tamaño input: {}.
        pesos: {}
        bias: {}
        """.format(self.nombre,self.dim_input,self.dim_output,self.w,self.b)
    
    def generar_pesos(self,dimensiones):
        self.w=np.random.random(dimensiones)
    
    def generar_bias(self,dim_output):
        self.b=np.random.random((dim_output,))
    
    def activar(self,x):
        return self.fn_activacion(self.w @ x+self.b)
    

def fn_sigmoide(x):
    return 1/(1+np.exp(-x))

n_input=4
n_ocultas=5
n_ouput=1

capa_oculta=Layer(n_input,n_ocultas,fn_sigmoide,'oculta')
capa_salida=Layer(n_ocultas,n_ouput,fn_sigmoide,'salida')
print(capa_oculta)


# # Propagacion hacia adelante

# In[5]:


class RedNeuronal:
    def __init__(self):
        self.layers=[]
    
    def add_layer(self,layer):
        self.layers.append(layer)
    
    def forward(self,x):
        print(""" input: {}""".format(x))
        for layer in self.layers:
            x=layer.activar(x)
            print(layer)
            print("""output : {} """.format(x))
        return x

red=RedNeuronal()
red.add_layer(capa_oculta)
red.add_layer(capa_salida)
indice_aleatorio=np.random.permutation(X.shape[0])# Tomando solo la primera muestra
x0=X_std[indice_aleatorio[0]]
y0=y[indice_aleatorio[0]]
print(x0,y0)


# In[6]:


red.forward(x0)


# # Propagacion hacia Atras (Formando una red completa basica)

# In[12]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
plt.rcParams['figure.figsize']=(9,9)

data=load_breast_cancer()
X,y=data.data,data.target
X=data.data[:,:4]

x_standarizador=StandardScaler()
X_std=x_standarizador.fit_transform(X)


# In[43]:


def fn_identidad(x,derivada=False):
    if derivada:
        return np.ones(x.shape)
    return x

def fn_sigmoide_modificada(x,derivada=False):
    if derivada:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def error_logloss(y_pred,y):
    p=np.clip(y_pred,1e-15,1-1e-15)
    if y==1:
        return -np.log(p)
    else:
        return -np.log(1-p)


# Redefinicion de la capa basica
class Layer_OK:
    def __init__(self,n_unidades,fn_activacion,bias=True):
        self.n_unidades=n_unidades
        self.fn_activacion=fn_activacion
        self.dim_output=n_unidades
        self.bias=bias
        self.dimensiones='no generada'
        self.w=None
    
    def __repr__(self):
        return """Capa {}. dimensiones = {} . pesos: {} """.format(self.nombre,self.dimensiones,self.w)
    
    def generar_pesos(self,dim_output_anterior):
        if self.bias:
            self.dimensiones=(self.n_unidades,dim_output_anterior + 1)
        else:
            self.dimensiones=(self.n_unidades,dim_output_anterior)
            
        self.w=np.random.random(self.dimensiones)
        
    def add_bias(self,x):
        if not self.bias:
            return x
        x_con_bias_1d=np.append(1,x)
        return x_con_bias_1d.reshape(x_con_bias_1d.shape[0],1)
    
    def activar(self,x):
        x_con_bias_2d=self.add_bias(x)
        return self.fn_activacion(self.w @ x_con_bias_2d)
    
    def calcular_delta(self,producto_capa,output_capa):
        return producto_capa * self.fn_activacion(output_capa,derivada=True)
    

class InputLayer(Layer_OK):
    nombre='entrada'
    
    def generar_pesos(self):
        pass
    def activar(self,x):
        return x
    
class HiddenLayer(Layer_OK):
    nombre='oculta'
    
class OutputLayer(Layer_OK):
    nombre='salida'

# Redefinicion de la red neuronal
class RedNeuronal_Ok:
    def __init__(self,ratio_aprendizaje,fn_error):
        self.layers=[]
        self.ratio_aprendizaje=ratio_aprendizaje
        self.fn_error=fn_error
    
    def add_layer(self,layer):
        if layer.nombre=='entrada':
            layer.generar_pesos()
        else:
            layer.generar_pesos(self.layers[-1].dim_output)
            
        self.layers.append(layer)
    
    def __repr__(self):
        info_red=''
        for layer in self.layers:
            info_red += '\nCapa: {} Nº unidades: {}'.format(layer.nombre,layer.n_unidades)
        return info_red
    
    def forward(self,x):
        for layer in self.layers:
            layer.input=layer.add_bias(x).T
            x=layer.activar(x)
            layer.output=x
        return x
    def calcular_error_predicion(self,y_pred,y):
        return self.fn_error(y_pred,y)
    
    def backward(self,y_pred,y):
        delta_capa=self.calcular_error_predicion(y_pred,y)
        for layer in reversed(self.layers):
            if layer.nombre=='entrada':
                continue
            if layer.nombre=='salida':
                producto_capa=delta_capa @ layer.w
            else:
                producto_capa=delta_capa[:,1:] @ layer.w
            delta_capa=layer.calcular_delta(producto_capa,layer.output)
            layer.delta=delta_capa
    
    def actualizar_pesos(self):
        # Actualiza pesos mediante el descenso del gradiente
        for layer in self.layers[1:]:
            layer.w=layer.w - self.ratio_aprendizaje * layer.delta * layer.input
        
    
    def aprendizaje(self,x,y):# Fucion principal para entrenar la red
        y_pred=self.forward(x)
        self.backward(y_pred,y)
        self.actualizar_pesos()
        error_prediccion=self.calcular_error_predicion(y_pred,y)
        return error_prediccion

    def predict(self,x):
        probabilidad=self.forward_Ok(x)
        if probabilidad >= 0.5:
            return 1
        else: return 0

    def predic_proba(self,x):
        return self.forward_Ok(x)    


# In[45]:


n_input=4
n_oculta=5
n_output=1
ratio_aprendizaje=0.0001
n_iter=1000
red_sigmoide=RedNeuronal_Ok(ratio_aprendizaje=ratio_aprendizaje,fn_error=error_logloss)
red_sigmoide.add_layer(InputLayer(n_input,bias=False,fn_activacion=fn_identidad))
red_sigmoide.add_layer(HiddenLayer(n_oculta,fn_activacion=fn_sigmoide_modificada))
red_sigmoide.add_layer(OutputLayer(n_output,fn_activacion=fn_sigmoide_modificada))


# In[46]:


red_sigmoide.layers


# In[47]:


red_sigmoide.aprendizaje(x0,y0)


# In[48]:


red_sigmoide.layers


# In[49]:


prediccion=red_sigmoide.forward(x0)
prediccion


# In[50]:


red_sigmoide.backward(prediccion,y0)


# In[51]:


red_sigmoide.layers


# In[52]:


red_sigmoide.actualizar_pesos()


# In[53]:


red_sigmoide.layers


# # Proceso del gradiente

# In[54]:


def iteracion_sgd(red,X,y):
    indice_aleatorio=np.random.permutation(X.shape[0])
    error=[]
    for i in range(indice_aleatorio.shape[0]):
        x0=X[indice_aleatorio[i]]
        y0=[indice_aleatorio[i]]
        err=red.aprendizaje(x0,y0)
        error.append(err)
    return np.nanmean(np.array(error))

def entrenar_sgd(red,n_epocas,X,y):
    epocas=[]
    for epoca in range(n_epocas):
        error_epoca=iteracion_sgd(red,X,y)
        epocas.append([epoca,error_epoca])
    return np.array(epocas)

resultado_sgmoide=entrenar_sgd(red_sigmoide,n_iter,X_std,y)


# In[55]:


resultado_sgmoide


# In[56]:


plt.scatter(x=resultado_sgmoide[:,0],y=resultado_sgmoide[:,1])
plt.title('Error para red con funcion sigmoide en capa oculta')
plt.xlabel('Numero de iteraciones')
plt.ylabel('Error medio');


# In[59]:


def fn_relu(x,derivada=False):
    if derivada :
        return 1.*(x>0.)
    return np.maximum(x,0.)

def fn_leakyrelu(x,derivada=False):
    if derivada:
        if x.any()>0:
            return 1.
        else:
            return 0.01
    return np.maximum(x,0.01*x)


red_relu=RedNeuronal_Ok(ratio_aprendizaje,fn_error=error_logloss)
red_relu.add_layer(InputLayer(n_input,bias=False,fn_activacion=fn_identidad))
red_relu.add_layer(HiddenLayer(n_oculta,fn_activacion=fn_relu))
red_relu.add_layer(HiddenLayer(n_oculta,fn_activacion=fn_relu))
red_relu.add_layer(OutputLayer(n_ouput,fn_activacion=fn_sigmoide_modificada))


# In[60]:


resultado_relu=entrenar_sgd(red_relu,n_iter,X_std,y)
plt.scatter(x=resultado_relu[:,0],y=resultado_relu[:,1])
plt.title('Error para  red con funcion Relu en capa oculta')
plt.xlabel('Numero de iteraciones')
plt.ylabel('Error medio');


# In[ ]:




