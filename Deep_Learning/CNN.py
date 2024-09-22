#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy import  signal
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize']=(4,4)

imag_size=64
img_borde_vertical=np.zeros((imag_size,imag_size))
img_borde_vertical[:,:int(imag_size/2)]=1
plt.imshow(img_borde_vertical,cmap='gray')
plt.title('Imagen original');


# In[5]:


filtro_sobel_vertical=np.array([[-1,0,1],
                               [-1,0,1],
                               [-1,0,1]])
output_convolucion=signal.convolve2d(img_borde_vertical,filtro_sobel_vertical,mode='valid')
plt.imshow(np.absolute(output_convolucion),cmap='gray')
plt.title('Output de la convolucion con filtro vertical');


# In[6]:


from scipy import misc

img_original=misc.ascent()
plt.imshow(img_original,cmap='gray')
plt.title('Imagen Original');


# # Aplicando el filtro  vertical

# In[7]:


output_convolucion=signal.convolve2d(img_original,filtro_sobel_vertical)
plt.imshow(np.absolute(output_convolucion),cmap='gray')
plt.title('Output de la convolucion con filtro vertical');


# # Aplicando filtro horizontal

# In[11]:


filtro_sobel_horizontal=np.array([[-1,-1,0],
                                 [0,0,0],
                                 [1,1,1]])
output_convolucion=signal.convolve2d(img_original,filtro_sobel_horizontal)
plt.imshow(np.absolute(output_convolucion),cmap='gray')
plt.title('Output de la convolucion con filtro horizontal');


# # CNN en Keras

# In[12]:


from keras.datasets import fashion_mnist

(x_train,y_train),(x_test,y_test)= fashion_mnist.load_data()


# In[13]:


x_train.shape


# In[15]:


y_train[:10]


# In[17]:


np.bincount(y_train) # Nos indica que el dataset esta balanciando


# In[20]:


from keras.utils import to_categorical

y_train_one_hot=to_categorical(y_train)
y_test_one_hot=to_categorical(y_test)
y_train_one_hot[:10]


# In[21]:


from ipywidgets import interact,IntSlider

@interact(i=IntSlider(min=0,max=100,step=1,value=1))
def dibujar_imagen(i):
    plt.imshow(x_train[i],cmap='gray')
    plt.title('Clase de prenda: {}'.format(y_train[i]))


# # Primero usamos una capa densa

# In[22]:


from keras.models import Sequential
from keras.layers import Dense,Dropout

x_train_plano=x_train.reshape(x_train.shape[0],28*28)
x_test_plano=x_test.reshape(x_test.shape[0],28*28)
x_train_plano[0].shape


# In[28]:


modelo_denso=Sequential()
modelo_denso.add(Dense(128,activation='relu',input_shape=(784,)))
modelo_denso.add(Dropout(0.2))
modelo_denso.add(Dense(256,activation='relu'))
modelo_denso.add(Dropout(0.2))
modelo_denso.add(Dense(128,activation='relu'))
modelo_denso.add(Dropout(0.2))
modelo_denso.add(Dense(np.unique(y_train).shape[0],activation='softmax'))

modelo_denso.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_denso.summary()


# In[29]:


modelo_denso.fit(x_train_plano,y_train_one_hot,epochs=50,batch_size=1000,verbose=0);


# In[30]:


modelo_denso.evaluate(x_test_plano,y_test_one_hot)


# # Red CNN

# In[31]:


img_row,img_cols=28,28
x_train=x_train.reshape(x_train.shape[0],img_row,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_row,img_cols,1)
input_shape=(img_row,img_cols,1)
x_train.shape


# In[33]:


from keras.layers import Conv2D,Flatten,MaxPool2D

batch_size=256
num_classes=10
epochs=50
input_shape=(28,28,1)

modelo_cnn=Sequential()
modelo_cnn.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
modelo_cnn.add(MaxPool2D(pool_size=(2,2)))
modelo_cnn.add(Dropout(0.25))
modelo_cnn.add(Flatten())
modelo_cnn.add(Dense(32,activation='relu'))
modelo_cnn.add(Dropout(0.5))
modelo_cnn.add(Dense(num_classes,activation='softmax'))
modelo_cnn.summary()


# In[34]:


modelo_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
modelo_cnn.fit(x_train,y_train_one_hot,epochs=50,batch_size=1000,verbose=0);


# In[35]:


modelo_cnn.evaluate(x_test,y_test_one_hot,verbose=0)


# In[45]:


import keras.backend as k
plt.rcParams['figure.figsize']=(15,15)

def get_activation(modelo,modelo_inputs,print_shape_only=False,layer_name=None):
    print('-----activation-----')
    activations=[]
    inp=modelo.input
    
    modelo_multi_inputs_cond=True
    
    if not isinstance(inp,list):
        inp=[inp]
        modelo_multi_inputs_cond=False
    
    output=[layer.output for layer in modelo.layers if layer.name == layer_name or layer_name is None]
    funcs =[k.function(inp + [k.learning_phase()],[out])  for out in output]
    
    if modelo_multi_inputs_cond :
        list_inputs=[]
        list_inputs.extend(modelo_inputs)
        list_inputs.append(0.)
    else:
        list_inputs=[modelo_inputs,0.]
    
    layer_outputs=[func(list_inputs)[0] for func in funcs]
    for layer_activation in layer_outputs:
        activations.append(layer_activation)
        if  print_shape_only:
            print(layer_activation.shape)
        else:
            print(layer_activation)
    return activations

def display_activation(activation_maps):
    batch_size=activation_maps[0].shape[0]
    assert  batch_size==1,'One image at  time to visualize'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map{}'.format(i))
        shape=activation_map.shape
        if len(shape)==4:
            print (shape)
            activations=np.hstack(np.transpose(activation_map[0],(2,0,1)))
        elif len(shape)==2:
            activations=activation_map[0]
            num_activations=len(activations)
            if num_activations > 1024:
                square_param=int(np.floor(np.sqrt(num_activations)))
                activations=activations[0:square_param*square_param]
                activations=np.reshape(activations,(square_param,square_param))
            else:
                activations=np.expand_dims(activations,axis=0)
        else:
            raise  Exception('len(shape)=3 has not been implemented')
        plt.imshow(activations,interpolation='None',cmap='jet')
        plt.show()


# In[54]:


activations=get_activation(modelo_cnn,x_train[0].reshape(1,28,28,1),print_shape_only=True)


# In[55]:


display_activation(activations)

