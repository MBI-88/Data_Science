#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


mi_dataset= pd.read_csv("C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/054 datos-misteriosos.csv")
mi_dataset.head()


# In[2]:


#Relacion de dataset con x y y 
get_ipython().run_line_magic('matplotlib', 'notebook')
mi_dataset[mi_dataset.dataset=="clase1"].plot.scatter(x="x", y="y")


# In[3]:


#Relacion de clase2 contra x y 
mi_dataset[mi_dataset.dataset=="clase2"].plot.scatter(x="x", y="y",marker="*",color="red")
plt.title("Relacion de clase2 con x y y")
plt.xlabel("Valor X")
plt.ylabel("Valor Y")


# In[4]:


plt.style.available


# In[5]:


plt.style.use("seaborn")
mi_dataset[mi_dataset.dataset=="clase2"].plot.scatter(x="x", y="y",marker="*",color="red")
plt.title("Relacion de clase2 con x y y")
plt.xlabel("Valor X")
plt.ylabel("Valor Y")


# In[6]:


from ipywidgets import interact
def grafico_variable(col):
    mi_dataset.plot.scatter(x=col,y="y")
    plt.title("{} vs valor de y".format(col))


grafico_variable("x")


# In[7]:


#Variante
def grafica(clase,col):
    mi_dataset[mi_dataset.dataset==clase].plot.scatter(x=col,y=col)
    plt.title("{} vs valor de {}".format(clase,col))
    plt.ylabel("Valor de {}".format(clase))
    plt.xlabel("Valor de {}".format(col))

grafica("clase1","y")


# In[8]:


grafica("clase2","y")


# In[9]:


grafica("clase1","x")


# In[10]:


grafica("clase2","x")


# In[11]:


#Trabajo con seaborn
import seaborn as sns
sns.lmplot(x="x",y="x",data=mi_dataset,markers="*")


# In[12]:


sns.heatmap(mi_dataset.corr())

