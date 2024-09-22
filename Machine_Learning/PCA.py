#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize']=[10,10]
np.random.seed(42)


# In[2]:


from sklearn.datasets import load_iris

iris=load_iris()
iris.feature_names


# In[3]:


iris.data[:10]


# In[4]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from mpl_toolkits.mplot3d import Axes3D # Esto es para proyectar la visualizacion en 3d
from IPython.display import display
import ipywidgets as widgets

fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('longitud sepalo',size=20)
ax.set_ylabel('anchura sepalo',size=20)
ax.set_zlabel('longitud petalo',size=20)
ax.scatter(iris.data[:,0],iris.data[:,1],iris.data[:,2],c=iris.target,cmap=plt.cm.prism)
ax.view_init(20,120)
plt.show()

def actulizar_grafica(angulo1=20,angulo2=120):
    ax.view_init(angulo1,angulo2)
    fig.canvas.draw_idle()

angulo_slider=widgets.IntSlider(20,min=0,max=90)
def actulizar_angulo1(value):
    actulizar_grafica(angulo1=value['New'])
    
    
angulo_slider.observe(actulizar_angulo1,names='values');


# # Centrando los datos

# In[5]:


iris_centrado=(iris.data-iris.data.mean(axis=0))[:,:3]
iris_centrado[:10]


# # Calcular la matriz de covarianza

# In[6]:


def varianza(var1,var2=None):
    if var2 is None:
        var2=var1
    assert var1.shape==var2.shape
    var1_mean=var1.mean()
    var2_mean=var2.mean()
    return np.sum((var1-var1_mean)*(var2-var2_mean))/(var1.shape[0]-1)
var1=np.array([5,10,17,35])
var2=np.array([34,70,75,50])
varianza(var1,var2)


# In[7]:


varianza(var1)


# In[8]:


varianza(var2)


# In[9]:


#ya esto esta implementado en numpy de forma optima y en la practica se hace asi:
np.cov(np.array([var1,var2]))


# In[10]:


cov_mat=np.cov(m=iris_centrado.T)
cov_mat


# In[11]:


val_propios,vec_propios=np.linalg.eig(cov_mat)
print('Vectores propios:\n',vec_propios)
print('\nValores propios:',val_propios)


# In[12]:


orden_componetes=np.argsort(val_propios)[::-1]
val_propios_ordenados=val_propios[orden_componetes]
vec_propios_ordenados=vec_propios[:,orden_componetes]
print('Vectores propios:\n',vec_propios_ordenados)
print('\nValores propios:',val_propios_ordenados)


# In[13]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3d(FancyArrowPatch):
    def __init__(self,xs,ys,zs,*args,**kwargs):
        FancyArrowPatch.__init__(self,(0,0),(0,0),*args,**kwargs)
        self._verts3d = xs,ys,zs
    
    def draw(self,renderer):
        xs3d,ys3d,zs3d=self._verts3d
        xs,ys,zs=proj3d.proj_transform(xs3d,ys3d,zs3d,renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self,renderer)
        


# In[14]:


media_x=iris_centrado[:,0].mean()
media_y=iris_centrado[:,1].mean()
media_z=iris_centrado[:,2].mean()

fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111,projection='3d')
ax.set_xlabel('eje X',size=20)
ax.set_ylabel('eje Y',size=20)
ax.set_zlabel('eje Z',size=20)

ax.scatter(iris_centrado[:,0],iris_centrado[:,1],iris_centrado[:,2],c=iris.target,cmap=plt.cm.prism)
for v in vec_propios:
    a=Arrow3d([media_x,v[0]],[media_y,v[1]],[media_z,v[2]],mutation_scale=20,lw=3,arrowstyle='-|>',color='r')
    ax.add_artist(a)

ax.view_init(20,120)
plt.show()


def actulizar_grafica(angulo1=20,angulo2=120):
    ax.view_init(angulo1,angulo2)
    fig.canvas.draw_idle()

angulo_slider=widgets.IntSlider(20,min=0,max=90)

def actulizar_angulo1(value):
    actulizar_grafica(angulo1=value['New'])
    
    
angulo_slider.observe(actulizar_angulo1,names='values');


# In[15]:


print("""
PCA 1: {0:.2f}% de la varianza
PCA 2: {1:.2f}% de la varianza
PCA 3: {2:.2f}% de la varianza
""".format(*tuple(val_propios_ordenados/val_propios_ordenados.sum()*100)))


# # Para reducir el dataset de 3d a 2d

# In[16]:


vec_propios_ordenados[:,:2].T


# In[17]:


iris_coord_pinc=iris_centrado @ vec_propios_ordenados[:,:2]
iris_coord_pinc[:10]


# In[18]:


fig=plt.figure(figsize=(12,12))
plt.scatter(iris_coord_pinc[:,0],iris_coord_pinc[:,1],c=iris.target,cmap=plt.cm.Set3)
plt.title('Dataset Iris descompuesto en sus 2 primeros componentes principales',size=18)
plt.xlabel('Componentes principal 1',size=18)
plt.ylabel('Componente principal 2',size=18);


# # Todo lo anterior hecho manualmente ahora se ara con sklearn

# In[19]:


from sklearn.decomposition import PCA

pca=PCA(n_components=2)
iris_pca=pca.fit_transform(iris_centrado)


# In[20]:


pca.components_


# In[21]:


pca.explained_variance_ratio_


# In[22]:


iris_pca[:10]


# In[23]:


fig=plt.figure(figsize=(12,12))
plt.scatter(iris_pca[:,0],iris_pca[:,1],c=iris.target,cmap=plt.cm.Set3)
plt.title('Dataset Iris descompuesto en sus 2 primeros componentes principales',size=18)
plt.xlabel('Componentes principal 1',size=18)
plt.ylabel('Componente principal 2',size=18);


# # Ejemplo 2

# In[24]:


from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
cancer.data.shape


# In[25]:


from sklearn.preprocessing import StandardScaler# el estandarizador resta la media y divide por la desviacion estandar,esto es porque hay que sustraerle la media

scalador=StandardScaler().fit_transform(cancer.data)
pca=PCA(n_components=2).fit(scalador)
cancer_pca=pca.transform(scalador)


# In[26]:


fig=plt.figure(figsize=(12,12))
plt.scatter(cancer_pca[:,0],cancer_pca[:,1],c=cancer.target,cmap=plt.cm.Set3)
plt.title('Dataset Iris descompuesto en sus 2 primeros componentes principales',size=18)
plt.xlabel('Componentes principal 1',size=18)
plt.ylabel('Componente principal 2',size=18);


# # Ejemplo 3

# In[27]:


alimentos=pd.read_csv('C:/Users\MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/alimentos_usda.csv')
alimentos.head()


# In[28]:


alimentos.shape


# In[29]:


nombre_alimentos=alimentos.Shrt_Desc
alimentos_datos=StandardScaler().fit_transform(alimentos.drop('Shrt_Desc',axis=1).fillna(0))
alimentos_pca=PCA(n_components=2).fit_transform(alimentos_datos)
alimentos_pca_df=pd.DataFrame(alimentos_pca,columns=['PC_1','PC_2'])
alimentos_pca_df['nombre']=nombre_alimentos
alimentos_pca_df.head()


# # Para grafico interactivo se utiliza bokeh

# In[30]:


from bokeh.io import output_notebook
from bokeh.plotting import figure,show,ColumnDataSource
from bokeh.models import HoverTool
output_notebook()


# In[31]:


source=ColumnDataSource(alimentos_pca_df)

hover=HoverTool(tooltips=[
    ('(x,y)','($x,$y)'),
    ('nombre','@nombre'),])

p=figure(plot_width=880,plot_height=880,tools=[hover,'box_zoom','pan','zoom_out','zoom_in'],
        title='2 Componentes Principales de Valores Nutritivos de alimentos')
p.circle('PC1','PC2',size=20,source=source,fill_alpha=0.5)
show(p);

