#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=[12,12]
np.random.seed(42)

face=fetch_lfw_people(min_faces_per_person=20,resize=0.7)


# In[2]:


face.target_names


# In[3]:


face.images=face.images[:10]
face.target=face.target[:10]
face.target_names[face.target]


# In[4]:


image_shape=face.images[0].shape

def dibujar_caras(array_caras):
    fig,axes=plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})
    for target , image, ax in zip(face.target,array_caras,axes.ravel()):
        ax.imshow(image,cmap='gray')
        ax.set_title(face.target_names[target])

dibujar_caras(face.images)


# In[5]:


def dibujar_face(matriz_face):
    plt.figure(figsize=(9,6))
    plt.imshow(matriz_face,cmap='gray')

dibujar_face(face.images[0])


# In[6]:


def kb_totales(*arrays):
    return sum(map(lambda a:a.nbytes,arrays ))/1024
kb_totales(face.images[0])


# # Usando el metodo de scipy.linalg.svd

# In[7]:


from scipy.linalg import svd

U,s,V=np.linalg.svd(face.images[0],full_matrices=True)
U.shape,s.shape,V.shape


# In[8]:


S=np.zeros((U.shape[0],V.shape[0]))
S[:V.shape[0],:V.shape[0]]=np.diag(s)
S.shape


# In[9]:


cara_svd=U @ S @ V
dibujar_face(cara_svd)


# In[10]:


kb_totales(U,s,V)


# In[11]:


from ipywidgets import IntSlider,interact

@interact(k=IntSlider(65,min=1,max=65))
def evaluar_k_imagen(k):
    U_k=U[:,:k]
    s_k=s[:k]
    V_k=V[:k,:]
    memoria_k=kb_totales(U_k,s_k,V_k)
    pct_reduccion=100*(1-(memoria_k/22.08984375))
    image_k_svd=U_k @ np.diag(s_k) @ V_k
    dibujar_face(image_k_svd)
    plt.title('{} valores singulares, tamaño: {:.1f} kb ({:.1f}% menor)'.format(k,memoria_k,pct_reduccion),size=20)
    plt.show();


# # Ejemplo de sistema de recomendacion

# In[12]:


ratings = pd.read_csv('C:/Users/MBI/Documents/Python_Scripts/Data_Set_Entrenamiento/ratings.csv')
ratings.shape


# In[13]:


ratings.head()


# In[14]:


ratings_mtx_df=ratings.pivot_table(values='rating',index='usuario',columns='pelicula',fill_value=0)
movie_index=ratings_mtx_df.columns


# In[15]:


ratings_mtx_df.shape


# In[16]:


ratings_mtx_df.head()


# In[17]:


ratings_mtx=np.array(ratings_mtx_df,dtype=float)
ratings_mtx.shape


# In[18]:


from scipy.sparse.linalg import  svds

U,s,V=svds(ratings_mtx, k=10)
U.shape,s.shape,V.shape


# In[19]:


s_diag_matix=np.zeros((s.shape[0],s.shape[0]))
for i in range(s.shape[0]):
    s_diag_matix[i,i]=s[i]

s_diag_matix


# In[20]:


ratings_svd=U @ s_diag_matix @ V
ratings_svd.shape


# In[21]:


ratings_svd


# # Ejemplo de SVD con sklearn

# In[22]:


from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
cancer.data.shape


# In[23]:


from sklearn.decomposition import TruncatedSVD

svd=TruncatedSVD()
svd.fit(cancer.data)
cancer_svd=svd.transform(cancer.data)


# In[24]:


svd.components_


# In[25]:


svd.singular_values_


# In[26]:


svd.explained_variance_ratio_


# In[27]:


from matplotlib import cm
fig=plt.figure(figsize=(12,12))
plt.scatter(cancer_svd[:,0],cancer_svd[:,1],c=cancer.target,cmap=cm.Set3)
plt.title('Dataset Cancer de mama 2 primeros valores singulares',size=18)
plt.xlabel('vector singular 1',size=10)
plt.ylabel('vector singular 2',size=10);

