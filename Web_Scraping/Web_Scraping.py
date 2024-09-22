#!/usr/bin/env python
# coding: utf-8

# # Peticiones HTTP con Requests

# In[1]:


import requests

respuesta=requests.get('http://www.meneame.net')


# In[2]:


respuesta.ok


# In[3]:


respuesta.status_code


# In[4]:


print(respuesta.content)


# In[5]:


with open ('meneame.html','wb') as fname:
    fname.write(respuesta.content)


# In[6]:


import webbrowser

webbrowser.open('meneame.html')


# In[7]:


respuesta=requests.get('http://www.reddit.com/r/food')


# In[8]:


respuesta.ok 


# In[9]:


respuesta.status_code


# In[10]:


respuesta.reason


# # Variante para si no funciona un request

# In[11]:


respuesta=requests.get('https://www.reddit.com/r/food/.json?limit=5') # Aqui se hace asi portque reddit tiene una app


# In[12]:


respuesta.ok # En este caso falla por haber hecho muchas peticiones (429)


# In[13]:


respuesta.status_code


# In[14]:


respuesta.reason


# In[15]:


datos=respuesta.json() # si no hubiera fallado la info viene en formato json y se puede procesar con json


# In[16]:


from pprint import pprint # se utiliza para visualizar la info de forma mas legible

pprint(datos)


# In[17]:


datos['data']['children'][1]['data']['title'] # buscando datos en el diccionario de diccionarios el 1 significa el primer post


# In[19]:


titulo= []
for post in datos['data']['children']:
    titulo.append(post['data']['title'])
titulo


# # Alternativa mas rapida

# In[21]:


from glom import glom

glom(datos,('data.children',['data.title']))


# # Yahoo Wheather

# In[32]:


url = """
https://query.yahooapis.com/v1/public/yql?q=select * from weather.forecast where woeid in (select woeid from geo.places(1) where text="{}") and u='c'&format=json
"""
url


# In[34]:


url.format("Murcia, Spain")


# In[35]:


datos=requests.get(url.format("Murcia,Spain")).json()


# In[ ]:


temperaturas=glom(datos,{
    'date':('query.results.channel.item.forecast',['date']),
    'high':('query.results.channel.item.forecast',['high']),
    'low':('query.results.channel.item.forecast',['low'])
})


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize']=(8,8)

plt.style.use('ggplot')

plt.plot(range(len(temperaturas['date'])),temperaturas['high'],label='Maxima')
plt.plot(range(len(temperaturas['date'])),temperaturas['low'],label='Minimas')
plt.xticks(range(len(temperaturas['date'])),temperaturas['date'],rotation=45,ha='right')
plt.title('Pronostico del tiempo');
plt.lengend();


# In[1]:


# Nota : no se puede visualizar por tener error con el acceso a la pagina


# In[3]:


import scrapy

get_ipython().run_line_magic('pinfo', 'scrapy.Response')


# In[ ]:




