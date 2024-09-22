#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# ### Intro a peticiones HTTP con Requests
# 
# En esta sección vamos a ver como usar la libreria [`requests`](https://github.com/requests/requests), que es la librería mas comunmente utilizada para hacer peticioes HTTP en python. Se instala desde Anaconda o pip (con `pip install requests`)

# In[4]:


import requests


# ### GET Requests
# 
# Cuando queremos obtener algo de una página web (un *recurso* como se llaman de forma más técnica), lo que nuestro navegador hace es una petición HTTP de tipo **GET**. 

# Por ejemplo, si queremos hacer una petición a meneame.net lo hacemos de forma sencilla de esta forma:

# In[5]:


respuesta = requests.get("http://www.meneame.net")


# Podemos verificar que la petición ha tenido exito con el atributo `ok`

# In[6]:


respuesta.ok


# Y podemos ver el status de la respuesta con `status_code`

# In[7]:


respuesta.status_code


# En este caso es un **200** por que la petición ha sido recibida con éxito

# podemos ver el contenido que nos ha enviado el servidor con `content`

# In[8]:


print(respuesta.content)


# Vemos que la petición a google nos ha devuelto el html de la página de inicio de google. Podemos escribirlo a un archivo y verlo

# In[10]:


with open("meneame.html", "wb") as fname:
    fname.write(respuesta.content)


# In[11]:


import webbrowser
webbrowser.open("meneame.html")


# Vemos que la página que hemos obtenido via `requests` no es la misma que la que veríamos desde el navegador. Esto es por que `requests` no sabe como interpretar javascript y dicha página genera los contenidos de forma dinámica.

# ### Reddit
# 
# Supongamos ahora que queremos obtener los 5 primeros posts del foro de comida en Reddit. Podemos hacer una petición a www.reddit.com/r/food (que es la misma url que usariamos en el navegador).

# In[12]:


respuesta = requests.get("http://www.reddit.com/r/food")


# podemos verificar que la peticion ha sido correcta comprobando el status de la respuesta:

# In[13]:


respuesta.ok


# Oh que sorpresa! La petición no ha funcionado, veamos que status hemos recibido.

# In[14]:


respuesta.status_code


# Sabemos que los códigos http que empiezan por *4xx* significan errores en la petición que hemos hecho. Podemos mirar lo que significa el código 429 en una [lista](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent), o podemos mirar la razon del error con el atributo `reason`

# In[15]:


respuesta.reason


# Sin embargo, si abrimos esa dirección web (url) en el navegador funciona sin problemas.

# In[16]:


webbrowser.open("www.reddit.com/r/food")


# Vaya! Nos dice el servidor de reddit que hemos hecho demasiadas peticiones, ¿como puede ser si solo hemos hecho una?. Hemos mencionado que ciertas páginas web tienen un conjunto de contramedidas para evitar bots (programas que pretenden ser usuarios).
# 
# Por este motivo, podemos hacer uso de la api que reddit nos proporciona, como dicha api está diseñada para ser consumida por bots, no tendremos problema.

# In[17]:


respuesta = requests.get("https://www.reddit.com/r/food/.json?limit=5")


# In[18]:


respuesta.ok


# In[19]:


print(respuesta.content)


# la petición de la api viene en formato json, con requests podemos parsearla facilmente:

# In[20]:


datos = respuesta.json()


# In[22]:


from pprint import pprint


# In[23]:


pprint(datos)


# Vemos que datos es un diccionario con la clave `data` y subclave `children` que tiene los posts en una lista. Dentro de cada lista, la clave `title` tiene el titulo del post. Por ejemplo, si queremos el título del segundo post en la lista:

# In[26]:


datos['data']['children'][1]['data']['title']


# Asi que si queremos todos los titulos no tenemos más que iterar los posts:

# In[27]:


titulos = []
for post in datos['data']['children']:
    titulos.append(post['data']['title'])
    
titulos


# Alternativamente, podemos usar [`glom`](https://github.com/mahmoud/glom) para iterar más rápidamente. **NOTA:** `glom` es una libreria nueva y puede no funcionar de forma estable

# In[28]:


from glom import glom
glom(datos, ('data.children', ['data.title']))


# ### Yahoo Weather
# 
# Podemos usar la [api de Yahoo weather](https://developer.yahoo.com/weather/?guccounter=2) de forma similar a la de reddit

# In[29]:


url = """
https://query.yahooapis.com/v1/public/yql?q=select * from weather.forecast where woeid in (select woeid from geo.places(1) where text="{}") and u='c'&format=json
"""

url


# In[30]:


url.format("Murcia, Spain")


# In[33]:


datos = requests.get(url.format("Madrid, Spain")).json()


# In[34]:


from pprint import pprint
pprint(datos)


# In[35]:


from glom import glom
temperaturas = glom(datos, {
    "date": ('query.results.channel.item.forecast', ['date']),
    "high":  ('query.results.channel.item.forecast', ['high']),
    "low":  ('query.results.channel.item.forecast', ['low'])
})


# In[36]:


temperaturas


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

plt.rcParams["figure.figsize"] = (7, 7)


# In[38]:


plt.style.use("ggplot")


# In[39]:


plt.plot(range(len(temperaturas["date"])), temperaturas["high"], 
         label="Máxima")

plt.plot(range(len(temperaturas["date"])), temperaturas["low"], 
         label="Mínima")
plt.xticks(range(len(temperaturas["date"])), temperaturas["date"],
           rotation=45, ha="right")
plt.title("Pronóstico de temperatura para Madrid");
plt.legend();


# In[40]:


def dibujar_grafico_temperaturas(temperaturas, ciudad):
    plt.plot(range(len(temperaturas["date"])), temperaturas["high"], 
         label="Máxima")
    plt.plot(range(len(temperaturas["date"])), temperaturas["low"], 
         label="Mínima")
    plt.xticks(range(len(temperaturas["date"])), temperaturas["date"],
           rotation=45, ha="right")
    plt.legend()
    plt.title("Pronóstico de temperatura para {}".format(ciudad));
    
def obtener_temperaturas(ciudad):
    datos = requests.get(url.format(ciudad)).json()
    temperaturas = glom(datos, {
        "date": ('query.results.channel.item.forecast', ['date']),
        "high":  ('query.results.channel.item.forecast', ['high']),
        "low":  ('query.results.channel.item.forecast', ['low'])
    })
    return temperaturas

def grafico_temperaturas(ciudad):
    temperaturas = obtener_temperaturas(ciudad)
    dibujar_grafico_temperaturas(temperaturas, ciudad)


# In[42]:


grafico_temperaturas("New York City")


# In[ ]:




