#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '')


# # Scraping sencillo

# ### 1. pandas.read_clipboard

# Pandas tiene una función que permite copiar directamente una tabla de una página web a un dataframe
# 
# Por ejemplo, podemos ir a [ésta página](https://es.wikipedia.org/wiki/Anexo:Ciudades_europeas_por_poblaci%C3%B3n) que contiene una lista de las ciudades más pobladas de Europa, copiar los datos y pegarlos en un dataframe con `pd.read_clipboard`

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_clipboard()


# In[4]:


df.head(10)


# In[5]:


df.to_csv("poblacion.csv", index=False)


# ### 2. requests + extruct
# 
# En ciertos casos, las páginas web anotan su html con información adicional para ayudar a que las máquinas puedan leer sus datos, con el objetivo generalmente de ayudar a los motores de búsqueda a indexar sus productos. Esto se llama [la web semántica](https://es.wikipedia.org/wiki/Web_sem%C3%A1ntica)). 
# 
# Por ejemplo, si buscamos un producto en google, generalmente veremos productos recomendados directamente en la página de búsqueda. Esto es así por que las paginas que tienen dichos productos tienen un conjunto de etiquetas especiales.
# 
# Podemos usar la librería [extruct](https://github.com/scrapinghub/extruct) para extraer dichos datos de forma sencilla en aquellos casos donde la página proporcione las etiquetas semánticas.
# 
# Extruct se instala de forma sencilla con `pip install extruct`

# Por ejemplo, supongamos que queremos extraer información de teléfonos móviles de la tienda online [Pc Componentes](https://www.pccomponentes.com/smartphone-moviles
# ).
# 
# Como dicha tienda online tiene etiquetados sus productos para la web semántica, es muy facil el obtener los datos

# In[6]:


import requests
import extruct

respuesta = requests.get("https://www.pccomponentes.com/smartphone-moviles")
datos_tienda = extruct.extract(respuesta.text)


# In[13]:


from pprint import pprint
pprint(datos_tienda)


# In[14]:


import requests
import extruct

r = requests.get("http://www.recetas.net/receta/76/Champinones-al-Ajillo")
datos_receta = extruct.extract(r.text)


# In[15]:


from pprint import pprint
pprint(datos_receta)


# In[17]:


datos_receta.keys()


# In[16]:


pprint(datos_receta["microdata"][0]["properties"])


# In[18]:


def procesar_recetas_net(receta_url):
    r = requests.get(receta_url)
    datos_receta = extruct.extract(r.text)
    receta = datos_receta["microdata"][0]["properties"]
    return {
        "instrucciones": ''.join(receta["recipeInstructions"]),
        "ingredientes": receta["recipeIngredient"],
        "nombre": receta["name"]
    }


# In[19]:


receta_nueva = procesar_recetas_net("http://www.recetas.net/receta/266/Spaguetti-a-la-carbonara")
pprint(receta_nueva)


# ### 3. requests + parsel
# 
# `extruct` es una herramienta muy potente cuando es posible el utilizarla. Sin embargo, el usar extruct significa que sólo podemos extraer aquellos datos que los creadores de la página han decidido que merece la pena especificar.
# 
# Por ejemplo, en la página de `recetas.net`, hay información de los platos relativa a si un plato es vegetariano o no. Dicha información no está etiquetada semánticamente y por lo tanto, extruct no la puede procesar.
# 
# Para todos aquellos casos en los que necesitemos extraer información de forma flexible, tenemos que procesar el html de forma "manual". Esto se hace especificando a nuestro script que elementos de la estructura de la página web queremos.
# 
# Una herramienta que nos permite hacer esto de forma sencilla es [`parsel`](https://github.com/scrapy/parsel). Es similar a otra herramienta más famosa `beautifulsoup`, pero en mi opinión `parsel` es más sencillo de usar.
# 
# Se instala con pip de la forma habitual (`pip install parsel` *desde fuera del notebook*).

# In[20]:


import requests
from parsel import Selector

url = "http://www.recetas.net/receta/266/Spaguetti-a-la-carbonara"
r = requests.get(url)
sel = Selector(r.text)


# In[21]:


sel


# Con parsel hay varias formas de seleccionar elementos específicos del html. En concreto las más potentes son mediante:
# 
# - selectores css (usando el metodo `.css`): Los selectores css son una syntaxis que nos permite hacer selecciones complejas en el html basandose en los atributos de cada elemento, en particular en sus clases de estilo css
# - xpath (usando el método `.xpath`): XPath es otra forma similar a los selectores css que nos permite hacer selecciones complejas de html.

# Por ejemplo si queremos seleccionar todos los elementos de tipo `div`, lo hacemos asi:

# In[25]:


sel.css("div")


# El método `.css` siempre devuelve una lista, aunque solo haya un elemento que cumpla las condiciones de seleccion.

# Podemos concatenar selecciones facilmente, por ejemplo, si queremos todos los elementos de tipo `li` que están dentro de un tipo `div` lo hacemos asi:

# In[26]:


sel.css("div li")


# Si queremos extraer el html de una seleccion lo hacemos con el método `extract`:

# In[28]:


print(sel.css("div li").extract()[0])


# In[29]:


type(sel.css("div li")[0])


# In[30]:


type(sel.css("div li").extract()[0])


# El método `extract` siempre devuelve una lista, aunque haya solo un elemento que cumpla el criterio de selección.
# Si solo queremos extraer un elemento, podemos usar `extract_first`

# In[31]:


sel.css("div li").extract_first()


# El selector `div li` va a seleccionar todos los `li` que estén dentro de una etiqueta `div`. Sin embargo, si queremos solo aquellos elementos **inmediatamente dentro** de una etiqueta (lo que se llama *hijos*) `div` lo hacemos asi:

# In[34]:


sel.css("div>ul>li")


# que no devuelve nada por que no hay ningun elemento `li` inmediatamente dentro de un `div`

# Si queremos seleccionar un elemento que tenga una clase específica, lo hacemos con la sintaxis, `css("elemento.clase")`

# Supongamos que queremos extraer los ingredientes de la receta, si vemos el html de la página, la seccion que nos interesa tiene la forma:
# 
# ```
# <div class="col col-12 col-sm-8">
#   <article>
#     <header>
#       <hgroup>
#         <span id="ContentPlaceHolder1_LMetaTipoPlato" itemprop="recipeCategory" style="display:none;">Pastas y arroces</span>
#         <h2 itemprop="description">
#           <a id="ContentPlaceHolder1_HLTipoPlato" href="../../tipo-plato-busqueda/14/Pastas-y-arroces">Pastas y arroces</a>
#           <span id="ContentPlaceHolder1_LAuthor" itemprop="author" style="display:none;">Medialabs</span>
#         </h2>
#         <h1 itemprop="name">
#            SPAGUETTI A LA CARBONARA
#         </h1>
#       </hgroup>
#     </header>
#     <div class="ingredientes">
#       <h3> INGREDIENTES PARA 4 PERSONAS</h3>
#       <ul>
#         <li itemprop="recipeIngredient">20 gramos de  aceite </li>
#         <li itemprop="recipeIngredient">1 pizca de  sal </li>
#         <li itemprop="recipeIngredient">4  huevo, las yemas </li>
#         <li itemprop="recipeIngredient">1 pizca de  pimienta negra </li>
#         <li itemprop="recipeIngredient">150 gramos de  queso pecorino, Grana Padano </li>
#         <li itemprop="recipeIngredient">400 gramos de  spaguetti </li>
#         <li itemprop="recipeIngredient">120 gramos de  tocino ahumado </li>
#       </ul>
#    </div>
# ```

# O sea que hay un elemento `div` con la clase *ingredientes* que contiene dentro todos el listado de ingredientes (*los elementos `li` son elementos de una lista (list items)*)
# 
# Podemos seleccionar dicho div de la forma siguiente:

# In[36]:


print(sel.css("div.ingredientes li").extract_first())


# Ahora dentro de este div, para conseguir los ingredientes, tenemos que tomar todos los elementos de la lista `li` y extraer su texto. Podemos extraer el texto usando el selector `::text`

# In[37]:


ingredientes = sel.css("div.ingredientes li::text").extract()
ingredientes


# Supongamos tambien que queremos la puntuación, dificultad media, información sobre si la receta es vegetariana o no y el tiempo de preparación. Toda esta información está en una barra lateral izquierda.
# 
# El HTML tiene este aspecto:
# 
# ```
# 
#             <div class="otrosDet">
#               <div class="vota easing"> <span id="closeVota" class="close">X</span>
#                 <h2>Vota por esta receta</h2>
#                 <div class="score">
#                   <div class="graf">
#                     <div class="star" data-valor-voto="1">
#                       <div data-tipo="estrella-voto" class="star"></div>
#                     </div>
#                     <div class="star" data-valor-voto="2">
#                       <div data-tipo="estrella-voto" class="star"></div>
#                     </div>
#                     <div class="star" data-valor-voto="3">
#                       <div data-tipo="estrella-voto" class="star"></div>
#                     </div>
#                     <div class="star" data-valor-voto="4">
#                       <div data-tipo="estrella-voto" class="star"></div>
#                     </div>
#                     <div class="star" data-valor-voto="5">
#                       <div data-tipo="estrella-voto" class="star"></div>
#                     </div>
#                   </div>
#                 </div>
#               </div>
#               <input type="hidden" name="ctl00$ContentPlaceHolder1$HFNumeroComensalesIncial" id="HFNumeroComensalesIncial" value="4" />
#               <input type="hidden" name="ctl00$ContentPlaceHolder1$HFIdReceta" id="HFIdReceta" value="266" />
#               <p>Dificultad: <strong>
#                 Media
#                 </strong></p>
#               <p>Tiempo: <strong>
#                 <meta id="ContentPlaceHolder1_LMetaCookTime" itemprop="cookTime" content="PT30M"></meta>
#                 30
#                 min.</strong></p>
#               <p>Vegetariana: <strong>
#                 No
#                 </strong></p>
#               <p>Calorías: <strong>
#                 Medio
#                 </strong></p>
# ```

# Vemos que la información que nos interesa está en un div de clase `otrosDet`. Dentro de dicho div, hay párrafos (etiqueta `p`) y dentro hay elementos `strong` (se usan para poner texto en negrita), con el texto que nos interesa dentro.

# In[38]:


detalles = sel.css("div.otrosDet p strong::text").extract() 
detalles


# Vemos que estos elementos tienen un monton de *basura*, es decir, un monton de espacios y de simbolos (`\n y \r`) que no nos interesan, asi que los removemos.

# In[39]:


detalles = list(map(lambda d: d.replace("\r\n                ",""), detalles))
detalles


# De la misma forma podemos obtener la categoria de la receta

# Las instrucciones están en el div con clase `elaboracion`, en un conjunto de parrafos `p`:

# In[41]:


instrucciones = sel.css("div.elaboracion p::text").extract()
instrucciones


# Vemos que hay unos cuantos elementos que no necesitamos (`\r`, `\n` y espacios) al inicio y final de cada cadena de texto. Los podemos remover con `strip`

# In[42]:


instrucciones = list(map(lambda x: x.strip("\r\n "),instrucciones))
instrucciones


# Ahora podemos unir los textos con `join`

# In[43]:


instrucciones = ''.join(instrucciones)
instrucciones


# La puntuación está en el mismo `div` que los detalles de la receta, solo que en otro div de clase `score`

# In[44]:


sel.css("div.otrosDet div.score span.num::text").extract_first().strip()


# Ahora podemos ponerlo todo en una función que nos permita extraer una receta

# In[53]:


def procesar_receta(url):
    r  = requests.get(url)
    sel = Selector(r.text)
    
    categoria = sel.css("h2 a::text").extract_first() 
    titulo = sel.css("section.receta h1::text").extract_first().strip().capitalize() 
    
    instrucciones = sel.css("div.elaboracion p::text").extract()
    instrucciones = list(map(lambda x: x.strip("\r\n "),instrucciones))
    instrucciones = ''.join(instrucciones)
   
    ingredientes = sel.css("div.ingredientes li::text").extract()

    puntuacion = sel.css("div.otrosDet div.score span::text").extract_first().strip()
    
    detalles = sel.css("div.otrosDet p strong::text").extract() 
    detalles = list(map(lambda d: d.replace("\r\n                ",""), detalles))

    return {
        "categoria": categoria,
        "ingredientes": ingredientes,
        "instrucciones": instrucciones,
       "titulo": titulo,
       "puntuacion": puntuacion, 
       "dificultad": detalles[0],
       "tiempo": detalles[2],
       "vegetariana": detalles[3],
       "calorias": detalles[4]
        }


# In[50]:


receta = procesar_receta("http://www.recetas.net/receta/266/Spaguetti-a-la-carbonara")


# In[51]:


pprint(receta)


# In[54]:


pprint(procesar_receta("http://www.recetas.net/receta/40/Rollo-de-carne"))


# In[ ]:




