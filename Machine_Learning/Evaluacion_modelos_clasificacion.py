#!/usr/bin/env python
# coding: utf-8

# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets,metrics

cancer_datos=datasets.load_breast_cancer()
cancer_df=pd.DataFrame(cancer_datos.data,columns=cancer_datos.feature_names)
cancer_df['objetivo']=cancer_datos.target
cancer_df['objetivo']=cancer_df.objetivo.replace({0:1,1:0})
cancer_df.objetivo.value_counts(True)


# In[25]:


cancer_df.head()


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X=cancer_df[cancer_datos.feature_names]
y=cancer_df.objetivo
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
modelo=LogisticRegression()
modelo.fit(X_train,y_train)
predicciones=modelo.predict(X_test)
clases_reales=y_test
predicciones_probabilidades=modelo.predict_proba(X_test)


# In[27]:


def tupla_clase_predicciones(y_real,y_pred):
    return list(zip(y_real,y_pred))
tupla_clase_predicciones(clases_reales,predicciones)[:10]


# In[28]:


def VP(clases_reales,predicciones):
    par_clase_prediccion=tupla_clase_predicciones(clases_reales,predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==1])
def VN(clases_reales,predicciones):
    par_clase_prediccion=tupla_clase_predicciones(clases_reales,predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==0])
def FP(clases_reales,predicciones):
    par_clase_prediccion=tupla_clase_predicciones(clases_reales,predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==1])
def FN(clases_reales,predicciones):
    par_clase_prediccion=tupla_clase_predicciones(clases_reales,predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==0])
print("""
Verdaderos Positivos: {}
Verdaderos Negativos: {}
Falsos Positivos: {}
Falsos Negativos: {}
""".format(VP(clases_reales,predicciones),
          VN(clases_reales,predicciones),
          FP(clases_reales,predicciones),
          FN(clases_reales,predicciones)))


# In[29]:


#Funcion de exactitud definida a mano
def exactitud(clases_reales,predicciones):
    vp=VP(clases_reales,predicciones)
    vn=VN(clases_reales,predicciones)
    return (vp+vn)/len(clases_reales)
exactitud(clases_reales,predicciones)


# In[30]:


#Utilizando sklearn(Hace lo mismo que la anterior )
metrics.accuracy_score(clases_reales,predicciones)


# In[31]:


#Funcion de precision definida a mano
def precision(clases_reales,predicciones):
    vp=VP(clases_reales,predicciones)
    fp=FP(clases_reales,predicciones)
    return vp/(vp+fp)
precision(clases_reales,predicciones)


# In[32]:


#Utilizando sklearn(Hace lo mismo que la anterior )
metrics.average_precision_score(clases_reales,predicciones)


# In[33]:


#Funcion de sensibilidad definida  a mano
def sensibilidad(clases_reales,predicciones):
    vp=VP(clases_reales,predicciones)
    fn=FN(clases_reales,predicciones)
    return vp/(vp+fn)
sensibilidad(clases_reales,predicciones)


# In[34]:


#Utilizando sklearn(Hace lo mismo que la anterior )
metrics.recall_score(clases_reales,predicciones)


# In[35]:


#Matriz de confusion de sklearn
from sklearn.metrics import confusion_matrix
confusion_matrix(clases_reales,predicciones)


# In[36]:


#La puntuacion f1_score es una media ponderada entre la sensibilidad y la precision
metrics.f1_score(clases_reales,predicciones)


# In[37]:


#Radio de falsos positivos , da una medida  de las probabilidades de nuestro modelo de asignar una
# clase positiva a un caso negativo (probabilidad de error del modelo)
def fpr(clases_reales,predicciones):
    return (FP(clases_reales,predicciones)/(FP(clases_reales,predicciones)+VN(clases_reales,predicciones)))
fpr(clases_reales,predicciones)


# In[38]:


df=pd.DataFrame({'clase_real':clases_reales,
                 'clase_pred':predicciones,
                 'probabilidad_0':modelo.predict_proba(X_test)[:,0],
                 'probabilidad_1':modelo.predict_proba(X_test)[:,1], 
})
df['sum_probas']=df.probabilidad_0+df.probabilidad_1
df.head()


# In[39]:


df.query('probabilidad_0>0.5 & clase_pred==1')#Comprovando la valides de las probabilidades de 0


# In[40]:


df.query('probabilidad_1>0.5 & clase_pred==0')#Comprovando la valides de las probabilidades de 1


# In[41]:


def probabilidades_a_clase(predicciones_probabilidades,umbral=0.5):
    prediccion=np.zeros([len(predicciones_probabilidades), ])
    prediccion[predicciones_probabilidades[:,1]>=umbral]=1
    return prediccion
predicciones_probabilidades[:10]


# In[42]:


probabilidades_a_clase(predicciones_probabilidades,umbral=0.5)[:10]


# In[43]:


from ipywidgets import interact,widgets,fixed

@interact(umbral=widgets.FloatSlider(min=0.01,max=0.99,step=0.1,value=0.01))
def evaluar_umbral(umbral):
    prediccion_en_umbral=probabilidades_a_clase(predicciones_probabilidades,umbral)
    sensibilidad_umbral=metrics.recall_score(clases_reales,prediccion_en_umbral)
    fpr_umbral=fpr(clases_reales,prediccion_en_umbral)
    precision_umbral=precision(clases_reales,prediccion_en_umbral)
    print("""
    Precision: {:.3f},
    Sensibilidad: {:.3f},
    Radio de Alarma: {:.3f},""".format(precision_umbral,sensibilidad_umbral,fpr_umbral))


# In[44]:


def evaluar_umbral(umbral):
    prediccion_en_umbral=probabilidades_a_clase(predicciones_probabilidades,umbral)
    sensibilidad_umbral=metrics.recall_score(clases_reales,prediccion_en_umbral)
    fpr_umbral=fpr(clases_reales,prediccion_en_umbral)
    precision_umbral=precision(clases_reales,prediccion_en_umbral)
    return precision_umbral,sensibilidad_umbral,fpr_umbral

rango_umbral= np.linspace(0., 1.,1000)
sensibilidad_umbrales=[]
precision_umbrales=[]
fpr_umbrales=[]

for umbral in rango_umbral:
    precision_umbral,sensibilidad_umbral,fpr_umbral=evaluar_umbral(umbral)
    precision_umbrales.append(precision_umbral)
    sensibilidad_umbrales.append(sensibilidad_umbral)
    fpr_umbrales.append(fpr_umbral)


# In[45]:


plt.plot(sensibilidad_umbrales,precision_umbrales);
plt.ylabel('Precision')
plt.xlabel('Radio de Verdaderos positivos (sensibilidad)')
plt.title('Curva de Presicion-Recall');


# In[46]:


def gradica_precision_recall(clases_reales,predicciones_probabilidades):
    precision_, recall_,_ =metrics.precision_recall_curve(
    clases_reales,predicciones_probabilidades[:,1])
    plt.step(recall_ ,precision_,color='b',alpha=0.2,where='post')
    plt.fill_between(recall_,precision_,step='post',alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.05])
    plt.title('Curva Precision-Recall');
    plt.show()

gradica_precision_recall(clases_reales,predicciones_probabilidades)


# In[47]:


plt.plot(fpr_umbrales,sensibilidad_umbrales);
plt.xlabel('Radio de Falsos positivos')
plt.ylabel('Radio de Verdaderos positivos')
plt.title('Curva ROC');


# In[48]:


metrics.roc_auc_score(clases_reales,predicciones)


# In[49]:


def grafica_curva_auc(clases_reales,predicciones,predicciones_probabilidades):
    fpr,tpr,_ =metrics.roc_curve(clases_reales,predicciones_probabilidades[:,1])
    roc_auc=metrics.roc_auc_score(clases_reales,predicciones)
    plt.figure()
    
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='Curva ROC (area =%0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--',label='estimador aleatorio')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show();

grafica_curva_auc(clases_reales,predicciones,predicciones_probabilidades)


# In[50]:


def evaluar_modelos(clases_reales,predicciones,predicciones_probabilidades):
    exactitud=metrics.accuracy_score(clases_reales,predicciones)
    precision=metrics.average_precision_score(clases_reales,predicciones)
    sensibilidad=metrics.recall_score(clases_reales,predicciones)
    roc_auc=metrics.roc_auc_score(clases_reales,predicciones)
    f1=metrics.f1_score(clases_reales,predicciones)
    print("""
    Exactitud: {:.3f}
    Precision: {:.3f}
    Sesibilidad: {:.3f}
    Area bajo curva(AUC): {:.3f}
    Puntuacion F1: {:.3f}
    
    """.format(exactitud,precision,sensibilidad,roc_auc,f1))
evaluar_modelos(clases_reales,predicciones,predicciones_probabilidades)
    


# In[51]:


cancer_df.objetivo.value_counts(True)


# In[52]:


len ([predi for predi in predicciones if predi==1])#Nuemero de predicciones para clase 1


# In[53]:


len ([predi for predi in predicciones if predi==0])#Numero de predicciones para clase 0


# In[54]:


#Evaluacion  de modelos Parte 3 seleccion del umbral
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_cancer = datasets.load_breast_cancer()
df_cancer= pd.DataFrame(data_cancer.data,columns=data_cancer.feature_names)
df_cancer['objetivo']=data_cancer.target
df_cancer['objetivo']=df_cancer.objetivo.replace({0:1,1:0})

X=df_cancer[data_cancer.feature_names]
y=df_cancer['objetivo']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
modelo=LogisticRegression()
modelo.fit(X_train,y_train)

predicciones=modelo.predict(X_test)
clases_rales=y_test
predicciones_probabilidades=modelo.predict_proba(X_test)

proba=modelo.predict_proba(X_test)[:5]
proba


# In[55]:


umbral_decision=0.5
proba[:,1]>umbral_decision


# In[56]:


#Para convertir el coste de un negocion a probabilidades
def softmax(coste_fp,coste_fn):
    return np.exp(coste_fp)/(np.exp(coste_fn)+np.exp(coste_fp))
coste_fn=1
coste_fp=2
softmax(coste_fp,coste_fn)


# In[57]:


from ipywidgets import widgets,interact
@interact
def calculo_umbral(
    coste_fp=widgets.FloatLogSlider(min=1,max=10,step=0.1,value=1),
    coste_fn=widgets.FloatLogSlider(min=1,max=10,step=0.1,value=1),):
    return softmax(coste_fp,coste_fn)


# In[58]:


coste_fn=10
coste_fp=1
umbral_decision=calculo_umbral(coste_fp,coste_fn)
print(umbral_decision)
decisiones=probabilidades_a_clase(proba,umbral)
decisiones# Nota debe predicir todo como 1 esta al revez


# In[66]:


class BusinessLogisticRegression(LogisticRegression):
  
    def decision_de_negocio(self,X,coste_fp=1,coste_fn=1,*args,**kwargs):
        probs=self.predict_proba(X)
        umbral_decision=calculo_umbral(coste_fp,coste_fn)
        print('Umbral de decision: {}'.format(umbral_decision))
        decisiones=probabilidades_a_clase(probs,umbral_decision)
        return decisiones

modelo_negocio=BusinessLogisticRegression()
modelo_negocio.fit(X_train,y_train)


# In[61]:


modelo_negocio.predict(X_test[:5])


# In[62]:


modelo_negocio.predict_proba(X_test[:5])


# In[63]:


modelo_negocio.decision_de_negocio(X_test[:5],1,1)


# In[67]:


@interact(
    coste_fp=widgets.FloatLogSlider(min=1,max=10,step=0.1,value=1),
    coste_fn=widgets.FloatLogSlider(min=1,max=10,step=0.1,value=1)
)
def decision_negocion(coste_fp,coste_fn):
    predicciones=modelo_negocio.decision_de_negocio(X_test,coste_fp,coste_fn)
    print(confusion_matrix(clases_rales,predicciones))


# In[ ]:




