#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn  import datasets
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
datos=datasets.load_boston()
boston=pd.DataFrame(datos.data,columns=datos.feature_names)
boston['objetivo']=datos.target
boston.head()


# In[2]:


class ClaseTest():
    'Esto es el docstring'


# In[3]:


ClaseTest.__doc__


# In[4]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet,Lasso,Ridge

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def  rmse_cv(estimador,X,y):
    preds=estimador.predict(X)
    return rmse(y,preds)

resultado={}
estimador_arbol=DecisionTreeRegressor()
error_cv=cross_val_score(estimador_arbol,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['arbol']=error_cv


estimador_elnet=ElasticNet()
resultado['elastinet']=cross_val_score(estimador_elnet,X=boston[datos.feature_names],y=boston['objetivo'],
                                      scoring=rmse_cv,cv=10).mean()

estimador_lassso=Lasso()
resultado['lasso']=cross_val_score(estimador_lassso,X=boston[datos.feature_names],y=boston['objetivo'],
                                  scoring=rmse_cv,cv=10).mean()

estimador_ridge=Ridge()
resultado['ridge']=cross_val_score(estimador_ridge,X=boston[datos.feature_names],y=boston['objetivo'],
                                  scoring=rmse_cv,cv=10).mean()

resultado


# ### Bagging

# In[5]:


from sklearn.ensemble import BaggingRegressor,BaggingClassifier
print(BaggingRegressor.__doc__)


# In[6]:


estimador_bagging_10=BaggingRegressor(n_estimators=10)
error_cv=cross_val_score(estimador_bagging_10,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['bagging_arbol_10']=error_cv
error_cv


# In[7]:


estimador_bagging_100=BaggingRegressor(n_estimators=100)
error_cv=cross_val_score(estimador_bagging_10,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['bagging_arbol_100']=error_cv
error_cv


# In[8]:


estimador_baggging_elnet=BaggingRegressor(n_estimators=100,base_estimator=ElasticNet())
error_cv=cross_val_score(estimador_baggging_elnet,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['baggin_elnet']=error_cv
error_cv


# In[9]:


from sklearn.tree import ExtraTreeRegressor

estimador_baggin_elnet_Ex=BaggingRegressor(n_estimators=100,base_estimator=ExtraTreeRegressor())
error_cv=cross_val_score(estimador_baggin_elnet_Ex,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['bagging_elnet_ex']=error_cv
error_cv


# ### Boosting

# In[10]:


from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier

print(AdaBoostRegressor.__doc__)


# In[11]:


estimador_adaboost=AdaBoostRegressor(n_estimators=100)

error_cv=cross_val_score(estimador_adaboost,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['adaboost']=error_cv
error_cv


# ### Gradient Boosting (GBRT)

# In[12]:


from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
print(GradientBoostingRegressor.__doc__)


# In[13]:


estimador_gradientboost=GradientBoostingRegressor(n_estimators=100,loss='ls')
error_cv=cross_val_score(estimador_gradientboost,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['gradientboost_100']=error_cv
error_cv


# In[14]:


estimador_gradientboost.fit(boston[datos.feature_names],boston['objetivo'])

importancia_variables=estimador_gradientboost.feature_importances_
importancia_variables=100.0*(importancia_variables/importancia_variables.max())
sorted_idx=np.argsort(importancia_variables)
pos=np.arange(sorted_idx.shape[0])+ .5
plt.barh(pos,importancia_variables[sorted_idx],align='center')
plt.yticks(pos,datos.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# ### Bosques Aleatorios (Random Forest)

# In[15]:


from sklearn.ensemble import RandomForestRegressor
print(RandomForestRegressor.__doc__)


# In[16]:


estimador_randormforest=RandomForestRegressor(n_estimators=100)

error_cv=cross_val_score(estimador_randormforest,X=boston[datos.feature_names],y=boston['objetivo'],
                        scoring=rmse_cv,cv=10).mean()
resultado['randomforest_100']=error_cv
error_cv


# ### XGBoost

# In[17]:


get_ipython().system('pip install xgboost')


# In[17]:


from xgboost import XGBRegressor
print(XGBRegressor.__doc__)


# In[18]:


estimador_xgboost=XGBRegressor(n_estimators=100)
error_cv=cross_val_score(estimador_xgboost,X=boston[datos.feature_names],y=boston.objetivo,
                        scoring=rmse_cv,cv=10).mean()
resultado['xgboost']=error_cv
error_cv


# In[19]:


from xgboost import plot_importance,to_graphviz

estimador_xgboost.fit(boston[datos.feature_names],boston.objetivo)
plot_importance(estimador_xgboost);


# In[20]:


to_graphviz(estimador_xgboost,num_trees=10,rankdir='LR')


# ### Stacking

# In[24]:


get_ipython().system('pip install mlxtend')


# In[21]:


from mlxtend.regressor import StackingRegressor

print(StackingRegressor.__doc__)


# In[22]:


estimador_stacking=StackingRegressor(
    regressors=[
        BaggingRegressor(n_estimators=100),
        AdaBoostRegressor(n_estimators=100),
        GradientBoostingRegressor(n_estimators=100),
        RandomForestRegressor(n_estimators=100)],
    meta_regressor=XGBRegressor(n_estimators=100)
)

error_cv=cross_val_score(estimador_stacking,X=boston[datos.feature_names],y=boston.objetivo,
                        scoring=rmse_cv,cv=10).mean()
resultado['stacking']=error_cv
error_cv


# In[23]:


resultado

