#Social networkd ads prediction

import pandas as pd 
data =pd.read_csv("D:\IT DOCxxx\datasets\Social_Network_Ads.csv")
data.head()
data.shape #400 X 5
data.describe()
data.info()

import seaborn as sns
dir(sns)
sns.pairplot(data)
correlation=data.corr()
sns.heatmap(correlation, annot=True)
sns.boxplot(data['Gender'],data['Purchased'])

#Encoding the values
from sklearn.preprocessing import LabelEncoder as LE
le = LE()

data['Gender']=le.fit_transform(data['Gender'])

cols=data.columns
cols
y=data.iloc[:,4]
y.head()
y.shape

x=data.iloc[:,1:4]
x.head()

from sklearn.model_selection import train_test_split as tts 
xtrain, xtest, ytrain, ytest =tts(x,y,test_size=0.3,random_state=13)

#Model preparation 

#Random Forest

from sklearn.ensemble import RandomForestClassifier as RFC
rf_model=RFC(random_state=13)
rf_model.fit(xtrain,ytrain)
rf_pred=rf_model.predict(xtest)
accuracy(ytest,rf_pred)
#Out[56]: 0.8916666666666667

#tuning 
param={'criterion':['gini','entropy'],'max_depth':range(1,10),'max_leaf_nodes':range(2,10)}
rf_cv=gscv(rf_model,param,scoring='accuracy',n_jobs=-1)
rf_cv.fit(xtrain,ytrain)
rf_cv.best_params_
#Out[51]: {'criterion': 'gini', 'max_depth': 3, 'max_leaf_nodes': 5}

final_rf_model=RFC(criterion= 'gini', max_depth=3, max_leaf_nodes=5,random_state=13)

final_rf_model.fit(xtrain,ytrain)
final_rf_pred=final_rf_model.predict(xtest)

accuracy(ytest,final_rf_pred)
#Out[60]: 0.9166666666666666

#xGBoost
from xgboost import XGBClassifier as xgb
xgb_model=xgb()
xgb_model.fit(xtrain,ytrain)
xgb_pred=xgb_model.predict(xtest)

accuracy(xgb_pred,ytest)
#Out[71]: 0.9166666666666666

cm=confusion_matrix(xgb_pred,ytest)
cm
