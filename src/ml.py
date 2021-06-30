
# Importing required libraries
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

"""# Data Collection

Data Set Creation using Web Scraping
"""
print("Loading Dataframes") 
import bs4 as bs
import pandas as pd
from urllib.request import Request, urlopen

df_list = []        #list storing dataframe of past 10 years
r_values = ['210008','200061','190075','180084','170099','160058','150059','140052','130034','120002']         #webpage values for datascraping
csv_files = ['20.csv','19.csv','18.csv','17.csv','16.csv','15.csv','14.csv','13.csv','12.csv','11.csv']        #attaching CSV files from folder
#Scraping data
for i,val in enumerate(r_values):
    url = 'https://sofifa.com/teams?lg=13&r='+val+'&set=true'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    webpage = urlopen(req).read()

    webpage = bs.BeautifulSoup(webpage,'lxml')

    table=webpage.find('table')
    table_rows=table.find_all('tr')

    info=[]
    for tr in table_rows:
        td=tr.find_all('td')
        row=[i.text.strip() for i in td]
        info.append(row)
    
    td = table.find_all("div",{"class":"bp3-text-overflow-ellipsis"})
    Names = []
    for idx,data in enumerate(td):
        if idx%2 == 0:
            Names.append(data.text)

    df = pd.DataFrame(info)

    df=df.drop([0,1],axis=1)
    df=df.drop([0],axis=0)

    df['Team']=Names
    os.chdir("..")
    os.chdir("data")
    res=pd.read_csv(csv_files[i])
    os.chdir("..")
    os.chdir("src")

    teamsset = set(res["HomeTeam"].tolist())
    teamsset = list(teamsset)
    Names.sort()
    
    teamsset_cleaned = [x for x in teamsset if str(x) != 'nan']

    teamsset_cleaned.sort()
    
    #the team names in csv are shortened team names whereas those on webpage are actual team names
    #inorder to merge the teams data correctly we create a dictionary of shortened team names corresponding to the actual ones
    #we replace the team names derived from csv to the actual names correspondingly
    dictionary = dict(zip(Names,teamsset_cleaned))
    
    teamslist = df["Team"].tolist()

    for index,teams in enumerate(teamslist):
        teamslist[index] = dictionary[teams]
    
    df["Team"] = teamslist
    
    df_home=df.copy()
    df_away=df.copy()     
    #adding the names of the newly added features after scraping
    df_home.columns=['Home_OVA','Home_ATT','Home_MID','Home_DEF','Home_TRANSFER','Home_PLAYERS','Home_HITS','HomeTeam']
    df_away.columns=['Away_OVA','Away_ATT','Away_MID','Away_DEF','Away_TRANSFER','Away_PLAYERS','Away_HITS','AwayTeam']

    #merging the dataset with the scraped data
    res=pd.merge(res,df_home,on="HomeTeam",how="left")
    res=pd.merge(res,df_away,on="AwayTeam",how="left")
    
    #removing unwanted features like referee names, team names, betting odds
    res = res[['HTHG','HTAG','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR','Home_OVA','Home_ATT','Home_MID','Home_DEF','Home_TRANSFER','Home_PLAYERS','Home_HITS','Away_OVA','Away_ATT','Away_MID','Away_DEF','Away_TRANSFER','Away_PLAYERS','Away_HITS','FTR']]
    
    df_list.append(res)
    print("Dataframe made for year: 20"+str(20-i)+"-20"+str(20-i+1))     #success message after receiving the dataset in our list

df = pd.DataFrame()
for i in df_list:
    df = pd.concat([df,i], ignore_index=True)
df

"""## Data Preprocessing"""

df = df.dropna()
df = df.reset_index(drop=True)
df

"""Encoding Categorical Variables"""

def categorize_victory(x):
    if x=='A':
        return 0
    if x=='D':
        return 1
    if x=='H':
        return 2

df['FTR'] = df['FTR'].apply(categorize_victory)
df

def trim(x):
    x=str(x)
    return x[1:len(x)-1]

df['Home_TRANSFER'] = df['Home_TRANSFER'].apply(trim)
df['Away_TRANSFER'] = df['Away_TRANSFER'].apply(trim)

def value_to_float(x):
    x=str(x)
    if 'K' in x:
        return (float(x.replace('K', '')) * 1000)
    return (x)

df['Home_HITS'] = df['Home_HITS'].apply(value_to_float)
df['Away_HITS'] = df['Away_HITS'].apply(value_to_float)

df['Home_OVA'] = pd.to_numeric(df['Home_OVA'])
df['Home_ATT'] = pd.to_numeric(df['Home_ATT'])
df['Home_DEF'] = pd.to_numeric(df['Home_DEF'])
df['Home_MID'] = pd.to_numeric(df['Home_MID'])
df['Home_TRANSFER'] = pd.to_numeric(df['Home_TRANSFER'])
df['Home_PLAYERS'] = pd.to_numeric(df['Home_PLAYERS'])
df['Home_HITS'] = pd.to_numeric(df['Home_HITS'])
df['Away_OVA'] = pd.to_numeric(df['Away_OVA'])
df['Away_ATT'] = pd.to_numeric(df['Away_ATT'])
df['Away_DEF'] = pd.to_numeric(df['Away_DEF'])
df['Away_MID'] = pd.to_numeric(df['Away_MID'])
df['Away_TRANSFER'] = pd.to_numeric(df['Away_TRANSFER'])
df['Away_PLAYERS'] = pd.to_numeric(df['Away_PLAYERS'])
df['Away_HITS'] = pd.to_numeric(df['Away_HITS'])

"""## Feature Selection

Feature Selection using Chi-squared as scoring function
"""
print("Feature Importance based on SelectKBest")
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = df.iloc[:,0:28]
#X = df_list[0].iloc[:,np.r_[1:16, 98:]]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 10 best features

"""Feature Selection using F-Classif as scoring function

(Tried but Chi^2 is giving better results)
"""

# import numpy as np
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
# X = df.iloc[:,0:28]
# #X = df_list[0].iloc[:,np.r_[1:16, 98:]]  #independent columns
# y = df.iloc[:,-1]    #target column i.e price range
# #apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=f_classif, k=10)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(20,'Score'))  #print 10 best features

"""
Correlation Matrix

#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

df
"""

"""Reduced Dataset obtained using Chi^2 K_Best Feature Selection"""

reduced_df = df[['Home_HITS','Away_HITS','Home_TRANSFER','Away_TRANSFER','HST','HTAG','AST','HTHG','AS','HS']]

X=reduced_df

"""Train_Test_Split"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Feature Scaling using MinMaxScalar"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


"""
Value Counts

y.value_counts()

Count Plot

sns.set_style('whitegrid')
sns.countplot(x='FTR', data=df)

Pair Plot


#scatterplot
sns.set()
sns.pairplot(reduced_df, height = 2.5)
plt.show();

"""#Logistic Regression


#Logistic Regression without hyper-parameter tuning


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial',max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn import metrics
LR_accuracy = metrics.accuracy_score(y_test, y_pred)
LR_accuracy

"""Hyper-tuning Logistic Regression using Grid Search"""

#GRID_SEARCH ON LR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=2000)
grid={"C":[0.1,0.01,0.001], "penalty":['l1','l2'], "solver":['lbfgs','newton-cg','sag','saga'],"multi_class":['multinomial']}
LR_CV = GridSearchCV(LR, param_grid=grid, n_jobs=-1, verbose=3, cv = 5)
LR_CV.fit(X_train, y_train)

LR_CV.best_params_

from sklearn.metrics import plot_confusion_matrix

"""Logistic Regression after Hyper-parameter Tuning"""

y_pred = LR_CV.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
LR_GS_accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""#Gaussian NB

Gaussian Naive Bayes without hyper-parameter tuning
"""

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
nb_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
NB_accuracy = accuracy_score(y_test,nb_pred)
print("Accuracy: ",accuracy_score(y_test,nb_pred))
print(classification_report(y_test,nb_pred))
cm = confusion_matrix(y_test,nb_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""Hyper-tuning Gaussian Naive Bayes using Grid Search"""

#With GridSearch
from sklearn.model_selection import GridSearchCV
NB = GaussianNB()
parameters = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(NB,parameters,cv=5,verbose=5,scoring='accuracy')
gs_NB.fit(X_train,y_train)
gsnb_pred = model.predict(X_test)

"""Gaussian Naive Bayes after Hyper-parameter Tuning"""

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
NB_GS_accuracy = accuracy_score(y_test,gsnb_pred)
print("Accuracy: ", accuracy_score(y_test,gsnb_pred))
print(classification_report(y_test,gsnb_pred))

cm = confusion_matrix(y_test,gsnb_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""#SVC

Support Vector Classifier without Hyper-parameter tuning
"""

# "Support vector classifier"
from sklearn.svm import SVC
model = SVC(kernel = 'linear', C = 1,decision_function_shape='ovo')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
SVM_accuracy = metrics.accuracy_score(y_test, y_pred)
SVM_accuracy
# 0.6504297994269341

"""Hyper-tuning Support Vector Classifier with Grid Search"""

#Using GSCV on SVC
from sklearn.model_selection import GridSearchCV
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# parameters = {'max_leaf_nodes': list(range(2, 100)),
#           'min_samples_split': [2, 3, 4]}
svc = SVC(kernel = 'linear', C = 1,decision_function_shape='ovo')
model = GridSearchCV(svc, parameters, cv=5, scoring = 'accuracy', verbose=50, n_jobs=-1)
model.fit(X_train,y_train)
model.best_params_

"""SVC after Hyper-parameter tuning"""

svc_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
SVM_GS_accuracy = accuracy_score(y_test,svc_pred)
print("Accuracy: ", accuracy_score(y_test,svc_pred))
print("Classification Report: \n",classification_report(y_test,svc_pred))
cm = confusion_matrix(y_test,svc_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""#Decision Tree

Decision Tree without hyper-parameter tuning
"""

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=42)
model.fit(X_train,y_train)

dt_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
DT_accuracy = accuracy_score(y_test,dt_pred)
print("Accuracy: ",accuracy_score(y_test,dt_pred))
print(classification_report(y_test,dt_pred))
cm = confusion_matrix(y_test,dt_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""Hypertuning Decision Tree using Grid Search"""

#GS
from sklearn.model_selection import GridSearchCV
parameters = {'criterion': ['entropy','gini'], 
            'min_samples_split' : range(10,500,20),
            'max_depth': range(1,20,2)}

# parameters = {'max_leaf_nodes': list(range(2, 100)),
#           'min_samples_split': [2, 3, 4]}
dt = DecisionTreeClassifier()  
model = GridSearchCV(dt, parameters, cv=5, scoring = 'accuracy', verbose=50, n_jobs=-1)
model.fit(X_train,y_train)
model.best_params_

"""Decision Tree after Hyper-parameter tuning"""

gsdt_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
DT_GS_accuracy = accuracy_score(y_test,gsdt_pred)
print("Accuracy: ",accuracy_score(y_test,gsdt_pred))
print(classification_report(y_test,gsdt_pred))
cm = confusion_matrix(y_test,gsdt_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""# RFC

Random Forest without hyper-parameter tuning
"""

from sklearn.ensemble import RandomForestClassifier
rfnew = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=8, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=500,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rfnew.fit(X_train,y_train)

rfnew_pred = rfnew.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print("Accuracy: ",accuracy_score(y_test,rfnew_pred))
print(classification_report(y_test,rfnew_pred))
cm = confusion_matrix(y_test,rfnew_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""Hyper-tuning Random Forest using Grid Search"""

#With Grid Search
'''
from sklearn.model_selection import GridSearchCV
parameters = { 
    'n_estimators': [200,500],
    'max_depth' : [8,20,50],
    'min_samples_leaf':[1,2,4],
    'min_samples_split':[2,10],
    'criterion' :['gini', 'entropy']
}


rf = RandomForestClassifier(max_features='auto')
model = GridSearchCV(rf, parameters, cv=5, scoring = 'accuracy', verbose=5, n_jobs=-1)
model.fit(X_train,y_train)

model.best_params_



"""Random Forest after Hyper-parameter Tuning"""

rf_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
RF_GS_accuracy = accuracy_score(y_test,rf_pred)
print("Accuracy: ",accuracy_score(y_test,rf_pred))
print(classification_report(y_test,rf_pred))
cm = confusion_matrix(y_test,rf_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])
'''
from sklearn.ensemble import RandomForestClassifier
rfnew = RandomForestClassifier(
  max_depth=8,
 min_samples_leaf= 4,
 min_samples_split=10,
 n_estimators=500,
 criterion='entropy'
)
rfnew.fit(X_train,y_train)
rf_pred = rfnew.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
RF_GS_accuracy = accuracy_score(y_test,rf_pred)
print("Accuracy: ",accuracy_score(y_test,rf_pred))
print(classification_report(y_test,rf_pred))
cm = confusion_matrix(y_test,rf_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""# MLPClassifier

MLP Classifier without Hyper-parameter Tuning
"""

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='sgd')
mlp.fit(X_train,y_train)

mlp_pred = mlp.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print("Accuracy: ",accuracy_score(y_test,mlp_pred))
MLP_accuracy = accuracy_score(y_test,mlp_pred)
print(classification_report(y_test,mlp_pred))
cm = confusion_matrix(y_test,mlp_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])

"""Hypertuning MLP with Grid Search"""

#GS
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
mlp_gs = MLPClassifier(max_iter=1000)
parameters = {
    'hidden_layer_sizes': [(100,50,100),(100,)],
    'activation': ['tanh', 'relu','softmax'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant','adaptive'],
}

model = GridSearchCV(mlp_gs, parameters, n_jobs=-1, cv=5, verbose=5,scoring='accuracy')
model.fit(X_train, y_train)

model.best_params_

"""MLP Classifier after Grid Search"""

mlpgs_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print("Accuracy: ",accuracy_score(y_test,mlpgs_pred))
MLP_GS_accuracy = accuracy_score(y_test,mlpgs_pred)
print(classification_report(y_test,mlpgs_pred))
cm = confusion_matrix(y_test,mlpgs_pred)
print("Confusion Matrix:\n")
sns.heatmap(cm, annot = True, cmap="Blues",cbar=False,xticklabels=['Away','Draw','Home'],yticklabels=['Away','Draw','Home'])


print("FINAL ACCURACY SCORES FOR EACH MODEL") 

models_sfm = pd.DataFrame({
    'Model'       : ['Logistic Regression','GaussianNB','Support Vector Machine','Decision Tree','Random Forest','MLPClassifier'],
    'Accuracy'    : [LR_GS_accuracy, NB_GS_accuracy, SVM_GS_accuracy, DT_GS_accuracy, RF_GS_accuracy,MLP_GS_accuracy],
    
    }, columns = ['Model', 'Accuracy'])

print(models_sfm.sort_values(by='Accuracy', ascending=False))

