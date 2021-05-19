#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns


# In[78]:


df = pd.read_csv('cses4_cut.csv')


# In[79]:


df.head() #Unnecesary column detected 'Unnamed: 0'. Dropping it.


# In[80]:


df = df.drop(columns = ['Unnamed: 0'])


# In[81]:


# D2029 , D2030 looks like the columns contain the same values. If so, we could drop them because they don't add variablity to our model.
df


# In[82]:


#The code below shows the unique values in the representing the features, 
#with using these we will be able to discover features and detect anomalies.

for column in df:
    print('Unique values in the', column, 'feature')
    print(df[column].value_counts())
    print()


# In[83]:


#After looking at the unique values, I have determined which features to encode in onehot format(onehot_list) and keep others in label encoding.
#Also, I have found some illogical values, for household features(2021, 2022, 2023) there are values like 97,98. I will change them to NA values.

df['D2021'] = df['D2021'].apply(lambda x: np.nan if x >20 else x)
df['D2022'] = df['D2022'].apply(lambda x: np.nan if x >18 else x)
df['D2023'] = df['D2023'].apply(lambda x: np.nan if x >19 else x)

#I have found some features looking not 'very usable'. However, we will implement all the features to ML and my strategy for ML: 
#    1. Keep all fetures at first. Encode them and use them in ML.
#    2. Find how they contribute, remove them if they don't contribute to ML. Or mix them in PCA. 
#    3. Implement different models, try to create best model.

one_hot_list = ['D2002', 'D2003', 'D2004', 'D2010', 'D2011', 'D2012', 'D2013', 'D2014', 'D2015', 
               'D2016', 'D2017', 'D2018', 'D2019', 'D2025', 'D2026', 'D2027', 'D2028', 
               'D2029', 'D2030', 'D2031']

#Encoding strategy:
#I have kept membership values as label encoding because they represent a value not category.
#Household income higher is good, lower is worse. That's why I kept it label encoded.
#D2021, D2022, D2023 represents number in household so they are not categorical features. I will keep them as they are.
#D2024, religous servicees attendence is also a number. So label encoding.


# In[84]:


#As we transformed some values to Null, I will convert them median value 
#because Null values may create problem with standardizing and ML algorithms.
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
df[['D2021', 'D2022', 'D2023']]= imp_median.fit_transform(df[['D2021', 'D2022', 'D2023']]) 


# In[85]:


#Change voted feature. True to 1 and False to 0. 
df['voted'] = df['voted'].apply(lambda x: 1 if x is True else 0)


# In[86]:


df.info() 


# In[88]:


new_df = pd.get_dummies(df, prefix=one_hot_list, columns=one_hot_list)


# In[89]:


df


# In[90]:


new_df


# In[91]:


y = new_df['voted']
X= new_df.drop(columns = ['voted'])


# In[93]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[94]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train1=sc.fit_transform(X_train)
X_train1[:]
X_test1 = sc.fit_transform(X_test)
X_test1[:]


# In[95]:


norm = StandardScaler().fit(X)

# transform training data
X_train_norm = norm.transform(X_train)
print("Scaled Train Data: \n\n")
print(X_train_norm)


# ## ML Implementation

# In[97]:


#My strategy with ML:
#    1.Apply ML without application first. Get the base result for comprasion
#    2.After every action (Standardization, PCA, ML tuning etc.), get the score to compare if the application is worthy.
#    3.Evaluate all results in the end. Find the best implementation and use it.

#ML algortihms: 
#    1.Random Forest Classifier, Logistic Regression, Gradient Boosting Classifier

#Metrics
#I will use AUC instead of Accuracy or other metrics. It is easy to compare and hence voted feature is imbalanced. 
#It is much better to use AUC.


# In[213]:


#Import necessary libraries and metrics
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[214]:


df['voted'].value_counts()


# In[240]:


def random_forest(X, y, ):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(X_train,y_train)
    predictions = rfc.predict(X_test)
    
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % accuracy_score(y_test, predictions))
    print ("AUC Score : %f" % roc_auc_score(y_test, predictions))
    
    #Print Feature Importance:
    #As there are lots of features its important to get only some of them. You can change feat_imp[:20]
    def plot_feature_importance(importance,names,model_type, top_x_features):

        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)
        
        feature_names = feature_names[:top_x_features]
        print(feature_names)
        feature_importance = feature_importance[:top_x_features]
        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        #Add chart labels
        plt.title(model_type + 'FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
    plot_feature_importance(rfc.feature_importances_, X.columns, 'Random Forest ', 20)


# In[241]:


def gbc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    gbc_ = GradientBoostingClassifier()
    gbc_.fit(X_train,y_train)
    predictions = gbc_.predict(X_test)
    
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % accuracy_score(y_test, predictions))
    print ("AUC Score : %f" % roc_auc_score(y_test, predictions))
    
    #Print Feature Importance:
    #As there are lots of features its important to get only some of them. You can change feat_imp[:20]
    def plot_feature_importance(importance,names,model_type, top_x_features):

        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)
        
        feature_names = feature_names[:top_x_features]
        print(feature_names)
        feature_importance = feature_importance[:top_x_features]
        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(10,8))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        #Add chart labels
        plt.title(model_type + 'FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
#     printFeatureImportance()
    plot_feature_importance(gbc_.feature_importances_, X.columns, 'Gradient Boosting Classifier ', 20)
    


# In[242]:


def log_reg(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    log_reg_ = LogisticRegression(max_iter = 100000)
    log_reg_.fit(X_train,y_train)
    predictions = log_reg_.predict(X_test)
    
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % accuracy_score(y_test, predictions))
    print ("AUC Score : %f" % roc_auc_score(y_test, predictions))
    


# In[243]:


def SVC_(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    svm = SVC()
    svm.fit(X_train,y_train)
    predictions = svm.predict(X_test)
    print(roc_auc_score(y_test, predictions))
    
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % accuracy_score(y_test, predictions))
    print ("AUC Score : %f" % roc_auc_score(y_test, predictions))


# ## Base Implementation 

# In[244]:


SVC_(X,y)


# In[245]:


random_forest(X,y)


# In[246]:


random_forest(X[['D2005', 'D2006', 'D2007', 'D2008', 'D2009', 'D2020', 'D2021', 'D2022', 'D2023',
 'D2024', 'age', 'D2002_1', 'D2002_2' ,'D2003_1', 'D2003_2', 'D2003_3', 'D2003_4',
 'D2003_5', 'D2003_6', 'D2003_7']],y)


# In[247]:


random_forest(X[['D2005', 'D2006',    'D2020', 'D2021', 'D2022', 'D2023',
 'D2024', 'age', 'D2002_1', 'D2002_2' ,   'D2003_4',
  'D2003_7']],y)


# In[248]:


gbc(X,y)


# In[249]:


gbc(X[['D2007', 'D2008', 'D2009', 'D2020', 
 'D2024', 'age', 'D2003_7']], y)


# In[250]:


log_reg(X,y)


# ## Standardizing

# In[251]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_scaled = X.copy()
col_names = X.columns
features = X[col_names]

df_scaled[col_names] = scaler.fit_transform(features.values)


# In[252]:


SVC_(df_scaled,y)


# In[253]:


random_forest(df_scaled,y)


# In[254]:


random_forest(df_scaled[['D2005', 'D2006',    'D2020', 'D2021', 'D2022', 'D2023',
 'D2024', 'age', 'D2002_1', 'D2002_2' ,   'D2003_4',
  'D2003_7']],y)


# In[255]:


gbc(df_scaled,y)


# In[256]:


gbc(df_scaled[['D2007', 'D2008', 'D2009', 'D2020', 
 'D2024', 'age', 'D2003_7']], y)


# In[257]:


log_reg(df_scaled,y)


# ## Base Model with KBest features

# In[258]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=15)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(15,'Score'))  #print 10 best features


# In[259]:


new_columns= np.array(featureScores.nlargest(15,'Score')['Specs'])


# In[260]:


SVC_(X[new_columns],y)


# In[261]:


SVC_(df_scaled[new_columns],y)


# In[262]:


random_forest(X[new_columns],y)


# In[263]:


gbc(X[new_columns],y)


# In[358]:


log_reg(df_scaled[new_columns],y)


# In[359]:


log_reg(X[new_columns],y)


# ## PCA

# In[270]:


from sklearn.decomposition import PCA


# In[272]:


pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[280]:


#In principal component analysis, one quantifies this relationship by finding a list of the principal axes in the data, and using those axes to describe the dataset

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(df_scaled)
print(pca.components_)
print(pca.explained_variance_)


# In[287]:


#Not good discrimination is available.
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c = y, cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[288]:


# I will pass PCA and don't implement it because the algorithms won't perform better 
# as it can be seen in the figure abovethere is not nice seperation between voted class 1 and 0.


# ## Tuning for RandomForest and Gradient Boosting

# In[266]:


## I will do parameter tuning for RandomForest
## RF performed better than all other algortihms.


# In[336]:


from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

# Number of trees in random forest, I choose low because otherwise takes lot of time.
n_estimators = [200]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
pprint(random_grid)


# In[335]:


#With these features, it performed best. That's why we will implement grid search on this data.
new_X = X[['D2005', 'D2006',    'D2020', 'D2021', 'D2022', 'D2023',
 'D2024', 'age', 'D2002_1', 'D2002_2' ,   'D2003_4',
  'D2003_7']]
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.2, random_state = 0)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring = 'roc_auc')
# Fit the random search model

rf_random.fit(X_train, y_train)


# In[338]:


rf_random.best_params_


# In[339]:


rfc = RandomForestClassifier(n_estimators= 1000,
 min_samples_split= 10,
 min_samples_leaf= 2,
 max_depth= 10)

rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
        
#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % accuracy_score(y_test, predictions))
print ("AUC Score : %f" % roc_auc_score(y_test, predictions))


# In[340]:


#Optimizing after RandomSearch
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [90, 100, 110,120],
    'max_features': ['auto'],
    'min_samples_leaf': [3,4,5,6],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [300]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring = 'roc_auc')


# In[341]:


grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[355]:


#Optimizing again.
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [120,130, 140,160],
    'max_features': ['auto'],
    'min_samples_leaf': [8,9,10],
    'min_samples_split': [10],
    'n_estimators': [300]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[356]:


grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[357]:


rfc = RandomForestClassifier(n_estimators= 1000,
 min_samples_split= 10,
 min_samples_leaf= 10,
 max_depth= 120)

rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
        
#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % accuracy_score(y_test, predictions))
print ("AUC Score : %f" % roc_auc_score(y_test, predictions))


# In[ ]:


#I tried to optimize it but base model performed better in terms of auc.
#In terms of accuracy there is a slight improvement after optimizing, it has increased to 0.8752 from 0.8655.
#Depending on business objective , we could choose the best option

