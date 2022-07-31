#!/usr/bin/env python
# coding: utf-8

# # 0. Load Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


np.set_printoptions(suppress=True)


# In[3]:


covid = pd.read_excel("./covid_kaggle.xlsx")


# In[4]:


covid.shape


# # 1. Data Wash

# Remove test results for other viruses, we'd like to predict the SARS-Cov-2 test result.

# In[5]:


covid = covid.drop(['Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Parainfluenza 1', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Influenza B, rapid test', 'Influenza A, rapid test'], axis=1)


# Remove irrelvalent Features including patients ID and patients intention to the ward level.

# In[6]:


covid = covid.drop(['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)


# Less than 100 patients among 5644 took urine tests.

# In[7]:


urine_features = ['Urine - Esterase', 'Urine - Aspect', 'Urine - pH', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Nitrite', 'Urine - Density', 'Urine - Urobilinogen', 'Urine - Protein', 'Urine - Sugar', 'Urine - Leukocytes', 'Urine - Crystals', 'Urine - Red blood cells', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']


# In[8]:


covid = covid.drop(urine_features, axis=1)


# Less than 100 patients among 5644 took aerial blood gas tests.

# In[9]:


arterial_blood_gas_features = ['Hb saturation (arterial blood gases)', 'pCO2 (arterial blood gas analysis)', 'Base excess (arterial blood gas analysis)', 'pH (arterial blood gas analysis)', 'Total CO2 (arterial blood gas analysis)', 'HCO3 (arterial blood gas analysis)', 'pO2 (arterial blood gas analysis)', 'Arteiral Fio2', 'Phosphor', 'ctO2 (arterial blood gas analysis)']


# In[10]:


covid = covid.drop(arterial_blood_gas_features, axis=1)


# Reamove features containing less than 100 patients' records

# In[11]:


i = 0
for column in covid:
    if (covid[column].count() < 100):
        covid = covid.drop(column, axis=1)


# Drop constant features

# In[12]:


covid = covid.loc[:,covid.apply(pd.Series.nunique) != 1]


# Drop features has least patients' records

# In[13]:


features = list(covid.columns)
sorted_features = [x for _,x in sorted(zip(covid[features].count(), features))]


# In[14]:


# [ [covid.columns.get_loc(c), c] for c in sorted_features if c in covid]


# Now all features contain at least 100 patients' record

# In[15]:


covid_init = covid[sorted_features[-1]]
    


# In[16]:


removed_features = ['Lactic Dehydrogenase', 'Creatine phosphokinase\xa0(CPK)\xa0', 'International normalized ratio (INR)', 'Base excess (venous blood gas analysis)', 'HCO3 (venous blood gas analysis)', 'Hb saturation (venous blood gas analysis)', 'Total CO2 (venous blood gas analysis)', 'pCO2 (venous blood gas analysis)', 'pH (venous blood gas analysis)', 'pO2 (venous blood gas analysis)', 'Alkaline phosphatase', 'Gamma-glutamyltransferase\xa0', 'Direct Bilirubin', 'Indirect Bilirubin', 'Total Bilirubin', 'Serum Glucose', 'Alanine transaminase', 'Aspartate transaminase', 'Strepto A', 'Sodium', 'Potassium', 'Urea', 'Creatinine']


# In[17]:


covid = covid.drop(removed_features, axis=1)


# Drop patients that have less than 10 records

# In[18]:


for index, row in covid.iterrows():
    if row.count() < 10:
        covid.drop(index, inplace=True)


# Now we have more than 500 records

# In[19]:


features = list(covid.columns)
sorted_features = [x for _,x in sorted(zip(covid[features].count(), features))]



# Drop NaN

# In[20]:


covid = covid.dropna()


# Map classification string to 0-1

# In[21]:


covid['SARS-Cov-2 exam result'] = covid['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0})


# In[22]:


covid.shape


# # 2. Train test split

# In[23]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[24]:


y = covid["SARS-Cov-2 exam result"].to_numpy()


# In[25]:


X = covid
X = X.drop(["SARS-Cov-2 exam result"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)


# In[26]:


X_train.shape


# In[27]:


X_test.shape


# In[28]:




# # 3. Feature Selection

# ### 3.1 Basic Methods

# #### 3.1.1 Drop constant and Quasi-constant features

# In[29]:


from sklearn.feature_selection import VarianceThreshold


# In[30]:


def drop_features(X_train, X_test, threshhold):
    sel = VarianceThreshold(threshold=threshhold)
    sel.fit(X_train)
    constant_features = [x for x in X_train.columns if x not in X_train.columns[sel.get_support()]]

    print(constant_features)
    X_train.drop(labels=constant_features, axis=1, inplace=True)
    X_test.drop(labels=constant_features, axis=1, inplace=True)


# In[31]:


drop_features(X_train, X_test, 0.01)


# #### 3.1.2 Drop Duplicated Features

# In[32]:


covid_t = covid.T


# ### 3.2 Correlations

# In[33]:


corrmat = X_train.corr()
corrmat = corrmat.abs().unstack()
corrmat = corrmat.sort_values(ascending=False)
corrmat = corrmat[corrmat >= 0.8]
corrmat = corrmat[corrmat < 1]
corrmat = pd.DataFrame(corrmat).reset_index()
corrmat.columns = ['feature1', 'feature2', 'corr']
corrmat


# In[34]:


# find groups of correlated features

grouped_feature_ls = []
correlated_groups = []

for feature in corrmat.feature1.unique():
    if feature not in grouped_feature_ls:

        # find all features correlated to a single feature
        correlated_block = corrmat[corrmat.feature1 == feature]
        grouped_feature_ls = grouped_feature_ls + list(
            correlated_block.feature2.unique()) + [feature]

        # append the block of features to the list
        correlated_groups.append(correlated_block)



# In[35]:


# now we can visualise each group. We see that some groups contain
# only 2 correlated features, some other groups present several features 
# that are correlated among themselves.



# In[36]:


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j] >= threshold):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[37]:


corr_features = list((correlation(X_train, 0.8)))


# In[38]:


X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)


# In[39]:


X_train.shape, X_test.shape


# ### 3.3 Statistical Methods

# #### 3.3.1 Mutual Information

# In[40]:


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile


# In[41]:


mi = mutual_info_classif(X_train, y_train)
mi = pd.Series(mi)
mi.index = X_train.columns


# In[42]:


mi.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[43]:


sel_ = SelectKBest(mutual_info_classif, k = 10).fit(X_train, y_train)


# In[44]:


mi_features = list(X_train.columns[ ~ sel_.get_support()].values)


# In[45]:


mi_features


# In[46]:


X_train.drop(labels=mi_features, axis=1, inplace=True)
X_test.drop(labels=mi_features, axis=1, inplace=True)


# In[47]:


X_train.shape


# In[48]:


X_test.shape


# # 3. Classifier

# In[49]:


import sklearn
import sklearn.ensemble
import sklearn.metrics
import xgboost as xgb


# In[50]:


from sklearn.model_selection import cross_val_score


# In[51]:


def cv_score(classifier, X, y, scoring):
    return cross_val_score(classifier, X, y, cv=5, scoring=scoring)


# ## 3.1 Decision Tree

# In[52]:


dt = sklearn.tree.DecisionTreeClassifier()

dt_f1 = cv_score(dt, X_train, y_train, 'f1')

dt.fit(X_train, y_train)


# In[53]:




# In[54]:


dt_pred = dt.predict(X_test)


dt_accuracy= sklearn.metrics.accuracy_score(y_test, dt_pred)



# In[55]:





# ## 3.2 Random Forests

# In[56]:


from sklearn.ensemble import RandomForestClassifier


# In[57]:


rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)

rf_f1 = cv_score(rf, X_train, y_train, 'f1')

rf.fit(X_train, y_train)


# In[58]:




# In[59]:


rf_pred = rf.predict(X_test)

rf_accuracy= sklearn.metrics.accuracy_score(y_test, rf_pred)


# In[60]:





# ## 3.3 XGBoost

# In[61]:


# Create a model
# Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
xgbc = xgb.XGBClassifier(
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.7,
    colsample_bylevel = 0.7,
    scale_pos_weight = 9,
    min_child_weight = 0,
    reg_alpha = 4,
    objective = 'binary:logistic',
    use_label_encoder=False,
    eval_metric='error'
)

xgbc_f1 = cv_score(xgbc, np.array(X_train), np.array(y_train), 'f1')

# Fit the models
xgbc.fit(np.array(X_train), np.array(y_train))


# In[62]:


xgbc_pred = xgbc.predict(np.array(X_test))

xgbc_accuracy = sklearn.metrics.accuracy_score(y_test, xgbc_pred)

# In[63]:




# In[64]:



# ## 3.4 Neural Networks

# In[65]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[66]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[67]:


def build_classifier() :
    nn = Sequential()
    nn.add(Dense(activation='relu', input_dim=X_train.shape[1], units=10))
    nn.add(Dropout(rate = 0.1))
    nn.add(Dense(kernel_initializer="uniform", activation='relu', units=15))
    nn.add(Dropout(rate = 0.1))
    nn.add(Dense(kernel_initializer="uniform", activation='relu', units=5))
    nn.add(Dropout(rate = 0.1))
    nn.add(Dense(kernel_initializer='uniform',activation='sigmoid', units=1))
    nn.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return nn


# In[68]:


nn = KerasClassifier(build_fn=build_classifier, epochs=200, batch_size=50, verbose=0)
# nn = build_classifier();


# In[69]:


history = nn.fit(X_train, y_train, batch_size=50, epochs=200, validation_split = 0.2) #verbose = 2


# In[70]:


import eli5
from eli5.sklearn import PermutationImportance


# In[71]:


nn_results = PermutationImportance(nn, cv="prefit").fit(X_train, y_train)
nn_importance = nn_results.feature_importances_


# In[72]:




# In[73]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'],loc='upper left')
plt.show()


# In[74]:


nn_f1 = cv_score(nn, X_train, y_train, 'f1')


# In[75]:


print(np.mean(nn_f1))


# In[76]:


nn_pred = nn.predict(X_test)
nn_pred[nn_pred > 0.5] = 1
nn_pred[nn_pred <= 0.5] = 0

nn_accuracy = sklearn.metrics.accuracy_score(y_test, nn_pred)


# In[77]:




# # 4. Prediction

# In[78]:


import math


# In[79]:

print("------ACCURACY (in%)------")
print("Decision Tree = ",dt_accuracy )

print()

print("Random Forest = ", rf_accuracy)

print()

print("XGBoost = ", xgbc_accuracy)

print()
print("Neral Network = ", nn_accuracy)

