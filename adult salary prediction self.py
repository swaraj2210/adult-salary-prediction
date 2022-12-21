#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib as plt
import seaborn as sns
import pandas_profiling as pp
import numpy as np


# 1.import libraries
# 2.import file
# 3.correct column name.
# 4.check shape
# 5.check data summary using info & describe.

# In[3]:


df= pd.read_csv("C:\\Users\\Swaraj\\OneDrive - Motherson Group\\Desktop\\SWARAJ\\Personal\\1.Study\\Data science\\tensorflow\\Machine Learning\\Machine Learning R_27.07.21\\Machine Learning Project 1 - Adult Salary Prediction\\adult_data.csv")
df.columns = ['age','workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
             'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary']


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# In[11]:


df.head()


# 1.from describe and isna we find there is no null values.
# 2.as we can see there is ouliers in capital gain. capital_loss & hours_per_week (from percentile row)

# In[19]:


sns.distplot(df[df['sex']=='Male']['age'],bins=None, hist=False, label = 'Male')
#sns.distplot(df[df.sex=='Female'],bins=None, hist=False, label = 'Female')


# In[8]:


sns.displot(y=df["marital_status"])


# In[9]:


sns.displot(y=df["occupation"])


# In[10]:


sns.displot(y=df["relationship"])


# In[11]:


sns.displot(y=df["race"])


# In[12]:


sns.displot(y=df["sex"])


# In[13]:


sns.displot(df['age'])


# In[14]:


sns.boxplot(df['capital_gain'])


# In[15]:


def handle_capital_gain(df):
    df['capital_gain'] = np.where(df['capital_gain'] == 0, np.nan, df['capital_gain'])
    df['capital_gain'] = np.log(df['capital_gain'])
    df['capital_gain'] = df['capital_gain'].replace(np.nan, 0)


# In[16]:


handle_capital_gain(df)


# In[17]:


def capital_loss_log(df):
    df['capital_loss'] = np.where(df['capital_loss'] == 0, np.nan, df['capital_loss'])
    df['capital_loss'] = np.log(df['capital_loss'])
    df['capital_loss'] = df['capital_loss'].replace(np.nan, 0)


# In[18]:


capital_loss_log(df)


# In[19]:


def remove_outlier_capital_loss(df):
    IQR = df['capital_loss'].quantile(0.75) - df['capital_loss'].quantile(0.25)
    
    lower_range = df['capital_loss'].quantile(0.25) - (1.5 * IQR)
    upper_range = df['capital_loss'].quantile(0.75) + (1.5 * IQR)
    
    df.loc[df['capital_loss'] <= lower_range, 'capital_loss'] = lower_range
    df.loc[df['capital_loss'] >= upper_range, 'capital_loss'] = upper_range


# In[20]:


remove_outlier_capital_loss(df)


# In[21]:


def remove_outlier_hours_per_week(df):
    IQR = df['hours_per_week'].quantile(0.75) - df['hours_per_week'].quantile(0.25)
    
    lower_range = df['hours_per_week'].quantile(0.25) - (1.5 * IQR)
    upper_range = df['hours_per_week'].quantile(0.75) + (1.5 * IQR)
    
    df.loc[df['hours_per_week'] <= lower_range, 'hours_per_week'] = lower_range
    df.loc[df['hours_per_week'] >= upper_range, 'hours_per_week'] = upper_range


# In[22]:


remove_outlier_hours_per_week(df)


# 1.all outliers are removed.
# 2.to use ML model we need to encode the string into int values.

# In[23]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
def feature_engg(df):
# Encode labels in column 'species'.
  df['race']= label_encoder.fit_transform(df['race'])
  df['sex']= label_encoder.fit_transform(df['sex'])
  df['workclass']= label_encoder.fit_transform(df['workclass'])
  df['education']= label_encoder.fit_transform(df['education'])
  df['marital_status']= label_encoder.fit_transform(df['marital_status'])
  df['occupation']= label_encoder.fit_transform(df['occupation'])
  df['relationship']= label_encoder.fit_transform(df['relationship'])
  df['native_country']= label_encoder.fit_transform(df['native_country'])


# In[24]:


feature_engg(df)


# In[25]:


df.drop('fnlwgt',axis=1,inplace=True)


# In[26]:


df.head()


# In[27]:


df.describe()


# In[28]:


pp.ProfileReport(df)


# In[29]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[30]:


sc = StandardScaler()


# In[31]:


X = df[['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 
          'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']]


# In[32]:


y = df['salary']


# In[33]:


y.value_counts()


# In[34]:


#X = sc.fit_transform(X)


# In[35]:


X


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


print("Train data shape: {}".format(X_train.shape))
print("Test data shape: {}".format(X_test.shape))


# In[38]:


from sklearn.linear_model import LogisticRegression


# In[39]:


lg_model = LogisticRegression()


# In[40]:


lg_model.fit(X_train, y_train)


# In[41]:


y_pred = lg_model.predict(X_test)


# In[42]:


result = {
    'Actual': y_test,
    'Predicted': y_pred
}


# In[43]:


pd.DataFrame(result)


# In[44]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[45]:


print("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
print("Classification Report:\n {}".format(classification_report(y_test, y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




