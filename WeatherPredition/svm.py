#!/usr/bin/env python
# coding: utf-8

# hw1：svm

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#讀檔
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_label = train_data['Attribute17']


# In[3]:


print(test_data.shape)
print(train_data.shape)


# 資料預處理

# In[4]:


df = train_data
tf = test_data


# In[5]:


#刪除日期
train_data.drop('Attribute1', axis = 1, inplace = True)
test_data.drop('Attribute1', axis = 1, inplace = True)


# In[6]:


train_data.columns


# In[7]:


#找尋遺失資料
train_data.isnull().sum()


# In[8]:


#train_label
for i in range(len(train_label)):
    if train_label[i] == 'Yes':
        train_label[i] = 1
    else:
        train_label[i] = 0
        
train_label = train_label.astype(int)

print(type(train_label))
train_label


# In[9]:


#非數字類別資料轉換
categorical = [var for var in df.columns if df[var].dtype == '0']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)

from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

#將NaN以眾數mode取代
df['Attribute8'].fillna(df['Attribute8'].mode()[0], inplace = True)
df['Attribute10'].fillna(df['Attribute10'].mode()[0], inplace = True)
df['Attribute16'].fillna(df['Attribute16'].mode()[0], inplace = True)
df['Attribute17'].fillna(df['Attribute17'].mode()[0], inplace = True)
tf['Attribute8'].fillna(tf['Attribute8'].mode()[0], inplace = True)
tf['Attribute10'].fillna(tf['Attribute10'].mode()[0], inplace = True)
tf['Attribute16'].fillna(tf['Attribute16'].mode()[0], inplace = True)

#非數字類別以數字替代
df['Attribute8']= label_encoder.fit_transform(df['Attribute8'])
df['Attribute10']= label_encoder.fit_transform(df['Attribute10'])
df['Attribute16']= label_encoder.fit_transform(df['Attribute16'])
df['Attribute17']= label_encoder.fit_transform(df['Attribute17'])
tf['Attribute8']= label_encoder.fit_transform(tf['Attribute8'])
tf['Attribute10']= label_encoder.fit_transform(tf['Attribute10'])
tf['Attribute16']= label_encoder.fit_transform(tf['Attribute16'])
df


# In[10]:


#Attribute8, Attribute10, Attribute16
df.isna().sum()
#tf.isna().sum()
#train_label.isna().sum()


# In[11]:


#數字類別
numerical = [var for var in df.columns if df[var].dtype != '0']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

#將NaN以中位數median取代
df = df.fillna(df.median())
tf = tf.fillna(tf.median())
tf


# In[12]:


df.isnull().sum()
#tf.isnull().sum()
#df.info()

相關係數
# In[13]:


#增加溫差欄位
df['temp'] = df['Attribute3'] - df['Attribute4']
tf['temp'] = df['Attribute3'] - df['Attribute4']


# In[14]:


cor = df.corr()
plt.figure(figsize = (20,12))
sns.heatmap(cor, annot = True)


# In[15]:


related = cor['Attribute17'].sort_values(ascending = False)
related


# In[60]:


#找出相關係數為正的欄位
x = []
for i in range(len(related)):
    if related[i] > 0:
        x.append(related.index[i])
print(x)

#挑選相關係數>0欄位 + 相關係數<-0.10的欄位
x = df[['Attribute12', 'temp', 'Attribute16', 'Attribute14', 'Attribute5', 'Attribute9', 
        'Attribute11', 'Attribute3', 'Attribute8', 'Attribute10', 'Attribute2', 
        'Attribute4', 'Attribute15', 'Attribute13', 'Attribute7']]
xt = tf[['Attribute12', 'temp', 'Attribute16', 'Attribute14', 'Attribute5', 'Attribute9', 
         'Attribute11', 'Attribute3', 'Attribute8', 'Attribute10', 'Attribute2', 
         'Attribute4', 'Attribute15', 'Attribute13', 'Attribute7']]
#x.drop('Attribute17', inplace = True, axis = 1)
xt


# 平衡資料：0/1預測結果數量差距太大

# In[61]:


#計算數量
y = train_label
y.value_counts()


# In[62]:


#計算出現機率
y.value_counts()/np.float(len(df))


# In[63]:


#平衡資料
from imblearn.over_sampling import SMOTE

bal = SMOTE()
x, y = bal.fit_resample(x, y)


# In[64]:


#已平衡
y.value_counts()


# In[65]:


x.describe()


# Feature Scaling 標準化資料

# In[66]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_sta = scaler.fit_transform(x)
x_sta = pd.DataFrame(x_sta, columns=[x.columns])

xt_sta = scaler.fit_transform(xt)
xt_sta = pd.DataFrame(xt_sta, columns=[xt.columns])


# In[67]:


x_sta.describe()


# In[68]:


xt_sta.describe()


# svm 建模

# Run SVM with rbf kernel and C=1, 100, 1000

# In[69]:


from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score

# C=1, kernel=rbf, gamma=auto
svc = SVC()
svc.fit(x_sta, y)

test_label = svc.predict(xt_sta)
#test_label


# In[26]:


# C=100
svc = SVC(C = 100.0)
svc.fit(x_sta, y)

test_label_c100 = svc.predict(xt_sta)
#test_label_c100


# In[27]:


# C=10，準確度下降
svc = SVC(C = 10.0)
svc.fit(x_sta, y)

test_label_c10 = svc.predict(xt_sta)
#test_label_c10


# Run SVM with linear kernel 

# In[28]:


# kernel=linear, C=1
linear_svc = SVC(kernel = 'linear', C = 1.0) 

linear_svc.fit(x_sta, y)

test_label_Lc1 = linear_svc.predict(xt_sta)
#test_label_Lc1


# In[70]:


# C=10，輸出資料
linear_svc = SVC(kernel = 'linear', C = 10.0) 

linear_svc.fit(x_sta, y)

test_label_Lc10 = linear_svc.predict(xt_sta)
test_label_Lc10


# Run SVM with polynomial kernel and C=1.0

# In[29]:


#準確度下降
poly_svc = SVC(kernel = 'poly', C = 100.0) 

poly_svc.fit(x_sta, y)

test_label_Pc100 = poly_svc.predict(xt_sta)
#test_label_Pc100


# In[71]:


#整理輸出資料
output = pd.DataFrame(data = test_label_Lc10)
output = output.reset_index()
output['id'] = output['index']
output['ans'] = output[0]
output['id'] = output['id'].astype(float)
del output['index']
del output[0]
output


# In[72]:


#輸出資料
import csv
with open('ex_submit.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile) #建立csv檔寫入器
    writer.writerow(['id', 'ans']) #建立標題
    #轉換格式
    for i in range(len(output)):
        writer.writerow([float(output['id'][i]), int(output['ans'][i])])


# In[ ]:




