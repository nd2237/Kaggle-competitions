#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pyplot import plot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#讀檔
tData = pd.read_csv('train_hw2.csv') #train
sData = pd.read_csv('test_3000.csv') #test
data = pd.read_csv('dataset.csv') #分類標準


# In[3]:


#轉換Feature 13
pitch = tData['Feature 13']
for i in range (len(pitch)):
    if pitch[i] == 'C':
        pitch[i] = 0
    elif pitch[i] == 'C#':
        pitch[i] = 1
    elif pitch[i] == 'D':
        pitch[i] = 2
    elif pitch[i] == 'D#':
        pitch[i] = 3
    elif pitch[i] == 'E':
        pitch[i] = 4
    elif pitch[i] == 'F':
        pitch[i] = 5
    elif pitch[i] == 'F#':
        pitch[i] = 6
    elif pitch[i] == 'G':
        pitch[i] = 7
    elif pitch[i] == 'G#':
        pitch[i] = 8
    elif pitch[i] == 'A':
        pitch[i] = 9
    elif pitch[i] == 'A#':
        pitch[i] = 10
    elif pitch[i] == 'B':
        pitch[i] = 11
    else:
        pitch[i] = -1
print(pitch)


# In[4]:


#object -> int
tData['Feature 13'] = pitch.astype(int)


# In[5]:


#tData.isnull().sum()
#tData.isna().sum()
print(tData.dtypes)
#sData.isnull().sum()
print(sData.dtypes)


# In[883]:


#Scaling data
x = tData.drop('song_id', axis = 1)
x = x[['Feature 4', 'Feature 12']]

scaler = StandardScaler()
xSta = scaler.fit_transform(x)
xSta = pd.DataFrame(xSta, columns = [x.columns])
xSta


# In[884]:


#pca
pca = PCA() 
pca.fit(xSta) 
evr = pca.explained_variance_ratio_ 
evr.cumsum()


# In[885]:


from kneed import KneeLocator

wcss = []
max_clusters = 13
for i in range(1, 13):
    kmeans = KMeans(i, init='k-means++', random_state = 42)
    kmeans.fit(xSta)
    wcss.append(kmeans.inertia_)
      
# programmatically locate the elbow
n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
print("Optimal number of clusters", n_clusters)


# In[887]:


#Kmeans
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state = 42, max_iter = 500000)
kmeans.fit(xSta)
x['Cluster'] = kmeans.labels_
label = x['Cluster']


# In[888]:


label.value_counts()


# In[889]:


#比對資料
ans = []

for i in range(len(sData)):
    if label[sData['col_1'][i]] == label[sData['col_2'][i]]:
        ans.append(1)
    else:
        ans.append(0)


# In[890]:


#整理輸出資料
output = pd.DataFrame(data = ans)
output = output.reset_index()
output['id'] = output['index']
output['ans'] = output[0]
#output['id'] = output['id'].astype(int)
del output['index']
del output[0]
output


# In[891]:


#輸出資料
import csv
with open('submit.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile) #建立csv檔寫入器
    writer.writerow(['id', 'ans']) #建立標題
    #轉換格式
    for i in range(len(output)):
        writer.writerow([int(output['id'][i]), int(output['ans'][i])])


# In[ ]:





# In[ ]:




