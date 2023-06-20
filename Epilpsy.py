#!/usr/bin/env python
# coding: utf-8

# ## SVM Algorithm
# # Epilpsy Disease Prediction on the basis of EEG data
# 
# 

# Import the libraries

# In[28]:


import numpy as np                                     # make arrays
import pandas as pd                                    # make data frame  
from sklearn.preprocessing import StandardScaler       # standardising the data to common range
from sklearn.model_selection import train_test_split   # split the data into train and test data set
from sklearn import svm                                # support vector machine
from sklearn.metrics import accuracy_score   
import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt


##KNN, Random forest, 
#Logistic regression, Naive Bayes, Decision tree, Random tree


# Data collection and analysis 
# 
# 

# Epilepsy data set

# In[2]:


#load the dataset into pandas dataframe
epilp_datasets= pd.read_csv("Epilepsy_data.csv")


# In[3]:


#printing the first 5 rows of the datasets
epilp_datasets.head()


# In[4]:


#number of rows and columns
epilp_datasets.shape


# In[5]:


#statistical measure of data
epilp_datasets.describe()


# In[6]:


epilp_datasets['y'].value_counts()


# 1 represent patient with epileptic seizure and all other 2,3,4,5 are with non epileptic seizure
# 

# In[7]:


epilp_datasets.groupby('y').mean()


# In[8]:


#seperating the data and labels 
z = epilp_datasets.drop(columns= 'Individual', axis=1)  #axis = 1 for column drop, and 0 for row drop.
x = z.drop(columns= 'y', axis=1) 
y= epilp_datasets['y']


# In[9]:


print(x)


# In[10]:


print(y)


# Here 1 means person is epilptic
# 
# and 2,3,4,5 means person is not epilptic

# # Data standardization
# 

# In[11]:


scaler = StandardScaler()
scaler.fit(x)


# In[12]:


standardized_data= scaler.transform(x)


# In[13]:


print(standardized_data)


# In[14]:


print(x)
print(y)


# In[15]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2,stratify= y, random_state=2)


# In[16]:


print(x.shape,x_train.shape,x_test.shape)


# In[17]:


print(y.shape, y_train.shape,y_test.shape)


# # Training the model 

# In[18]:


classifier = svm.SVC()
classifier


# In[19]:


#training the svm classifier
classifier.fit(x_train,y_train)


# Model evaluation

# In[20]:


#Accuracy score on traing data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction, y_train)


# In[21]:


print(training_data_accuracy)


# In[22]:


#accuracy score of test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction, y_test)


# In[23]:


print ("Accuracy of test data is", test_data_accuracy)


# Making a Predictive system

# In[25]:


input_data = (-167,-230,-280,-315,-338,-369,-405,-392,-298,-140,27,146,211,223,214,187,167,166,179,192,190,168,129,85,43,4,-28,-47,-43,-24,-7,12,32,43,12,-70,-181,-292,-374,-410,-382,-335,-232,-128,-6,106,233,312,423,550,695,816,839,769,661,525,383,292,267,339,451,537,564,534,444,305,160,27,-74,-147,-205,-242,-274,-304,-331,-355,-372,-380,-370,-341,-299,-257,-235,-249,-300,-381,-399,-345,-183,17,178,274,288,265,229,193,160,106,34,-51,-120,-166,-189,-207,-225,-242,-251,-255,-237,-202,-120,19,186,340,441,465,410,288,130,-16,-123,-194,-232,-255,-272,-266,-255,-209,-168,-142,-148,-169,-180,-174,-107,12,206,419,596,683,679,596,472,330,168,26,-63,-73,-37,25,61,67,53,28,-6,-44,-92,-154,-211,-257,-258,-168,-32,140,277,366,408,416,415,423,434,416,374,319,268,215,165,103)
#changing the input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one array 
input_data_reshape=  input_data_as_numpy_array.reshape(1,-1)

#standardize the input data bcz the training data is standardized
std_data = scaler.transform(input_data_reshape)
#print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction==1):
    print("The person is epileptic:")
else:
    print("Person is not epileptic:")


# In[26]:


epilp_datasets.corr()


# In[33]:


corr_matrix= epilp_datasets.corr()
fig,ax= plt.subplots(figsize=(179,90))
ax= sns.heatmap(corr_matrix,annot=True,linewidths=1.5,fmt=".2f", cmap="YlGnBu");


# In[ ]:




