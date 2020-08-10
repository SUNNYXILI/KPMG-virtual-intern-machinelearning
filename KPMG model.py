#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

df=pd.read_excel('/Users/lixi/Desktop/KPMG_module2.xlsx', sheet_name='Transactions')


# In[11]:


df=df[['gender','past_3_years_bike_related_purchases','Age','job_industry_category',
       'wealth_segment','owns_car','tenure','type']]
df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
y=df['type']
x=df[['gender','past_3_years_bike_related_purchases','Age','job_industry_category',
       'wealth_segment','owns_car','tenure']]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0) # 70% training and 30% test
le = LabelEncoder()
le.fit(X_train['gender'].astype(str))
X_train['gender'] = le.transform(X_train['gender'].astype(str))
X_test['gender'] = le.transform(X_test['gender'].astype(str))

le.fit(X_train['owns_car'].astype(str))
X_train['owns_car'] = le.transform(X_train['owns_car'].astype(str))
X_test['owns_car'] = le.transform(X_test['owns_car'].astype(str))


le.fit(X_train['job_industry_category'].astype(str))
X_train['job_industry_category'] = le.transform(X_train['job_industry_category'].astype(str))
X_test['job_industry_category'] = le.transform(X_test['job_industry_category'].astype(str))

le.fit(X_train['wealth_segment'].astype(str))
X_train['wealth_segment'] = le.transform(X_train['wealth_segment'].astype(str))
X_test['wealth_segment'] = le.transform(X_test['wealth_segment'].astype(str))
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[12]:


y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[117]:





# In[118]:





# In[13]:


from sklearn.ensemble import RandomForestRegressor
my_model = RandomForestRegressor()

#display(graphviz.Source(export_graphviz(clf)))
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image,display
import pydotplus


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(graph)
graph.write_png('diabetes.png')
Image(graph.create_png())


# In[ ]:





# In[ ]:




