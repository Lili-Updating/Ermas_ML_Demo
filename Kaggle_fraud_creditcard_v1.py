#!/usr/bin/env python
# coding: utf-8

# In[26]:


# load packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


# import data
creditcard = pd.read_csv("kaggle_fraud_creditcard.csv")


# In[28]:


# preprocess
creditcard['Amount'] = StandardScaler().fit_transform(creditcard['Amount'].values.reshape(-1,1)) 
creditcard['Time'] = StandardScaler().fit_transform(creditcard['Time'].values.reshape(-1,1)) 


# In[29]:


# store features and target seperately
X = creditcard.drop('Class', axis=1)
y = creditcard['Class']


# In[30]:


# split training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[31]:


# train a logistic regression classifier 
classifer = LogisticRegression(solver="lbfgs").fit(X_train, y_train)


# In[32]:


# predict probabilities on the test data
prediction_proba = classifer.predict_proba(X_test)


# In[33]:


# compute roc curve
fpr, tpr, threshold = roc_curve(y_test, prediction_proba[:, 1])
roc_auc = auc(fpr, tpr)


# In[34]:


# draw roc curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[35]:


# find the optimal probability cutoff, which is the intersection point of sensitivity plot and specificity plot
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
cutoff = roc_t["threshold"].values[0]
print("optimal probability cutoff: %.6f" % cutoff)


# In[36]:


# perform predictions based on the probability cutoff
prediction = np.where(prediction_proba[:, 1]>=cutoff, 1, 0)


# In[37]:


# compute performance measures
# confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
# type I error
type_I_error = float(fp) / (fp+tn)
# type II error
type_II_error = float(fn) / (fn+tp)
# accuracy
accuracy = accuracy_score(y_test, prediction)
# f1 score
f1 = f1_score(y_test, prediction)


# In[38]:


print("Type I Error: %.2f%%" % (type_I_error*100))
print("Type II Error: %.2f%%" % (type_II_error*100))
print("Accuracy: %.2f%%" % (accuracy*100))
print("F1 Score: %.6f" % f1)

