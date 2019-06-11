#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load packages
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib



get_ipython().run_line_magic('matplotlib', 'inline')
 
import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies


# In[2]:


ws=Workspace.from_config()


# In[3]:


exp = Experiment(workspace=ws, name='kagglefraud')
run = exp.start_logging()                   
run.log("Experiment start time", str(datetime.datetime.now()))


# In[4]:


# import data
creditcard = pd.read_csv("kaggle_fraud_creditcard.csv")
# preprocess
creditcard['Amount'] = StandardScaler().fit_transform(creditcard['Amount'].values.reshape(-1,1)) 
creditcard['Time'] = StandardScaler().fit_transform(creditcard['Time'].values.reshape(-1,1)) 
# store features and target seperately
X = creditcard.drop('Class', axis=1)
y = creditcard['Class']
# split training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[5]:


# train a logistic regression classifier 
classifer = LogisticRegression(solver="lbfgs").fit(X_train, y_train)


# In[6]:


# predict probabilities on the test data
prediction_proba = classifer.predict_proba(X_test)


# In[7]:


# compute roc curve
fpr, tpr, threshold = roc_curve(y_test, prediction_proba[:, 1])
roc_auc = auc(fpr, tpr)


# In[8]:


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


# In[9]:


# find the optimal probability cutoff, which is the intersection point of sensitivity plot and specificity plot
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
cutoff = roc_t["threshold"].values[0]
print("optimal probability cutoff: %.6f" % cutoff)


# In[10]:


# perform predictions based on the probability cutoff
prediction = np.where(prediction_proba[:, 1]>=cutoff, 1, 0)


# In[11]:


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


# In[12]:


print("Type I Error: %.2f%%" % (type_I_error*100))
print("Type II Error: %.2f%%" % (type_II_error*100))
print("Accuracy: %.2f%%" % (accuracy*100))
print("F1 Score: %.6f" % f1)


# In[13]:


filename = 'kaggle_fraud_model.pkl'
joblib.dump(classifer, filename)


# In[14]:


run.log("Experiment end time", str(datetime.datetime.now()))
run.complete()


# In[15]:


model = Model.register(model_path = "kaggle_fraud_model.pkl",
                       model_name = "kaggle_fraud_model",
                       description = "Kaggle Fraud Detection",
                       workspace = ws)

modelenv = CondaDependencies()
modelenv.add_conda_package("scikit-learn")
 
with open("modelenv.yml","w") as f:
    f.write(modelenv.serialize_to_string())
with open("modelenv.yml","r") as f:
    print(f.read())


# In[20]:


get_ipython().run_cell_magic('writefile', 'score.py', '\nimport json\nimport numpy as np\nimport os\nimport pickle\nfrom sklearn.externals import joblib\nfrom azureml.core.model import Model\n \ndef init():\n    global model\n    # retrieve the path to the model file using the model name\n    model_path = Model.get_model_path(\'kaggle_fraud_model\')\n    model = joblib.load(model_path)\n \ndef run(raw_data):\n    data = np.array(json.loads(raw_data)[\'data\'])\n    # make prediction\n    y_proba = model.predict_proba(data)[:, 1]\n    if y_proba > 0.000835:\n        y_pred = "Fraud"\n    else:\n        y_pred = "Not Fraud"\n    return json.dumps(y_pred)')


# In[22]:


get_ipython().run_cell_magic('time', '', ' \nimage_config = ContainerImage.image_configuration(execution_script="score.py", \n                                                  runtime="python", \n                                                  conda_file="modelenv.yml")\n\naciconfig = AciWebservice.deploy_configuration(cpu_cores=1, \n                                               memory_gb=1, \n                                               tags={"data": "kaggle_fraud",  "method" : "sklearn"}, \n                                               description=\'Kaggle Fraud Detection\')\n \nservice = Webservice.deploy_from_model(workspace=ws,\n                                       name=\'kaggle-fraud-svc4\',\n                                       deployment_config=aciconfig,\n                                       models=[model],\n                                       image_config=image_config)\n \nservice.wait_for_deployment(show_output=True)\n\n\nprint(service.scoring_uri)')


# In[58]:


fraud_example = list(np.array(X_test.iloc[355, :]))
print("X values: ", fraud_example)
print("\ntrue label (1 denoting fraud and 0 denoting nonfraud): ", y_test.iloc[355])


# In[59]:


nonfraud_example = list(np.array(X_test.iloc[0, :]))
print("X values:", nonfraud_example)
print("\ntrue label (1 denoting fraud and 0 denoting nonfraud): ", y_test.iloc[0])


# In[60]:


import requests
url = 'http://01479f65-2f69-4759-8c5d-1fa62c4cc3b3.eastus.azurecontainer.io/score'


# In[61]:


r = requests.get(url, params={'data': '['+str(fraud_example)+']'})
print(r.text)


# In[62]:


r = requests.get(url, params={'data': '['+str(nonfraud_example)+']'})
print(r.text)

