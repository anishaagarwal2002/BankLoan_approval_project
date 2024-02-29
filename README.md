# BankLoan_approval_project

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

df=pd.read_csv('/Users/bhavya/Downloads/application_data.csv')

df.head()

Df.shape

Df.size

Df.dtypes

df.drop(["CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","CNT_CHILDREN","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
         "NAME_HOUSING_TYPE","REGION_POPULATION_RELATIVE","DAYS_BIRTH","DAYS_EMPLOYED","DAYS_REGISTRATION","DAYS_ID_PUBLISH","OWN_CAR_AGE","FLAG_MOBIL",
         "FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL","FLAG_EMAIL","CNT_FAM_MEMBERS","REGION_RATING_CLIENT",
         "REGION_RATING_CLIENT_W_CITY","REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","NAME_TYPE_SUITE",
         "LIVE_REGION_NOT_WORK_REGION","REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY"],axis=1,inplace=True)


df.drop(['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
'AMT_REQ_CREDIT_BUREAU_YEAR'],axis=1,inplace=True)

df.tail()

Df.columns

df.drop(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG',
       'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
       'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
       'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG','NAME_CONTRACT_TYPE',
       'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
       'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
       'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE','WEEKDAY_APPR_PROCESS_START',
       'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
       'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
       'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI','OCCUPATION_TYPE',
       'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
       'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
       'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
       'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE'],axis=1,inplace=True)

Df.columns

Df.shape

df.info()

df.describe()

def missing (df1):
    missing_number = df1.isnull().sum().sort_values(ascending=False)
    missing_percent = ((df1.isnull().sum()/df1.isnull().count())*100).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values

df.isnull().sum()

#fill null values with mean,median ,mode
for i in df.columns:
    if df[i].dtypes == 'object':
        df[i].fillna(df[i].mode()[0], inplace=True)
    else:
        df[i].fillna(df[i].median(), inplace=True)
print(df)

df.isnull().sum()

df.head()

# finding numerical data and categorical data in dataset
numerical= df.drop(['TARGET'], axis=1).select_dtypes('number').columns

categorical = df.select_dtypes('object').columns

print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')


sns.countplot(df.dtypes.map(str))
plt.show()

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['NAME_INCOME_TYPE']= label_encoder.fit_transform(df['NAME_INCOME_TYPE'])

df['NAME_INCOME_TYPE']

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df['ORGANIZATION_TYPE']= label_encoder.fit_transform(df['ORGANIZATION_TYPE'])

df['ORGANIZATION_TYPE']

for col in df.columns:
    if df[col].dtypes != object:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1
        llp = q1-1.5*IQR
        ulp = q3+1.5*IQR
        print('column name',col)
        print('mean:',df[col].mean())
        print('mode:',df[col].mode()[0])
        print('median:',df[col].median())
        print('skewness:',df[col].skew())
        print('kurtosis:',df[col].kurtosis())
        print('null_value count:',df[col].isnull().sum())
        print('\n')

#checking outliers
#Outlier Search: This helps you to get some insights about the outliers in the data.
df.plot(kind='box', layout=(3,3),subplots=1,figsize=(25,11))
plt.show()

def out_to_null(data=df):
    for x in df.columns:
        if df[x].dtypes ==np.float64:
            #In the given column the outliers will convert into null values
            print('\n\n Column name:', x)
            q1=data[x].quantile(0.25)
            print('Quatile, Q1: ',q1)
            q3=data[x].quantile(0.75)
            print('Quatile, Q3: ',q3)
            iqr=q3-q1
            print('Inter Quatile Range, IQR: ',iqr)
            ulp=q3+(1.5*iqr)
            print('Upper Limit Point, ULP: ',ulp)
            llp=q1-(1.5*iqr)
            print('Lower Limit Point, LLP: ',llp)
            #here the values which are above upperlimit and below lowerlimt are replaced with null values          
            data[x]=np.where(data[x] > ulp,np.nan,
                                   np.where(data[x] < llp,np.nan ,data[x]))
            print('\n\nin the given column the outliers are replaced with null values\n')

count=1
plt.subplots(figsize=(30,25))
for i in df.columns:
    if df[i].dtypes!='object':
        plt.subplot(6,7,count)
        sns.distplot(df[i])
        count+=1

plt.show()

cor = df.corr()
cor

ax = sns.heatmap(df.corr(), annot=True)
plt.show()

Df

import imblearn
from imblearn.under_sampling import RandomUnderSampler 

undersample = RandomUnderSampler(sampling_strategy=0.5)

cols = df.columns.tolist()
cols = [c for c in cols if c not in ["TARGET"]]
target = "TARGET"

#define X and Y
X = df[cols]
Y = df[target]

#undersample
X_under, Y_under = undersample.fit_resample(X, Y)

print(X_under.shape)

from pandas import DataFrame
test = pd.DataFrame(Y_under, columns = ['TARGET'])

#visualizing undersampling results
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="TARGET", data=df, ax=axs[0])
sns.countplot(x="TARGET", data=test, ax=axs[1])

fig.suptitle("Target repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")

X = df.drop("TARGET",axis=1)
y = df.TARGET

X.shape


from sklearn.model_selection import train_test_split
​
X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.3)

print(X_train.shape , X_test.shape , y_train.shape ,y_test.shape)

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5)

# Fit the grid search to your data
grid_search.fit(X_train, y_train)

# Get the best parameters and estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Use the best estimator for predictions
y_pred = best_estimator.predict(X_test)

# Applying random forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_Classifier = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
rf_Classifier.fit(X_train, y_train)

y_pred = rf_Classifier.predict(X_test)
y_pred

from sklearn.metrics import  confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#importing packages for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

#train the model
model1 = LogisticRegression(random_state=42)
logit = model1.fit(X_train, y_train)

#predictions
y_pred_logit = model1.predict(X_test) 

#scores
print("Accuracy Logit:",metrics.accuracy_score(y_test, y_pred_logit))
print("Precision Logit:",metrics.precision_score(y_test, y_pred_logit))
print("Recall Logit:",metrics.recall_score(y_test, y_pred_logit))
print("F1 Score Logit:",metrics.f1_score(y_test, y_pred_logit))

#print CM
matrix_logit = confusion_matrix(y_test, y_pred_logit)
cm_logit = pd.DataFrame(matrix_logit, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_logit, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix Logit"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

#AUC
y_pred_logit_proba = model1.predict_proba(X_test)[::,1]
fpr_logit, tpr_logit, _ = metrics.roc_curve(y_test,  y_pred_logit_proba)
auc_logit = metrics.roc_auc_score(y_test, y_pred_logit_proba)
print("AUC Logistic Regression :", auc_logit)

#ROC
plt.plot(fpr_logit,tpr_logit,label="Logistic Regression, auc={:.3f})".format(auc_logit))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic Regression ROC curve')
plt.legend(loc=4)
plt.show()

logit_precision, logit_recall, _ = precision_recall_curve(y_test, y_pred_logit_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(logit_recall, logit_precision, color='orange', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

#Support Vector Machine
#train the model
model2 = SVC()
svm = model2.fit(X_train, y_train)

#predictions
y_pred_svm = model2.predict(X_test)

#scores
precision_svm = precision_score(y_test, y_pred_svm, zero_division=1)

# Calculate other metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

# Print the metrics
print("Accuracy SVM:", accuracy_svm)
print("Precision SVM:", precision_svm)
print("Recall SVM:", recall_svm)
print("F1 Score SVM:", f1_svm)

#CM matrix
matrix_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm = pd.DataFrame(matrix_svm, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_svm, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix SVM"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

#AUC
svm_classifier = SVC(probability=True)
calibrated_svm = CalibratedClassifierCV(svm_classifier, method='sigmoid')
calibrated_svm.fit(X_train, y_train)
y_pred_svm_proba = calibrated_svm.predict_proba(X_test)[:,1]
#decision_values = model2.decision_function(X_test)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm_proba)
auc_svm = roc_auc_score(y_test, y_pred_svm_proba)
print("AUC SVM :", auc_svm)

#ROC
plt.plot(fpr_svm,tpr_svm,label="SVM, auc={:.3f})".format(auc_svm))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('SVM ROC curve')
plt.legend(loc=4)
plt.show()

svm_precision, svm_recall, _ = precision_recall_curve(y_test, y_pred_svm_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(svm_recall, svm_precision, color='orange', label='SVM')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

#train the model
model3 = RandomForestClassifier(random_state=2)
rf = model3.fit(X_train, y_train)

#predictions
y_pred_rf = model3.predict(X_test)

#scores
print("Accuracy RF:",metrics.accuracy_score(y_test, y_pred_rf))
print("Precision RF:",metrics.precision_score(y_test, y_pred_rf))
print("Recall RF:",metrics.recall_score(y_test, y_pred_rf))
print("F1 Score RF:",metrics.f1_score(y_test, y_pred_rf))
#CM matrix
matrix_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf = pd.DataFrame(matrix_rf, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_rf, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix RF"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

#AUC
y_pred_rf_proba = model3.predict_proba(X_test)[::,1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test,  y_pred_rf_proba)
auc_rf = metrics.roc_auc_score(y_test, y_pred_rf_proba)
print("AUC Random Forest :", auc_rf)

#ROC
plt.plot(fpr_rf,tpr_rf,label="Random Forest, auc={:.3f})".format(auc_rf))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Random Forest ROC curve')
plt.legend(loc=4)
plt.show()

rf_precision, rf_recall, _ = precision_recall_curve(y_test, y_pred_rf_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(rf_recall, rf_precision, color='orange', label='RF')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

#train the model
model4 = XGBClassifier(random_state=2)
xgb = model4.fit(X_train, y_train)

#predictions
y_pred_xgb = model4.predict(X_test) 

#scores
print("Accuracy XGB:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision XGB:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall XGB:",metrics.recall_score(y_test, y_pred_xgb))
print("F1 Score XGB:",metrics.f1_score(y_test, y_pred_xgb))

#CM matrix
matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_xgb = pd.DataFrame(matrix_xgb, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_xgb, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix XGBoost"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

#AUC
y_pred_xgb_proba = model4.predict_proba(X_test)[::,1]
fpr_xgb, tpr_xgb, _ = metrics.roc_curve(y_test,  y_pred_xgb_proba)
auc_xgb = metrics.roc_auc_score(y_test, y_pred_xgb_proba)
print("AUC XGBoost :", auc_xgb)

#ROC
plt.plot(fpr_xgb,tpr_xgb,label="XGBoost, auc={:.3f})".format(auc_xgb))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('XGBoost ROC curve')
plt.legend(loc=4)
plt.show()

xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, y_pred_xgb_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(xgb_recall, xgb_precision, color='orange', label='XGB')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()


Role of Machine Learning: Transformative integration for enhanced efficiency and accuracy.
ML Algorithms' Advantage: Analyzing extensive data, uncovering patterns beyond human capabilities.

Certainly! Here's a brief overview of each library and whether they have been used in the provided code:

1. **Scikit-learn (Sklearn):**
   - **Overview:** Scikit-learn is a machine learning library in Python. It provides simple and efficient tools for data analysis and modeling, including various machine learning algorithms.
   - **Usage in Code:** Yes, the code uses several modules from Scikit-learn, such as `RandomForestClassifier`, `LogisticRegression`, `SVC`, `train_test_split`, `GridSearchCV`, and others.

2. **TensorFlow:**
   - **Overview:** TensorFlow is an open-source machine learning library developed by Google. It is widely used for building and training deep learning models.
   - **Usage in Code:** The code does not explicitly use TensorFlow.

3. **Keras:**
   - **Overview:** Keras is a high-level neural networks API written in Python. It is capable of running on top of TensorFlow, Theano, or Microsoft Cognitive Toolkit (CNTK).
   - **Usage in Code:** The code does not explicitly use Keras.

4. **NumPy:**
   - **Overview:** NumPy is a fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements.
   - **Usage in Code:** Yes, the code uses NumPy for numerical operations and array manipulations.

5. **Pandas:**
   - **Overview:** Pandas is a data manipulation and analysis library. It provides data structures for efficiently storing large datasets and tools for working with them.
   - **Usage in Code:** Yes, the code uses Pandas for data manipulation and analysis, including reading data from a CSV file, dropping columns, and filling missing values.

6. **Seaborn:**
   - **Overview:** Seaborn is a statistical data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
   - **Usage in Code:** Yes, the code uses Seaborn for creating count plots and heatmaps to visualize data distribution and correlation.

7. **Matplotlib:**
   - **Overview:** Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is often used in conjunction with other libraries for data visualization.
   - **Usage in Code:** Yes, the code uses Matplotlib for creating various plots, including box plots, distribution plots, and heatmaps.

In summary, the code leverages Scikit-learn, NumPy, Pandas, Seaborn, and Matplotlib for tasks related to data analysis, preprocessing, and machine learning model evaluation and visualization.




1. Minimizing False Positives (Approving a loan when it shouldn't be approved):
Model: Logistic Regression
Metrics: Precision, Confusion Matrix, ROC Curve, AUC Score
Variables: model1, y_pred_logit
2. Capturing as Many Defaults as Possible:
Model: Support Vector Machine (SVM)
Metrics: Recall, Confusion Matrix, ROC Curve, AUC Score
Variables: model2, y_pred_svm
3. Balancing Precision and Recall:
Model: Random Forest
Metrics: F1 Score, Confusion Matrix, ROC Curve, AUC Score
Variables: model3, y_pred_rf
4. Overall Performance:
Model: XGBoost
Metrics: Accuracy, Confusion Matrix, ROC Curve, AUC Score
Variables: model4, y_pred_xgb

It looks like you've provided a detailed description of a dataset, including information about tables, rows, and columns. The dataset appears to be related to loan applications, with various features and attributes associated with each application.

Here's a summary of the key information based on the provided description:

### Table: Special
- **Label Count:**
  - 1.00 - 22.20: 20
  - 22.20 - 43.40: 21
  - 43.40 - 64.60: 21
  - 64.60 - 85.80: 21
  - 85.80 - 107.00: 21
  - 107.00 - 128.20: 18
  - 170.60 - 191.80: 16
  - 191.80 - 213.00: 22

### Table: application_data (Partial Column Descriptions)
- **SK_ID_CURR:** ID of the loan in the sample
- **TARGET:** Target variable (1 - client with payment difficulties)
- **NAME_CONTRACT_TYPE:** Identification if the loan is cash or revolving
- **CODE_GENDER:** Gender of the client
- **FLAG_OWN_CAR:** Flag if the client owns a car
- **FLAG_OWN_REALTY:** Flag if the client owns a house or flat
- **CNT_CHILDREN:** Number of children the client has
- ... (and many more columns)

### Table: previous_application.csv (Partial Column Descriptions)
- **SK_ID_PREV:** ID of the previous credit in Home credit related to the loan in the sample
- **SK_ID_CURR:** ID of the loan in the sample
- **NAME_CONTRACT_TYPE:** Contract product type of the previous application
- **AMT_ANNUITY:** Annuity of the previous application
- **AMT_APPLICATION:** For how much credit did the client ask on the previous application
- ... (and many more columns)

This dataset seems to contain a mix of numerical and categorical features related to loan applications, with information about the applicants, their financial history, and details about previous applications. It also includes normalized scores, time-related features, and flags indicating various aspects of the applications.

If you have any specific questions or tasks related to this dataset, feel free to ask!





It looks like you've shared a comprehensive code for data preprocessing, exploration, and classification using various machine learning algorithms. Here's a brief overview of the main steps in your code:

1. **Data Loading and Initial Exploration:**
   - Loading the dataset using Pandas.
   - Displaying the first few rows of the dataset.
   - Checking the shape, size, and data types of the DataFrame.

2. **Data Cleaning:**
   - Removing columns with a high number of missing values.
   - Dropping specific columns from the dataset.

3. **Handling Missing Values:**
   - Checking and visualizing missing values.
   - Filling missing values using mean, median, or mode depending on the data type.

4. **Encoding Categorical Variables:**
   - Using Label Encoding to convert categorical variables into numerical format.

5. **Outlier Detection and Treatment:**
   - Identifying outliers using the IQR method.
   - Replacing outliers with null values.

6. **Data Visualization:**
   - Visualizing the distribution of numerical and categorical features.
   - Creating box plots for outlier detection.

7. **Undersampling:**
   - Using Random UnderSampling to balance the target variable.

8. **Model Building:**
   - Splitting the data into training and testing sets.
   - Performing a grid search for hyperparameter tuning on a Random Forest Classifier.
   - Building and evaluating models using Random Forest, Logistic Regression, Support Vector Machine (SVM), and XGBoost.

9. **Model Evaluation:**
   - Displaying confusion matrices, accuracy, precision, recall, F1 score, ROC curves, and AUC for each model.

10. **Neural Network:**
   - Using a simple neural network for classification.

Overall, it seems like you have covered a lot of ground in terms of data preprocessing, exploration, and model building. If you have any specific questions or if there's something specific you'd like to discuss or improve, feel free to let me know!


Q-I want to know that on which rows am I applying machine algorithms. Explain the whole code to me like i am a noob in EDA and machine learning. I want to understand the relation on which the machine algorithm is being applied in this code


Certainly! Let's break down the code into sections and explain each part:

### Section 1: Importing Libraries and Loading Data
```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import imblearn
from imblearn.under_sampling import RandomUnderSampler
```
In this section, you are importing the required Python libraries for data analysis, visualization, and machine learning. Pandas is used for handling data, NumPy for numerical operations, Seaborn and Matplotlib for visualization, and scikit-learn for machine learning algorithms. The imbalanced-learn library is also imported for undersampling.

```python
# Loading the dataset
df = pd.read_csv('/Users/bhavya/Downloads/application_data.csv')
```
You are reading a CSV file containing your dataset into a Pandas DataFrame named `df`.

### Section 2: Initial Data Exploration and Cleaning
```python
# Displaying the first few rows of the dataset
df.head()

# Checking the shape and size of the dataset
df.shape
df.size

# Displaying data types of each column
df.dtypes
```
Here, you are inspecting the initial structure of your dataset by displaying the first few rows, checking its dimensions (shape and size), and examining the data types of each column.

```python
# Dropping specified columns from the dataset
df.drop(["list_of_columns_to_drop"], axis=1, inplace=True)
df.drop(["list_of_other_columns_to_drop"], axis=1, inplace=True)
```
You are removing certain columns from the dataset using the `drop` method. The columns to drop are specified in the lists.

### Section 3: Further Data Cleaning
```python
# Displaying the last few rows of the dataset
df.tail()

# Dropping additional columns from the dataset
df.drop(["list_of_more_columns_to_drop"], axis=1, inplace=True)
```
Similar to the previous step, you are displaying the last few rows and dropping more columns from the dataset.

### Section 4: Checking and Handling Missing Values
```python
# Displaying column names
df.columns

# Displaying information about the dataset
df.info()

# Displaying summary statistics of the dataset
df.describe()

# Function to calculate and display missing values
def missing(df1):
    # ...
    return missing_values

# Displaying the count of missing values in each column
df.isnull().sum()
```
You are inspecting the column names, obtaining information about the dataset, and calculating the count and percentage of missing values for each column using the `missing` function.

```python
# Filling null values with mean, median, or mode
for i in df.columns:
    if df[i].dtypes == 'object':
        df[i].fillna(df[i].mode()[0], inplace=True)
    else:
        df[i].fillna(df[i].median(), inplace=True)
```
You are filling missing values in each column based on the data type. For object (categorical) columns, you are using the mode, and for numerical columns, you are using the median.

### Section 5: Identifying Numerical and Categorical Data
```python
# Identifying numerical and categorical columns
numerical = df.drop(['TARGET'], axis=1).select_dtypes('number').columns
categorical = df.select_dtypes('object').columns

# Displaying the identified columns
print(f'Numerical Columns:  {df[numerical].columns}')
print('\n')
print(f'Categorical Columns: {df[categorical].columns}')
```
You are identifying numerical and categorical columns in the dataset. The results are printed to the console.

### Section 6: Encoding Categorical Variables
```python
# Importing label encoder
from sklearn import preprocessing

# Creating label_encoder object
label_encoder = preprocessing.LabelEncoder()

# Encoding 'NAME_INCOME_TYPE' column
df['NAME_INCOME_TYPE'] = label_encoder.fit_transform(df['NAME_INCOME_TYPE'])

# Encoding 'ORGANIZATION_TYPE' column
df['ORGANIZATION_TYPE'] = label_encoder.fit_transform(df['ORGANIZATION_TYPE'])
```
You are using Label Encoding to convert categorical variables ('NAME_INCOME_TYPE' and 'ORGANIZATION_TYPE') into numerical format.

### Section 7: Handling Outliers
```python
# Checking for outliers and replacing them with null values
out_to_null(data=df)
```
You are identifying outliers in numerical columns and replacing them with null values.

### Section 8: Data Visualization
```python
# Count plot of data types
sns.countplot(df.dtypes.map(str))
plt.show()
```
You are creating a count plot to visualize the distribution of data types in the dataset.

### Section 9: Undersampling
```python
# Defining columns and target variable
cols = df.columns.tolist()
cols = [c for c in cols if c not in ["TARGET"]]
target = "TARGET"

# Defining X and Y
X = df[cols]
Y = df[target]

# Undersampling
X_under, Y_under = undersample.fit_resample(X, Y)
```
You are undersampling the dataset using RandomUnderSampler to balance the target variable.

### Section 10: Splitting Data and Hyperparameter Tuning
```python
# Splitting the undersampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.3)

# Defining hyperparameters for grid search
param_grid = {...}

# Creating a grid search for hyperparameter tuning
grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5)

# Fitting the grid search to the training data
grid_search.fit(X_train, y_train)
```
You are splitting the undersampled data into training and testing sets, defining hyperparameters for a Random Forest Classifier, and performing a grid search for hyperparameter tuning.

### Section 11: Model Building and Evaluation
```python
# Getting the best parameters and estimator from the grid search
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Using the best estimator for predictions
y_pred = best_estimator.predict(X_test)

# Building and evaluating models using various classifiers (Random Forest, Logistic Regression, SVM, XGBoost)
# ...

# Evaluating a simple neural network
# ...
```
You are building and evaluating machine learning models using classifiers such as Random Forest, Logistic Regression, SVM, and XGBoost. Additionally, you are applying a simple neural network for classification.

This breakdown should provide you with an overview of each section and its purpose in the code. If you have specific questions about any part or need further clarification, feel free to ask!


Certainly! Let's break down Section 11, which is focused on building and evaluating machine learning models.

### Section 11: Model Building and Evaluation
```python
# Section 11: Model Building and Evaluation

# Importing packages for modeling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Train the model - Logistic Regression
model1 = LogisticRegression(random_state=42)
logit = model1.fit(X_train, y_train)

# Predictions
y_pred_logit = model1.predict(X_test) 

# Scores
print("Accuracy Logit:", metrics.accuracy_score(y_test, y_pred_logit))
print("Precision Logit:", metrics.precision_score(y_test, y_pred_logit))
print("Recall Logit:", metrics.recall_score(y_test, y_pred_logit))
print("F1 Score Logit:", metrics.f1_score(y_test, y_pred_logit))

# Confusion Matrix
matrix_logit = confusion_matrix(y_test, y_pred_logit)
cm_logit = pd.DataFrame(matrix_logit, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

# Heatmap of Confusion Matrix
sns.heatmap(cm_logit, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix Logit"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

# AUC (Area Under the Curve) Calculation
y_pred_logit_proba = model1.predict_proba(X_test)[:, 1]
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_logit_proba)
auc_logit = roc_auc_score(y_test, y_pred_logit_proba)
print("AUC Logistic Regression :", auc_logit)

# ROC (Receiver Operating Characteristic) Curve
plt.plot(fpr_logit, tpr_logit, label="Logistic Regression, auc={:.3f})".format(auc_logit))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic Regression ROC curve')
plt.legend(loc=4)
plt.show()

# Precision-Recall Curve
logit_precision, logit_recall, _ = precision_recall_curve(y_test, y_pred_logit_proba)
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(logit_recall, logit_precision, color='orange', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
```

This section involves training, evaluating, and visualizing the results of different machine learning models. Here's a step-by-step breakdown:

1. **Logistic Regression Model:**
   - A logistic regression model is created and trained on the training data (`X_train` and `y_train`).
   - Predictions are made on the test data (`X_test`).
   - Model performance metrics (accuracy, precision, recall, and F1 score) are calculated and printed.
   - Confusion matrix is generated and displayed as a heatmap.
   - ROC curve and AUC score are calculated and plotted.
   - Precision-Recall curve is plotted.

This process is repeated for other models in the subsequent sections (`SVM`, `Random Forest`, `XGBoost`). Each model follows a similar structure: training, prediction, performance metrics calculation, confusion matrix generation, ROC curve plotting, and Precision-Recall curve plotting.

You can adapt this structure for additional models or modify hyperparameters for optimization. The evaluation metrics and visualizations help assess how well each model performs on the given data.


Certainly! Let's break down each step in more detail for each machine learning model (Logistic Regression, SVM, Random Forest, and XGBoost).

### 1. Logistic Regression Model
```python
# Train the model - Logistic Regression
model1 = LogisticRegression(random_state=42)
logit = model1.fit(X_train, y_train)
```
- **Training the Model:** A logistic regression model is created (`LogisticRegression()`) and trained using the training data (`X_train`, `y_train`).

```python
# Predictions
y_pred_logit = model1.predict(X_test)
```
- **Making Predictions:** The trained model is used to make predictions on the test data (`X_test`), resulting in `y_pred_logit`.

```python
# Scores
print("Accuracy Logit:", metrics.accuracy_score(y_test, y_pred_logit))
print("Precision Logit:", metrics.precision_score(y_test, y_pred_logit))
print("Recall Logit:", metrics.recall_score(y_test, y_pred_logit))
print("F1 Score Logit:", metrics.f1_score(y_test, y_pred_logit))
```
- **Model Performance Metrics:** Various performance metrics such as accuracy, precision, recall, and F1 score are calculated using the predicted labels (`y_pred_logit`) and the actual labels (`y_test`). These metrics provide insights into how well the model is performing.

```python
# Confusion Matrix
matrix_logit = confusion_matrix(y_test, y_pred_logit)
cm_logit = pd.DataFrame(matrix_logit, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])
```
- **Confusion Matrix:** A confusion matrix is generated to visualize the count of true positive, true negative, false positive, and false negative predictions. This matrix is then displayed as a heatmap.

```python
# AUC (Area Under the Curve) Calculation
y_pred_logit_proba = model1.predict_proba(X_test)[:, 1]
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_logit_proba)
auc_logit = roc_auc_score(y_test, y_pred_logit_proba)
print("AUC Logistic Regression :", auc_logit)
```
- **AUC Calculation:** The Area Under the Receiver Operating Characteristic (ROC) Curve is calculated to measure the model's ability to distinguish between classes. The ROC curve is created by plotting the true positive rate against the false positive rate.

```python
# ROC Curve
plt.plot(fpr_logit, tpr_logit, label="Logistic Regression, auc={:.3f})".format(auc_logit))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic Regression ROC curve')
plt.legend(loc=4)
plt.show()
```
- **ROC Curve Plotting:** The ROC curve is plotted to visualize the trade-off between true positive rate and false positive rate. The area under this curve (AUC) is a summary measure of the model's performance.

```python
# Precision-Recall Curve
logit_precision, logit_recall, _ = precision_recall_curve(y_test, y_pred_logit_proba)
no_skill = len(y_test[y_test == 1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(logit_recall, logit_precision, color='orange', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
```
- **Precision-Recall Curve Plotting:** This curve is plotted to visualize the trade-off between precision and recall. It helps understand the model's performance in situations where class imbalance exists.

### 2. Support Vector Machine (SVM)
```python
# Train the model - Support Vector Machine (SVM)
model2 = SVC()
svm = model2.fit(X_train, y_train)
```
- Similar to logistic regression, a Support Vector Machine model is created and trained using the training data.

```python
# Predictions
y_pred_svm = model2.predict(X_test)
```
- Predictions are made on the test data.

... (Continued for SVM)

The steps are analogous for the Random Forest and XGBoost models, with slight variations based on the algorithms used. If you'd like, I can continue with detailed explanations for the SVM, Random Forest, and XGBoost sections as well.





I apologize for any confusion. Let me provide a clearer explanation.

In the provided code, four machine learning models are applied to a dataset to predict whether a loan applicant is likely to default on a loan. The target variable is 'TARGET', where 0 represents no default, and 1 represents a default.

Here's a brief summary of the evaluation metrics used to assess the performance of each model:

1. **Logistic Regression:**
   - **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC Score, Precision-Recall Curve.
   - **Result Interpretation:** The metrics help to evaluate how well the logistic regression model predicts loan default. Accuracy measures the overall correctness of predictions, precision focuses on the accuracy of positive predictions, recall assesses the ability to capture all actual positives, and the F1 score is a balance between precision and recall. The ROC curve and AUC score evaluate the trade-off between true positive rate and false positive rate, while the precision-recall curve assesses the precision-recall trade-off.

2. **Support Vector Machine (SVM):**
   - **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC Score, Precision-Recall Curve.
   - **Result Interpretation:** Similar to logistic regression, SVM is evaluated using various metrics to understand its performance in predicting loan default.

3. **Random Forest:**
   - **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC Score, Precision-Recall Curve.
   - **Result Interpretation:** Random Forest is another model used for prediction, and its performance is assessed using the same set of metrics.

4. **XGBoost:**
   - **Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, AUC Score, Precision-Recall Curve.
   - **Result Interpretation:** XGBoost, like the other models, is evaluated using a range of metrics.

**Interpreting Results:**
- **Accuracy:** The proportion of correctly classified instances. A higher accuracy indicates better overall performance.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. High precision means fewer false positives.
- **Recall:** The ratio of correctly predicted positive observations to the actual positives. High recall means fewer false negatives.
- **F1 Score:** The weighted average of precision and recall. It balances precision and recall.
- **AUC Score:** The area under the ROC curve. A higher AUC score indicates a better trade-off between sensitivity and specificity.
- **Precision-Recall Curve:** It shows the trade-off between precision and recall for different thresholds.

The choice of the "better" model depends on the specific goals and trade-offs relevant to the loan approval scenario. For example, if minimizing false positives (approving a loan when it shouldn't be approved) is crucial, precision may be a more critical metric. If it's crucial to capture as many defaults as possible, recall might be prioritized. The balance between these metrics is crucial and often involves domain expertise and business considerations.











To determine which model is better for each case, you would typically consider the trade-off between precision and recall, as well as other metrics like accuracy and the area under the ROC curve (AUC). The choice depends on the specific business objectives and priorities. Let's break it down:

### Minimizing False Positives (Approving a loan when it shouldn't be approved):

- **Precision** is a crucial metric for this scenario because it measures the accuracy of positive predictions. A high precision means fewer false positives.

- **Model Recommendation:**
  - The model with the highest precision would be considered better for minimizing false positives.

### Capturing as Many Defaults as Possible:

- **Recall** is a critical metric in this case as it measures the ability to capture all actual positives (defaults). High recall means fewer false negatives.

- **Model Recommendation:**
  - The model with the highest recall would be considered better for capturing as many defaults as possible.

### Balancing Precision and Recall:

- **F1 Score** provides a balance between precision and recall. It might be useful to consider if you want a single metric that considers both false positives and false negatives.

### Overall Performance:

- **Accuracy** provides an overall measure of correct predictions, but it might not be the best metric if there is a class imbalance (e.g., a small percentage of defaults).

- **AUC Score:** The model with a higher AUC score generally has a better trade-off between sensitivity (recall) and specificity (precision).

### Recommendation:

- **Compare Models:** Evaluate the precision, recall, F1 score, accuracy, and AUC for each model.
  
- **Consider Business Priorities:** Choose the model that aligns better with business priorities. If minimizing false positives is more critical, prioritize precision. If capturing defaults is more critical, prioritize recall.

- **Threshold Adjustment:** Depending on the specific business needs, you might adjust the classification threshold to increase precision or recall.

It's important to note that the "better" model depends on the specific goals and constraints of the business problem. It's often a matter of finding the right balance that aligns with the business's risk tolerance and priorities.




Certainly! Here's a summary of the models applied in the provided code for different categories:

### 1. Minimizing False Positives (Approving a loan when it shouldn't be approved):

- **Model:** Logistic Regression
  - **Metrics:** Precision, Confusion Matrix, ROC Curve, AUC Score
  - **Variables:** `model1`, `y_pred_logit`

### 2. Capturing as Many Defaults as Possible:

- **Model:** Support Vector Machine (SVM)
  - **Metrics:** Recall, Confusion Matrix, ROC Curve, AUC Score
  - **Variables:** `model2`, `y_pred_svm`

### 3. Balancing Precision and Recall:

- **Model:** Random Forest
  - **Metrics:** F1 Score, Confusion Matrix, ROC Curve, AUC Score
  - **Variables:** `model3`, `y_pred_rf`

### 4. Overall Performance:

- **Model:** XGBoost
  - **Metrics:** Accuracy, Confusion Matrix, ROC Curve, AUC Score
  - **Variables:** `model4`, `y_pred_xgb`

### Note:
- Each model is trained on the `X_train` and `y_train` datasets.
- Predictions are made on the test dataset (`X_test`).
- Various performance metrics (precision, recall, accuracy, F1 score, ROC curve, and AUC) are calculated for each model.
- The confusion matrix provides a breakdown of true positives, true negatives, false positives, and false negatives.
- The ROC curve visualizes the trade-off between sensitivity (recall) and specificity (precision) at different classification thresholds.
- AUC (Area Under the Curve) score summarizes the ROC curve's performance in a single value.

### Recommendation:
- Examine the performance metrics for each model and choose based on the specific business priorities.
- Consider the trade-off between precision and recall.
- Adjust the classification threshold if needed based on business requirements.

Remember, the choice of the "best" model depends on the business context and objectives.


Yes, that's correct. Logistic Regression is often a suitable model when minimizing false positives (approving a loan when it shouldn't be approved) is crucial. The precision metric, which is a key consideration in this scenario, is one of the strengths of Logistic Regression.

In the context of credit risk or loan approval, false positives (approving a loan for someone who might default) can have significant financial implications. Precision, in this case, measures the accuracy of the positive predictions, i.e., the proportion of approved loans that are genuinely good.

Therefore, if your primary concern is minimizing the risk of approving loans for applicants who might default, Logistic Regression can be a suitable choice based on its precision performance.


Yes, that's correct. When the goal is to capture as many defaults as possible, prioritizing recall, Support Vector Machine (SVM) might be a suitable choice. Recall measures the ability of a model to capture all the relevant cases of a certain class (in this context, capturing all loan defaults).

In the context of credit risk or loan approval, capturing as many defaults as possible is crucial for identifying high-risk applicants and preventing potential financial losses. SVM, with proper tuning and calibration, can be effective in identifying cases of loan default.

However, it's essential to note that there's often a trade-off between precision and recall. While SVM might perform well in capturing defaults (high recall), it may also generate more false positives (lower precision). The choice between models depends on the specific goals and priorities of the lending institution, considering the balance between false positives and false negatives.




Yes, that's a reasonable assessment. Random Forest is known for providing a balance between precision and recall. It is an ensemble learning method that combines the predictions of multiple decision trees, and it can be effective in capturing complex relationships in the data.

Random Forest tends to offer a good trade-off between precision and recall, making it suitable for scenarios where both minimizing false positives and capturing as many positives as possible are important. The ensemble nature of Random Forest helps in reducing overfitting and improving the generalization of the model.

In the context of credit risk or loan approval, where a balance between precision and recall is crucial, Random Forest can be a robust choice. However, it's important to note that the optimal choice of model depends on the specific characteristics of the dataset and the goals of the lending institution. It's often recommended to experiment with multiple models and evaluate their performance based on the business requirements.



The choice of the best model for overall performance can depend on various factors, and it's not always straightforward to declare one model as universally superior. However, XGBoost is indeed a powerful and widely used algorithm that often performs well across different types of datasets.

XGBoost (Extreme Gradient Boosting) is an ensemble learning method that has gained popularity for its speed and performance. It is an extension of gradient boosting algorithms and is known for handling missing data, regularization, and providing high predictive accuracy.

In many cases, XGBoost can deliver competitive or even superior performance compared to other algorithms. It's suitable for a wide range of applications and is effective in capturing complex relationships within the data.

When considering overall performance, factors such as computational efficiency, interpretability, and ease of use may also play a role in the selection of the best model. It's recommended to perform thorough cross-validation and hyperparameter tuning to ensure that the chosen model performs well on the specific dataset at hand.




Certainly! Here's a brief overview of each library and whether they have been used in the provided code:

1. **Scikit-learn (Sklearn):**
   - **Overview:** Scikit-learn is a machine learning library in Python. It provides simple and efficient tools for data analysis and modeling, including various machine learning algorithms.
   - **Usage in Code:** Yes, the code uses several modules from Scikit-learn, such as `RandomForestClassifier`, `LogisticRegression`, `SVC`, `train_test_split`, `GridSearchCV`, and others.

2. **TensorFlow:**
   - **Overview:** TensorFlow is an open-source machine learning library developed by Google. It is widely used for building and training deep learning models.
   - **Usage in Code:** The code does not explicitly use TensorFlow.

3. **Keras:**
   - **Overview:** Keras is a high-level neural networks API written in Python. It is capable of running on top of TensorFlow, Theano, or Microsoft Cognitive Toolkit (CNTK).
   - **Usage in Code:** The code does not explicitly use Keras.

4. **NumPy:**
   - **Overview:** NumPy is a fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements.
   - **Usage in Code:** Yes, the code uses NumPy for numerical operations and array manipulations.

5. **Pandas:**
   - **Overview:** Pandas is a data manipulation and analysis library. It provides data structures for efficiently storing large datasets and tools for working with them.
   - **Usage in Code:** Yes, the code uses Pandas for data manipulation and analysis, including reading data from a CSV file, dropping columns, and filling missing values.

6. **Seaborn:**
   - **Overview:** Seaborn is a statistical data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
   - **Usage in Code:** Yes, the code uses Seaborn for creating count plots and heatmaps to visualize data distribution and correlation.

7. **Matplotlib:**
   - **Overview:** Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is often used in conjunction with other libraries for data visualization.
   - **Usage in Code:** Yes, the code uses Matplotlib for creating various plots, including box plots, distribution plots, and heatmaps.

In summary, the code leverages Scikit-learn, NumPy, Pandas, Seaborn, and Matplotlib for tasks related to data analysis, preprocessing, and machine learning model evaluation and visualization.




work for a consumer finance company that specialises in lending various types of loans to customers


Clients having Medium salary range are more likely to apply for the loan for both Defaulters and Non-defaulters.
Pensioners have applied the most for the loan in case of Defaulters and Non-Defaulters.
Married Clients seems to be applied most for the loan compared to others for both Defaulters and Non-Defaulters.
Most of the clients have applied for Cash Loan while very small proportion have applied for Revolving loan for both Defaulters as well as Non-defaulters.
Clients who applied for loans were getting income by Working,Commercial associate and Pensioner are more likely to apply for the loan, highest being the Working class category .


**Univariate Analysis of Numerical Columns W.R.T Target Variable:**

**Definition:**
- **Univariate Analysis:** Involves analyzing one variable at a time. It helps to understand the distribution and characteristics of individual variables.
- **Numerical Columns W.R.T Target Variable:** In the context of machine learning or data analysis, this implies analyzing the distribution and characteristics of numerical columns concerning the target variable.

**Process:**
1. **Single Variable Focus:** Univariate analysis focuses on one numerical variable (feature) at a time.
2. **Distribution Examination:** It includes exploring the distribution of a numerical variable, such as using histograms or kernel density plots.
3. **Statistical Measures:** Descriptive statistics like mean, median, standard deviation, and percentiles are often used.
4. **Visualization:** Visual representations like box plots or violin plots may be employed to understand the spread and central tendency.
5. **Comparison with Target Variable:** The analysis involves comparing the distribution or statistical measures of the numerical variable for different values or groups of the target variable.

**Example:**
- If analyzing the distribution of income (numerical variable) concerning loan approval (target variable), univariate analysis would involve examining the income distribution overall and separately for approved and rejected loans.

**Bivariate Analysis: Numerical & Categorical W.R.T Target Variables:**

**Definition:**
- **Bivariate Analysis:** Involves the simultaneous analysis of two variables. It helps to understand the relationships and interactions between pairs of variables.
- **Numerical & Categorical W.R.T Target Variables:** In the context of machine learning, this implies analyzing the joint influence of both numerical and categorical variables on the target variable.

**Process:**
1. **Two Variable Focus:** Bivariate analysis considers the relationship between a numerical variable and a categorical variable concerning the target variable.
2. **Comparison Across Categories:** It involves comparing the distribution or characteristics of the numerical variable for different categories of the categorical variable (or vice versa).
3. **Statistical Testing:** Statistical tests or measures may be employed to assess the significance of differences or associations.
4. **Visualization:** Visualization techniques like grouped bar charts, box plots, or violin plots can help illustrate the relationship.

**Example:**
- Analyzing the relationship between income (numerical) and loan approval (categorical) by comparing income distributions for approved and rejected loans. This could be done separately for different income brackets or using statistical tests to assess if the income differs significantly between approval categories.

**Summary:**
- Univariate analysis focuses on understanding the characteristics of individual variables independently.
- Bivariate analysis considers the interaction between two variables, particularly their joint influence on the target variable.




·   	Checked the shape and size of dataset
·   	Here, we've decided to remove columns with null values exceeding 35% after careful observation. Tried to find null values in dataset and then remove those null values or replace them with mean, median and mode values 
: We split the numerical and categorical attributes in dataset to have clear idea of knowing how many numerical and categorical attributes

Label encoding:

·   	Statistical Analysis: The statistical values like mean, median, mode, skewness, null_value_count, etc are found here.
 
Outlier analysis: Outliers in a dataset or graph are extreme values that stand out significantly from the main pattern of values.

·   	as we plotted the data we realised Our dataset is highly imbalanced. It's a big problem because classifiers will always predict the most common class without performing any analysis of the features and it will have a high accuracy rate, obviously not the correct one. To change that, we proceeded to undersampling.

Pair plotting It helps us to try to find correlation between attributes.

·   	Feature selection: Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modelling and, in some cases, to improve the performance of the model.


