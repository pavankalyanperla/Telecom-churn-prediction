# REPO_AUDIT.md — Telecom Customer Churn Prediction
**Audit Date:** 2026-06-27  
**Destination Repo:** https://github.com/pavankalyanperla/Telecom-churn-prediction (currently empty)  
**Auditor:** Claude Code (read-only reconnaissance — nothing modified)

---

## 1. GIT STATUS CHECK

**This folder is NOT a git repository.**

```
fatal: not a git repository (or any of the parent directories): .git
```

- No `.git` directory exists anywhere in the project tree.
- `git remote -v` and `git log` both fail with the same fatal error.
- There is zero local commit history.
- The destination GitHub repo (`pavankalyanperla/Telecom-churn-prediction`) is freshly created and empty.

**Action required before pushing:** `git init`, add remote, initial commit.

---

## 2. FULL DIRECTORY TREE

```
Project - Telecom customer churn prediction/          ← project root (OneDrive Desktop)
└── telecom customer churn project/                   ← single subfolder containing ALL files
    ├── customer_churn_data.csv          954.6 KB  (0.93 MB)
    ├── Telecom Customer Churn Prediction Final.html  1,275.3 KB  (1.25 MB)
    ├── Telecom Customer Churn Prediction Final.ipynb   581.5 KB  (0.57 MB)
    ├── Telecom-Customer-Churn-Prediction.pptx        3,311.5 KB  (3.23 MB)
    └── Telecom_Churn_Prediction_Report.pdf             448.2 KB  (0.44 MB)
```

**Total files:** 5  
**Total size:** ~6.5 MB

**Large-file flag:** No file exceeds GitHub's 50 MB soft warning or 100 MB hard limit. All files are safe to push.

**Structural note:** All project files live inside a subfolder called `telecom customer churn project` rather than in the repo root. The project root itself contains nothing. This means the GitHub repo root will show only one folder unless restructured.

---

## 3. NOTEBOOK: `Telecom Customer Churn Prediction Final.ipynb`

### 3.1 Summary Statistics
| Property | Value |
|---|---|
| Kernel | Python 3 (ipykernel) |
| Total cells | 118 (cells 0–117) |
| Code cells | 74 |
| Markdown cells | 44 |
| Cells with stored error outputs | **0** |
| Cells with DeprecationWarnings in output | 1 (cell 0) |

### 3.2 Narrative / Workflow Mapping

| Step | Present? | Cells |
|---|---|---|
| Session reset / environment setup | Yes | 0 |
| Import libraries | Yes | 2, 4, 6, 8 |
| Data loading | Yes | 12 |
| Initial EDA (head, dtypes, describe, nulls) | Yes | 13–23 |
| Class distribution check | Yes | 28 |
| Data cleaning (type coercion, null fill) | Yes | 30–36 |
| Label encoding (binary columns) | Yes | 39–41 |
| Histograms / distribution plots | Yes | 46–57 |
| Churn rate plots (contract, payment method) | Yes | 59–66 |
| Correlation analysis & VIF | Yes | 68–80 |
| EDA summary | Yes | 81 |
| Feature engineering (OHE, drop columns) | Yes | 83–89 |
| Train/test split | Yes | 92 |
| Feature scaling | Yes | 96–98 |
| Model training & evaluation | Partial | 100–115 |
| Cross-validation (KFold) | **MISSING** | — |
| Hyperparameter tuning (GridSearchCV) | **MISSING** | — |
| ROC curve / AUC plot | **MISSING** | — |
| Confusion matrix heatmap | **MISSING** | — |
| Model serialization (joblib.dump) | **MISSING** | — |
| Conclusion / summary | Partial | 117 (just a to-do list) |

**Missing steps are explicitly planned in cell 117 but never executed.** The notebook ends with a bulleted future-plan list, not actual results.

### 3.3 Full Code — Every Code Cell (in order)

**Cell 0** — Session Reset
```python
#Restart the session
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
```
*Output: Two DeprecationWarnings for `magic()` usage.*

---

**Cell 2** — Standard Analysis Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm
```

---

**Cell 4** — Preprocessing Imports
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder          # DUPLICATE IMPORT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

**Cell 6** — Model Imports
```python
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.neighbors import KNeighborsClassifier       # DUPLICATE IMPORT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

---

**Cell 8** — Evaluation & Utility Imports
```python
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score
import seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt                          # DUPLICATE IMPORT
import matplotlib.pylab as pylab
import matplotlib 
%matplotlib inline
color = sn.color_palette()
import matplotlib.ticker as mtick
from IPython.display import display
pd.options.display.max_columns = None
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve                   # DUPLICATE IMPORT
import random
import os
import re
import sys
import timeit
import string
import time
from datetime import datetime
from time import time
from dateutil.parser import parse
import joblib
```

---

**Cell 10** — *(Empty code cell — no source)*

---

**Cell 12** — Data Loading
```python
dataset = pd.read_csv('customer_churn_data.csv')
```

---

**Cell 13** — Preview
```python
dataset.head()
```
*Output: First 5 rows shown (see CSV section below).*

---

**Cell 14** — Column Names
```python
dataset.columns
```
*Output: 21 columns listed.*

---

**Cell 15** — Descriptive Stats
```python
dataset.describe()
#Average customer stays for 32 days and pays about $65. But there may be categories with different ranges
#Since there was just 3 columns in describe we can assume that there are lot of non numerical/categorical coolumns
```
*Output: Only 3 numeric columns appear (SeniorCitizen, tenure, MonthlyCharges). TotalCharges is object dtype.*

---

**Cell 16** — dtypes
```python
dataset.dtypes
```

---

**Cell 17** — Null Check
```python
dataset.isnull().sum()
```
*Output: All 0 nulls (before TotalCharges coercion).*

---

**Cell 19** — Group by dtype
```python
dataset.columns.to_series().groupby(dataset.dtypes).groups
```

---

**Cell 20** — Dataset Info
```python
dataset.info()
```
*Output: 7043 rows, 21 columns, 1.1+ MB.*

---

**Cell 22** — isna check
```python
dataset.isna().sum()
```

---

**Cell 23** — isna any
```python
dataset.isna().any()
```

---

**Cell 25** — PaymentMethod unique values
```python
dataset["PaymentMethod"].nunique()
dataset["PaymentMethod"].unique()
```
*Output: 4 unique payment methods.*

---

**Cell 26** — Contract unique values
```python
dataset["Contract"].nunique()
dataset["Contract"].unique()
```
*Output: 3 contract types.*

---

**Cell 28** — Target Distribution
```python
dataset["Churn"].value_counts()
```
*Output: No=5174, Yes=1869*

---

**Cell 30** — Type Coercion
```python
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'],errors='coerce')
dataset['TotalCharges'] = dataset['TotalCharges'].astype("float")
```

---

**Cell 31** — Post-coercion null check
```python
dataset.isnull().any()
dataset.isnull().sum()
```
*Output: TotalCharges now has 11 nulls (blank string rows that became NaN).*

---

**Cell 34** — Find null columns
```python
na_cols = dataset.isnull().any()
```

---

**Cell 35** — List null column names
```python
na_cols = na_cols[na_cols == True].reset_index()['index'].to_list()
```

---

**Cell 36** — Fill nulls with column mean
```python
for col in dataset.columns[1:]:
    if col in na_cols:
        print(col)  
        if dataset[col].dtypes != 'object':
            dataset[col]  =dataset[col] .fillna(dataset[col].mean()).round(0)
            
dataset.isnull().sum()          
```
*Output: "TotalCharges" printed; all 0 nulls afterwards.*

---

**Cell 39** — LabelEncoder Init
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
```

---

**Cell 41** — Encode Binary Columns
```python
le_count = 0
for i in dataset.columns[1:]:
    if dataset[i].dtypes == "object":
        print(len(dataset[i].unique()))
        if len(dataset[i].unique()) ==2:
            dataset[i] = le.fit_transform(dataset[i])
            le_count +=1
            
print("{} columns were encoded".format(le_count))
print(f"{le_count} columns were encoded")
```
*Output: 6 binary columns encoded (gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn).*

---

**Cell 44** — Subset for Histograms
```python
dataset2 = dataset[['gender', 
'SeniorCitizen', 'Partner','Dependents',
'tenure', 'PhoneService', 'PaperlessBilling',
'MonthlyCharges', 'TotalCharges']]
```

---

**Cell 46** — Histograms + Contract Distribution
```python
fig = plt.figure(figsize=(15,12))
plt.suptitle('Histograms of Numerical Columns\n',horizontalalignment="center",fontstyle = "normal", fontsize = 24, fontfamily = "sans-serif")
for i in range(dataset2.shape[1]):
    plt.subplot(6,3,i+1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    plt.hist(dataset2.iloc[:, i], bins=vals, color = '#ec838a')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
contract_split = dataset[[ "customerID", "Contract"]]
contract_split = contract_split.groupby ("Contract")['customerID'].count().reset_index()   
contract_split.rename(columns={'customerID':'No. of customers'}, inplace=True)
```
*Output: histogram plot image (too large to inline).*

---

**Cell 47** — Contract Bar Chart
```python
ax =  contract_split[["No. of customers"]].plot.bar(title = 'Customers by Contract Type', legend =True, table = False, grid = False,  subplots = False,  figsize =(12, 7), color ='#ec838a', fontsize = 15, stacked=False)
plt.ylabel('No. of Customers\n',horizontalalignment="center",fontstyle = "normal", fontsize = "large", fontfamily = "sans-serif")
plt.xlabel('\n Contract Type',horizontalalignment="center",fontstyle = "normal", fontsize = "large", fontfamily = "sans-serif")
plt.title('Customers by Contract Type \n',horizontalalignment="center", fontstyle = "normal", fontsize = "22", fontfamily = "sans-serif")
plt.legend(loc='upper right', fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
```

---

**Cell 48** — x_labels array
```python
x_labels = np.array(contract_split[["No. of customers"]])
```

---

**Cell 49** — Value Label Helper (FIRST DEFINITION)
```python
def add_value_labels(ax, spacing=5):   
    for rect in ax.patches:      
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2       
        space = spacing        
        va = 'bottom'      
        if y_value < 0:           
            space *= -1            
            va = 'top'       
        label = "{:.0f}".format(y_value)      
        ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va=va)                                                             
add_value_labels(ax)
```

---

**Cell 51** — Payment Method Split
```python
payment_method_split = dataset[[ "customerID", "PaymentMethod"]]
sectors = payment_method_split  .groupby ("PaymentMethod")
payment_method_split  = pd.DataFrame(sectors["customerID"].count())
payment_method_split.rename(columns={'customerID':'No. of customers'}, inplace=True)
```

---

**Cell 52** — Payment Method Bar Chart (real data)
```python
ax =  payment_method_split [["No. of customers"]].plot.bar(title = 'Customers by Payment Method', legend =True, table = False, grid = False,  subplots = False,  figsize =(15, 10), color ='#ec838a', fontsize = 15, stacked=False)
```

---

**Cell 53** — ⚠️ DUMMY DATA PLACEHOLDER PLOT
```python
# Example dummy data
contracts = ['Month-to-month', 'One year', 'Two year']
customers = [350, 150, 200]

# Plot bar chart
plt.bar(contracts, customers, label='Auto-pay Customers')
plt.title('Customers by Payment Method', fontsize=22, fontweight='bold')
plt.xlabel('Contract Type', fontsize=12, fontweight='bold')
plt.ylabel('No. of Customers', fontsize=12, fontweight='bold')
plt.legend()
plt.show()
```
*⚠️ This cell uses hardcoded fake data `[350, 150, 200]` mislabelled as "Customers by Payment Method". It should not be in a final portfolio notebook.*

---

**Cell 54** — x_labels for payment method
```python
x_labels = np.array(payment_method_split [["No. of customers"]])
```

---

**Cell 55** — Value Label Helper (SECOND / DUPLICATE DEFINITION)
```python
def add_value_labels(ax, spacing=5):   
    for rect in ax.patches:      
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2       
        space = spacing        
        va = 'bottom'      
        if y_value < 0:           
            space *= -1            
            va = 'top'       
        label = "{:.0f}".format(y_value)      
        ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va=va)                                                             
add_value_labels(ax)
```
*Identical to cell 49 — exact duplicate.*

---

**Cell 56** — Services List
```python
services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
```

---

**Cell 57** — Services Bar Charts
```python
fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
for i, item in enumerate(services):
    if i < 3:
        ax = dataset[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0, color ='#f3babc' )
    elif i >=3 and i < 6:
        ax = dataset[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0,color ='#9b9c9a')
    elif i < 9:
        ax = dataset[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0,color = '#ec838a')
    ax.set_title(item)
```

---

**Cell 59** — Overall Churn Rate Bar
```python
import matplotlib.ticker as mtick
churn_rate = dataset[["Churn", "customerID"]]
churn_rate ["churn_label"] = pd.Series(np.where((churn_rate["Churn"] == 0), "No", "Yes"))
sectors = churn_rate .groupby ("churn_label")
churn_rate = pd.DataFrame(sectors["customerID"].count())
churn_rate ["Churn Rate"] = (churn_rate ["customerID"] / sum(churn_rate ["customerID"]) )*100
ax =  churn_rate[["Churn Rate"]].plot.bar(title = 'Overall Churn Rate', legend =True, table = False, grid = False,  subplots = False,  figsize =(12, 7), color = '#ec838a', fontsize = 15, stacked=False, ylim =(0,100))
```

---

**Cell 60** — Churn Rate Chart Labels
```python
plt.ylabel('Proportion of Customers',horizontalalignment="center",fontstyle = "normal", fontsize = "large", fontfamily = "sans-serif")
plt.xlabel('Churn',horizontalalignment="center",fontstyle = "normal", fontsize = "large", fontfamily = "sans-serif")
plt.title('Overall Churn Rate \n',horizontalalignment="center", fontstyle = "normal", fontsize = "22", fontfamily = "sans-serif")
plt.legend(loc='upper right', fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
x_labels = np.array(churn_rate[["customerID"]])
```

---

**Cell 61** — Churn Rate Labels with % formatting
```python
def add_value_labels(ax, spacing=5):   
    for rect in ax.patches:     
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2       
        space = spacing
        va = 'bottom'        
        if y_value < 0:           
            space *= -1          
            va = 'top'
        label = "{:.1f}%".format(y_value)    
        ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va=va)                                                            
add_value_labels(ax)
ax.autoscale(enable=False, axis='both', tight=False)
```

---

**Cell 63** — Churn Rate by Contract Type (Stacked Bar)
```python
import matplotlib.ticker as mtick
contract_churn =dataset.groupby(['Contract','Churn']).size().unstack()
contract_churn.rename(columns={0:'No', 1:'Yes'}, inplace=True)
colors  = ['#ec838a','#9b9c9a']
ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',width = 0.3,stacked = True,rot = 0,figsize = (12,7),color = colors)
plt.ylabel('Proportion of Customers\n', ...)
plt.xlabel('Contract Type\n', ...)
plt.title('Churn Rate by Contract type \n', ...)
plt.legend(loc='upper right', fontsize = "medium")
plt.xticks(rotation=0, horizontalalignment="center")
plt.yticks(rotation=0, horizontalalignment="right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.text(x+width/2, y+height/2, '{:.1f}%'.format(height), ...)
ax.autoscale(enable=False, axis='both', tight=False)
```

---

**Cell 65** — Payment Method Group
```python
import matplotlib.ticker as mtick
contract_churn = dataset.groupby(['Contract','PaymentMethod']).size().unstack()
contract_churn.rename(columns={0:'No', 1:'Yes'}, inplace=True)
```

---

**Cell 66** — Churn Rate by Payment Method (Stacked Bar)
```python
colors  = ['#ec838a','#9b9c9a', '#f3babc' , '#4d4f4c']
ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',width = 0.3,stacked = True,rot = 0,figsize = (12,7),color = colors)
plt.ylabel('Proportion of Customers\n', ...)
plt.xlabel('Contract Type\n', ...)
plt.title('Churn Rate by Payment Method \n', ...)
# [same labelling pattern as cell 63]
```

---

**Cell 68** — Correlations with Churn
```python
dataset2 = dataset[['SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'PaperlessBilling',
        'MonthlyCharges', 'TotalCharges']]
correlations = dataset2.corrwith(dataset.Churn)
correlations = correlations[correlations!=1]
positive_correlations = correlations[correlations >0].sort_values(ascending = False)
negative_correlations =correlations[correlations<0].sort_values(ascending = False)
print('Most Positive Correlations: \n', positive_correlations)
print('\nMost Negative Correlations: \n', negative_correlations)
```
*Output:*
```
Most Positive Correlations: 
 MonthlyCharges      0.193356
 PaperlessBilling    0.191825
 SeniorCitizen       0.150889
 PhoneService        0.011942

Most Negative Correlations: 
 Partner        -0.150448
 Dependents     -0.164221
 TotalCharges   -0.199426
 tenure         -0.352229
```

---

**Cell 69** — Correlation filter
```python
correlations =  dataset2.corrwith(dataset.Churn)
correlations = correlations[correlations!= -1]
```

---

**Cell 70** — Correlation Bar Chart
```python
correlations.plot.bar(figsize = (18, 10), fontsize = 15, color = '#ec838a', rot = 45, grid = True)
plt.title('Correlation with Churn Rate \n', ...)
```

---

**Cell 72** — ⚠️ Correlation Heatmap (STRAY BACKSLASH + DEPRECATED np.bool)
```python
\sn.set(style="white")          # ← STRAY BACKSLASH — this line is syntactically a string literal, not a function call
corr = dataset2.corr()
mask = np.zeros_like(corr, dtype=np.bool)    # ← np.bool DEPRECATED/REMOVED in NumPy >= 1.24
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr = dataset2.corr()    
```
*The heatmap rendered (display_data output exists), suggesting this ran on a NumPy version <1.24. Will break on current NumPy.*

---

**Cell 74** — VIF Function
```python
def calc_vif(x):
    var_df = pd.DataFrame()
    var_df['Columns'] =  x.columns
    var_df['value'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
    return var_df
```

---

**Cell 75** — VIF Subset
```python
dataset2 = dataset[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'PaperlessBilling','MonthlyCharges','TotalCharges']]
```

---

**Cell 76** — VIF Calculation
```python
calc_vif(dataset2)
```
*Output:*
```
   Columns          value
0  gender           1.921286
1  SeniorCitizen    1.327766
2  Partner          2.815272
3  Dependents       1.921208
4  tenure          10.549667
5  PhoneService     7.976386
6  PaperlessBilling 2.814160
7  MonthlyCharges  13.988649
8  TotalCharges    12.570269
```

---

**Cell 78** — Scatter: Monthly vs Total Charges
```python
dataset2[['MonthlyCharges', 'TotalCharges']].plot.scatter(figsize = (15, 10), x='MonthlyCharges', y='TotalCharges', color='#ec838a')
plt.title('Collinearity of Monthly Charges and Total Charges \n', ...)
```

---

**Cell 80** — Drop TotalCharges
```python
dataset2 = dataset2.drop(columns = "TotalCharges")
dataset2 = dataset[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges']]
calc_vif(dataset2)
dataset = dataset.drop(columns = "TotalCharges")
```

---

**Cell 83** — Preserve customerID
```python
identity =  dataset['customerID']
```

---

**Cell 84** — Drop customerID from dataset
```python
dataset.drop(columns = ['customerID'], inplace = True)
```

---

**Cell 85** — Empty cat_cols init
```python
cat_cols = []
```

---

**Cell 86** — Find multi-category columns
```python
for col in dataset.columns:
    if dataset[col].dtypes =='object':
        if len(dataset[col].unique())>=2:
            print(col)
```
*Output: MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod*

---

**Cell 87** — One-Hot Encoding (full dataset, before split)
```python
dataset = pd.get_dummies(dataset)
```

---

**Cell 88** — Re-attach identity
```python
dataset = pd.concat([identity, dataset], axis =1)
```

---

**Cell 89** — Separate target
```python
response = dataset['Churn']
dataset = dataset.drop(columns=['Churn'])
```

---

**Cell 91** — Re-import train_test_split (already imported)
```python
from sklearn.model_selection import train_test_split
```

---

**Cell 92** — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(dataset, response, stratify=response, test_size = 0.2)
# NOTE: random_state is commented out — results are NOT reproducible
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
```
*Output:*
```
Number transactions X_train dataset:  (5634, 40)
Number transactions y_train dataset:  (5634,)
Number transactions X_test dataset:   (1409, 40)
Number transactions y_test dataset:   (1409,)
```

---

**Cell 94** — Drop customerID from splits
```python
train_identity = X_train['customerID']
X_train = X_train.drop(columns = ['customerID'])
test_identity = X_test['customerID']
X_test = X_test.drop(columns = ['customerID'])
```

---

**Cell 96** — StandardScaler init
```python
sc_X = StandardScaler()
```

---

**Cell 97** — Fit+Transform train
```python
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_train2.columns = X_train.columns
X_train2.index = X_train.index.values
X_train = X_train2
```

---

**Cell 98** — Transform test only (correct)
```python
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_test2.columns = X_test.columns.values
X_test2.index = X_test.index.values
X_test = X_test2
```

---

**Cell 100** — Logistic Regression (first pass)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("📊 Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
```
*Output:*
```
              precision    recall  f1-score   support
           0       0.84      0.88      0.86      1035
           1       0.62      0.55      0.59       374
    accuracy                           0.79      1409
   macro avg       0.73      0.72      0.72      1409
weighted avg       0.79      0.79      0.79      1409
Confusion Matrix:
 [[909 126]
  [167 207]]
```

---

**Cell 102** — Naive Bayes (first pass)
```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("📊 Naive Bayes Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
```
*Output:*
```
              precision    recall  f1-score   support
           0       0.93      0.62      0.75      1035
           1       0.45      0.87      0.60       374
    accuracy                           0.69      1409
   macro avg       0.69      0.75      0.67      1409
weighted avg       0.80      0.69      0.71      1409
Confusion Matrix:
 [[645 390]
  [ 49 325]]
```

---

**Cell 104** — ⚠️ K-Means (NOT a classifier — misapplied here)
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_test)

X_test_copy = X_test.copy()
X_test_copy['Churn'] = y_test.values
X_test_copy['Cluster'] = clusters

print("📊 Average Churn per Cluster:")
print(X_test_copy.groupby('Cluster')['Churn'].mean())
```

---

**Cell 109** — Logistic Regression (second pass, with F2 Score)
```python
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2.0)
results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1, f2]],
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score'])
results = results.sort_values(["Precision", "Recall", "F2 Score"], ascending = False)
print(results)
```
*Output:*
```
                 Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0  Logistic Regression  0.792051   0.621622  0.553476  0.585573  0.565883
```

---

**Cell 111** — Naive Bayes (second pass, tabular format)
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import pandas as pd

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

acc_nb = accuracy_score(y_test, y_pred_nb)
prec_nb = precision_score(y_test, y_pred_nb)
rec_nb = recall_score(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)
f2_nb = fbeta_score(y_test, y_pred_nb, beta=2.0)

nb_results = pd.DataFrame    # ← SYNTAX ODDITY: constructor split across lines
([['Naive Bayes', acc_nb, prec_nb, rec_nb, f1_nb, f2_nb]],
columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score'])
```
*No output (result not printed).*

---

**Cell 112** — *(Empty code cell)*

---

**Cell 113** — K-Means as Classifier (label-flipping hack)
```python
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
clusters = kmeans.predict(X_test)

if accuracy_score(y_test, clusters) < accuracy_score(y_test, 1 - clusters):
    clusters = 1 - clusters

acc_km = accuracy_score(y_test, clusters)
prec_km = precision_score(y_test, clusters)
rec_km = recall_score(y_test, clusters)
f1_km = f1_score(y_test, clusters)
f2_km = fbeta_score(y_test, clusters, beta=2.0)

km_results = pd.DataFrame([['K-Means Clustering', acc_km, prec_km, rec_km, f1_km, f2_km]],
                          columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score'])
```
*No output.*

---

**Cell 115** — Final Model Comparison Table
```python
final_results = pd.concat([results, nb_results, km_results], ignore_index=True)
final_results = final_results.sort_values(by=['Precision', 'Recall', 'F2 Score'], ascending=False)
print(final_results)
```
*Output:*
```
                 Model  Accuracy  Precision    Recall  F1 Score  F2 Score
0  Logistic Regression  0.792051   0.621622  0.553476  0.585573  0.565883
1          Naive Bayes  0.688432   0.454545  0.868984  0.596878  0.734962
2   K-Means Clustering  0.546487   0.065574  0.053476  0.058910  0.055525
```

---

### 3.4 Markdown Cells — Completeness Note

Markdown cells are present (44 total) but **many are partial/truncated**. Several look like they were created by cutting off the first character of section headings (e.g., `"mport relevant libraries"`, `"klearn modules"`, `"tack bar chart"`). Cells 101, 103, 105, 110, 114, 116 are empty markdown cells providing no commentary. The EDA narrative in cell 81 is the only meaningful multi-sentence summary block.

---

## 4. `.py` FILES

**None found.** There are no standalone Python script files in this project.

---

## 5. OTHER FILES

### `customer_churn_data.csv` (954.6 KB)
- **Rows:** 7,043 data rows + 1 header = 7,044 lines
- **Columns (21):**
  ```
  customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
  TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
  PaymentMethod, MonthlyCharges, TotalCharges, Churn
  ```
- **First 5 rows:**
  ```
  7590-VHVEG, Female, 0, Yes, No, 1, No, No phone service, DSL, No, Yes, No, No, No, No, Month-to-month, Yes, Electronic check, 29.85, 29.85, No
  5575-GNVDE, Male, 0, No, No, 34, Yes, No, DSL, Yes, No, Yes, No, No, No, One year, No, Mailed check, 56.95, 1889.5, No
  3668-QPYBK, Male, 0, No, No, 2, Yes, No, DSL, Yes, Yes, No, No, No, No, Month-to-month, Yes, Mailed check, 53.85, 108.15, Yes
  7795-CFOCW, Male, 0, No, No, 45, No, No phone service, DSL, Yes, No, Yes, Yes, No, No, One year, No, Bank transfer (automatic), 42.3, 1840.75, No
  ```
- **Target distribution:** Churn=No: 5,174 (73.5%) | Churn=Yes: 1,869 (26.5%)
- **Known quirk:** `TotalCharges` is stored as string/object type; 11 rows have blank string values that become NaN after `pd.to_numeric(errors='coerce')`.
- **Size:** Safe to push to GitHub (< 50 MB).

### `Telecom Customer Churn Prediction Final.html` (1.25 MB)
- Jupyter notebook exported to HTML — a static render of the notebook with all outputs embedded. Redundant if the `.ipynb` is in the repo, but useful as a preview. Large image outputs (matplotlib figures as base64) are why this file is 2× the notebook size.

### `Telecom-Customer-Churn-Prediction.pptx` (3.23 MB)
- PowerPoint presentation. Binary file — cannot read column names. Size is safe to push. This is likely a project walkthrough/slide deck.

### `Telecom_Churn_Prediction_Report.pdf` (0.44 MB)
- PDF report. Binary file — cannot read inline. Safe to push.

### Model pickle/joblib files
- **None found.** Despite `joblib` being imported and the workflow plan (cell 117, step 15) mentioning model deployment, no `.pkl`, `.joblib`, or `.h5` files exist.

---

## 6. EXISTING README

**No README file exists** anywhere in the project tree. Neither `README.md`, `README.txt`, nor `README.rst` is present at any level.

---

## 7. MISSING STANDARD FILES CHECK

| File | Present? | Notes |
|---|---|---|
| `README.md` | **ABSENT** | No readme at all |
| `.gitignore` | **ABSENT** | Will need one before first push (especially for `.ipynb_checkpoints/`, OneDrive cache files) |
| `requirements.txt` | **ABSENT** | No dependency spec whatsoever |
| `LICENSE` | **ABSENT** | No license file |
| `environment.yml` | **ABSENT** | — |
| `Pipfile` | **ABSENT** | — |
| `setup.py` / `pyproject.toml` | **ABSENT** | Not required for a notebook project but worth noting |

---

## 8. MODEL & METRICS INVENTORY

### 8.1 Models Imported vs. Actually Trained

| Model | Imported? | Actually Fit? | Notes |
|---|---|---|---|
| `LogisticRegression` | Yes (cells 6, 8, 100, 109) | **Yes** (cells 100, 109 — twice) | Primary classifier |
| `GaussianNB` | Yes (cells 6, 102, 111) | **Yes** (cells 102, 111 — twice) | Secondary |
| `KMeans` | Yes (cells 104, 113) | **Yes** (cells 104, 113) | ⚠️ Clustering algorithm misused as classifier |
| `SVC` | Yes (cell 6) | **No** | Imported, never trained |
| `KNeighborsClassifier` | Yes (cells 4, 6) | **No** | Imported twice, never trained |
| `DecisionTreeClassifier` | Yes (cell 6) | **No** | Imported, never trained |
| `RandomForestClassifier` | Yes (cell 6) | **No** | Imported, never trained |
| `XGBClassifier` | Yes (cell 6) | **No** | Imported, never trained |
| `LinearDiscriminantAnalysis` | Yes (cells 6, 8) | **No** | Imported, never trained |
| `GridSearchCV` | Yes (cell 8) | **No** | Imported for planned tuning, never used |
| `cross_val_score` / `KFold` | Yes (cell 8) | **No** | Imported, never used |

### 8.2 Evaluation Metrics from Stored Outputs

**Logistic Regression** (evaluated twice — results consistent):
```
Accuracy:  0.792
Precision: 0.622  (churn class)
Recall:    0.553  (churn class)
F1 Score:  0.586  (churn class)
F2 Score:  0.566
Confusion Matrix: [[909, 126], [167, 207]]
  True Negatives:  909 | False Positives: 126
  False Negatives: 167 | True Positives:  207
```

**Naive Bayes** (evaluated twice — results consistent):
```
Accuracy:  0.688
Precision: 0.455  (churn class)
Recall:    0.869  (churn class)
F1 Score:  0.597  (churn class)
F2 Score:  0.735
Confusion Matrix: [[645, 390], [49, 325]]
```

**K-Means** (used as pseudo-classifier):
```
Accuracy:  0.546
Precision: 0.066
Recall:    0.054
F1 Score:  0.059
F2 Score:  0.056
```
*(Near-random performance — expected, as K-Means is unsupervised and inappropriate here.)*

**Metrics NOT computed despite being imported:**
- ROC-AUC score (`roc_auc_score` imported, never called)
- ROC curve plot (`roc_curve` imported twice, never called)
- Average precision score (imported, never called)
- Log loss (imported, never called)
- Precision-recall curve (imported, never called)

### 8.3 Class Imbalance Handling

- **Dataset is imbalanced:** 73.5% No-churn vs. 26.5% Churn
- **Acknowledged:** Yes — cell 81 explicitly notes "The dataset is imbalanced with the majority of customers being active."
- **Stratified split:** Yes — `stratify=response` in `train_test_split` (correct, maintains class ratio in both splits)
- **SMOTE / oversampling:** **NOT used**
- **`class_weight='balanced'`:** **NOT used** in LogisticRegression or any other model
- **Undersampling:** **NOT used**

**Impact:** Without class_weight or SMOTE, models are biased toward the majority class. This is visible — Logistic Regression recall for churn is only 0.553 (misses nearly half of churners). This is the single biggest modeling gap.

### 8.4 Train/Test Split

- **Method:** `train_test_split` from scikit-learn
- **Ratio:** 80% train / 20% test (5,634 train | 1,409 test)
- **Stratified:** Yes (`stratify=response`)
- **`random_state`:** **NOT SET** — the parameter is commented out in the code (`# random_state = 0)` appears as a comment). Results are non-reproducible across runs.
- **No k-fold cross-validation** (imported but never used).

---

## 9. CODE QUALITY OBSERVATIONS

### 9.1 Hardcoded / Path Issues
- **Cell 12:** `dataset = pd.read_csv('customer_churn_data.csv')` — relative path. Works only if Jupyter kernel CWD matches the file location. Fragile for anyone cloning the repo to a different directory. No `os.path` or `pathlib` usage.

### 9.2 Unused Imports / Dead Code
- `SimpleImputer` — imported, never used (manual imputation done instead)
- `ColumnTransformer` — imported, never used
- `svm`, `tree`, `linear_model`, `neighbors` (sklearn submodules) — imported, never used directly (specific classes re-imported later)
- `naive_bayes`, `ensemble`, `discriminant_analysis`, `gaussian_process` (submodules) — same
- `LinearDiscriminantAnalysis` — imported, never used
- `XGBClassifier` — imported, never used
- `ShuffleSplit` — imported, never used
- `feature_selection`, `model_selection` (sklearn submodules) — imported, never used
- `log_loss`, `average_precision_score`, `precision_recall_curve`, `auc` — imported, never used
- `random`, `timeit`, `string`, `dateutil.parser.parse` — imported, never used
- `scatter_matrix` — imported, never used
- `joblib` — imported but `joblib.dump()` is never called; no model is saved
- `pylab` — imported, never used
- `cat_cols = []` (cell 85) — defined, never populated or used

### 9.3 Duplicate Code / Cells
- `add_value_labels()` function defined identically in cells 49, 55, and 61 — three definitions, no reason to repeat
- `from sklearn.preprocessing import OneHotEncoder` — imported in both cell 4 and cell 6
- `from sklearn.neighbors import KNeighborsClassifier` — imported in both cell 4 and cell 6
- `import matplotlib.pyplot as plt` — imported in cell 2 and cell 8
- `from sklearn.metrics import roc_curve` — imported in cell 8 and again somewhere later
- `from sklearn.model_selection import train_test_split` — imported in cell 4 and re-imported in cell 91
- `import matplotlib.ticker as mtick` — imported in cell 8 AND re-imported inside cells 59, 63
- LogisticRegression trained twice (cells 100 and 109) with slightly different object names; second run uses `random_state=0` while first does not
- GaussianNB trained twice (cells 102 and 111) — identical results, second produces no printed output
- Empty markdown cells: 101, 103, 105, 110, 114, 116

### 9.4 Data Leakage Risks

| Risk | Verdict |
|---|---|
| `StandardScaler` fit on full dataset before split | **NOT present** — scaler fit on X_train only (cell 97), transform on X_test (cell 98). Correct. |
| LabelEncoder fit on full dataset before split | **PRESENT** — LabelEncoder applied to full `dataset` in cell 41, before `train_test_split` in cell 92. Minor risk for binary columns (only 2 values, mapping is deterministic), but the pattern is incorrect. |
| `pd.get_dummies` applied to full dataset before split | **PRESENT** — cell 87 one-hot encodes the entire dataset before splitting. If test data in production had unseen categories, this would fail silently in a real pipeline. |
| Target column in features | **NOT present** — `Churn` separated into `response` before split (cell 89). |
| `TotalCharges` null-fill using full dataset mean | **PRESENT** — mean computed before split (cell 36). Minor in practice since 11 rows / 7043. |

### 9.5 Deprecated / Anti-Pattern Syntax

| Issue | Location | Severity |
|---|---|---|
| `get_ipython().magic('clear')` / `.magic('reset -f')` | Cell 0 | Low — DeprecationWarning, works but messy |
| `np.bool` (removed in NumPy 1.24) | Cell 72 | **High** — `AttributeError` on current NumPy; use `bool` instead |
| `\sn.set(style="white")` — stray leading backslash | Cell 72 | Medium — this line is silently treated as a string expression, `sn.set()` is never called |
| `pd.get_dummies(dataset)` without `dtype=bool` or `drop_first=True` | Cell 87 | Medium — omitting `drop_first=True` introduces dummy variable trap (perfect multicollinearity) for linear models; also returns `uint8` which may cause dtype warnings in newer pandas |
| No `random_state` in `train_test_split` | Cell 92 | Medium — results change every run, not reproducible |
| `.corrwith()` called on a column that will later be dropped | Cells 68–69 | Low — TotalCharges included in correlation but removed in cell 80; order is fine logically but confusing |

### 9.6 Plot Labeling
- Most plots have titles, x-labels, and y-labels — good practice followed consistently.
- Cell 53 (dummy data plot) has labels but the data is fake.
- Cell 72 (heatmap): `sn.set(style="white")` is silently not called (backslash issue), so default seaborn theme is used.
- Some subplot grids (cell 57 — services bar charts) have subplot titles but no shared y-axis labels.
- No plot has a `plt.savefig()` call — all figures exist only as notebook outputs, not as standalone image files in the repo.

---

## 10. SCOPE GAP CHECK

**Question:** Does this project actually deliver a trained predictive model with clear results, or does it stop at EDA?

**Answer: It reaches modeling, but with significant gaps.**

### What IS present:
- Complete data pipeline: load → clean → encode → feature engineer → split → scale → train → evaluate
- Two functioning classifiers (Logistic Regression, Naive Bayes) with classification reports, F1/F2 scores, and confusion matrices
- EDA is thorough: distributions, churn rates by segment, correlation analysis, VIF, collinearity scatter

### What is MISSING (per the notebook's own stated plan in cell 117):
1. **Steps 8 (incomplete):** The plan says "fit logistic, linear svm, kernel svm, knn, decision tree, random forest" — only logistic and naive bayes were actually fit
2. **Step 11:** K-Fold cross-validation — imported, never used
3. **Step 12:** GridSearchCV hyperparameter tuning — imported, never used
4. **Step 13:** Probability outputs and ranking — never computed
5. **Step 14:** Confusion matrix visualization (heatmap style) — only text confusion matrices, no visual
6. **Step 15:** Model serialization for deployment (`joblib.dump`) — `joblib` imported but dump never called; **no `.pkl` file saved**
7. **ROC-AUC / ROC curve:** Imported, never computed or plotted — a standard expectation for churn prediction
8. **K-Means as classifier:** Architecturally wrong — K-Means is unsupervised clustering, not a churn predictor. Its inclusion inflates the "model count" while its 0.066 precision makes the comparison table misleading

### Verdict:
The notebook goes beyond EDA and produces working trained models, but it is **incomplete as a prediction system**. The best-performing model (Logistic Regression at 79.2% accuracy, 0.622 precision, 0.553 recall on churn) is a reasonable baseline, but:
- No model is saved/serialized
- No ROC-AUC is reported
- No hyperparameter tuning was done
- The comparison table includes K-Means which is not a valid churn predictor

**Portfolio presentation impact:** This is a solid EDA-to-baseline-model project. As-is, it demonstrates ML fundamentals well, but reviewers will notice the absent ROC curve, missing model export, and the fact that cell 117 lists unfinished steps — suggesting the project was planned to go further than it did.

---

## Summary of Critical Actions Before Pushing

| Priority | Action |
|---|---|
| P0 | `git init` in project root, add GitHub remote, create initial commit |
| P0 | Create `README.md` (currently absent entirely) |
| P0 | Create `.gitignore` (no git tracking yet; need to exclude `.ipynb_checkpoints/`, OneDrive sync artifacts) |
| P1 | Restructure: move files to repo root rather than inside a subfolder |
| P1 | Create `requirements.txt` (no environment spec exists) |
| P1 | Fix `np.bool` → `bool` in cell 72 (breaks on NumPy ≥ 1.24) |
| P1 | Fix stray `\` before `sn.set(...)` in cell 72 |
| P1 | Remove or replace dummy data in cell 53 |
| P2 | Set `random_state` in `train_test_split` for reproducibility |
| P2 | Remove 3 duplicate `add_value_labels` definitions → keep one |
| P2 | Remove ~15 unused imports |
| P2 | Add ROC-AUC computation and ROC curve plot |
| P2 | Add `class_weight='balanced'` to LogisticRegression to address imbalance properly |
| P3 | Remove or relabel K-Means from classifier comparison |
| P3 | Add `joblib.dump` to serialize the best model |
| P3 | Add `random_state=42` to all model instantiations |
