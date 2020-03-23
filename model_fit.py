import pandas as pd 
from sklearn.ensemble import RandomForestClassifier as RFC 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('df_dum.csv')
df = df.drop(columns = ['Unnamed: 0'], axis = 1)

X = df.loc[:, df.columns != 'responded']
y = df.loc[:, df.columns == 'responded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

os = SMOTE(random_state=0)

# We don't need to do train_test_split here because the data is already split into training and test sets.
# speaking of which, I am afraid we will have to clean the test set as well.

os_data_X, os_data_y = os.fit_sample(X_train, y_train)
print("Ratio of SMOTE oversampled data YES to NO respondents:", len(os_data_X)/len(os_data_y))
print("where 1.0 is a perfect ratio")

# useless
# df_vars=df.columns.values.tolist()
# y=['responded']
# X=[i for i in df_vars if i not in y]

logreg = LogisticRegression(max_iter=5000) # this doesn't coverge nicely so I think we will skip these steps...
rfe = RFE(logreg, 50)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

#have now chosen the most predictive columns, we should now select them from X
#honestly, these don't look right. There are no numeric values in here.

col_names = os_data_X.columns[rfe.support_]
print(col_names)
OS_X = os_data_X[col_names].values
OS_y = os_data_y.values #not as memory efficient but we've been good so far on memory

logit_model = sm.Logit(OS_y, OS_X)
predict = logit_model.fit()
print(predict.summary2())

#now import data and bin the numeric test data accordingly