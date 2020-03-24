import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('df_dum.csv')
df = df.drop(columns = ['Unnamed: 0'], axis = 1)

X = df.loc[:, df.columns != 'responded']
y = df.loc[:, df.columns == 'responded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

os = SMOTE(random_state=0)

# We don't need to do train_test_split here because the data is already split into training and test sets.
# speaking of which, I am afraid we will have to clean the test set as well.

#Update we will have to clean the test data AND we will have to do a train test split on the training data because this is a blind test/prediction

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

# drop the large p-values of x2 and x12 (first index = 1)
l_col_names = list(col_names)
del l_col_names[1], l_col_names[11], l_col_names[41], l_col_names[39]

#print l_col_names to terminal, the lazy way
l_col_names

os_data_X = os_data_X[l_col_names]

OS_X = os_data_X.values

#### Fit the model and test it using .predict method
X_train, X_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
logreg.score(X_test, y_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
cf = metrics.classification_report(y_test, y_pred)
print(cf)

#### Fit the model and test it using .predict method
X_train, X_test, y_train, y_test = train_test_split(os_data_X, os_data_y, test_size=0.4, random_state=0)
rand_forest = RandomForestClassifier(bootstrap = True)
rand_forest.fit(X_train.values, y_train.values.ravel())
y_pred = rand_forest.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

#finally import the blind test data, clean and make prediction
#in theory I SHOULD be able to just import and filter by column names and then make predictions based on the remaining cols
#update: of course that did not work at first. 
dft = pd.read_csv('dft_dum.csv')

# predictor_list = col_names.values
i = 0
for pred in l_col_names:
    if pred not in dft.columns:
        print('Problem', pred)
        i = i+1
if (i == 0):
    print('No Problem')
    

Blind_X_test = dft[l_col_names]
Blind_y_pred = rand_forest.predict(Blind_X_test)

np.savetxt("Blind Prediction.csv", Blind_y_pred, delimiter=",", fmt='%s')
