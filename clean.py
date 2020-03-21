### Developed by Chris McLean ###
# Mar 2020
# For more hobby projects of varying degrees of completion see my github
# https://www.github.com/m-0day

import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Import Data
df = pd.read_csv('marketing_training.csv')
# df.show()
# there are not that many rows of data, only about 7400
df_clean = df
df_clean = df_clean.replace('unknown', np.NaN)

df_clean.isna().sum()

df_clean = df.dropna(axis = 'rows', thresh = 2)
df_clean = df_clean.drop(['pmonths'], axis = 1) #pmonths is just pdays divided by 30 or something, pdays are all 22 or less unless they are 999
df_clean['p_last_mon'] = df_clean.pdays < 30
# dummies = pd.get_dummies(df['profession', 'marital', 'schooling', 'housing', 'loan', 'contact'])
df_clean['custAge'].fillna(df_clean['custAge'].mean(), inplace = True)

#### Exploratory Data Analysis ####
# To choose the right model we will want to have determined if things were independent, so we should plot the correlation matrix

corr = df_clean.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title('Correlation Matrix for Numeric Independent Vars')

ax.title('Correlation Matrix for Numeric Independent Vars')
plt.show()

