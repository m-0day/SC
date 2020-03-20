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

# Naive Bayes
# Advantages: This algorithm requires a small amount of training data to estimate the necessary parameters.
# Naive Bayes classifiers are extremely fast compared to more sophisticated methods.
# Disadvantages: Naive Bayes is is known to be a bad estimator.

# Logistic Regression
# Advantages: Logistic regression is designed for this purpose (classification), and is most useful for understanding the influence of several independent variables
# on a single outcome variable.
# Disadvantages: Works only when the predicted variable is binary, assumes all predictors are independent of each other, and assumes data is free of missing values.

# RF 
# Advantages: Reduction in over-fitting and random forest classifier is more accurate than decision trees in most cases.
# Another advantage is that I think RFC allows non-numeric values
# Disadvantages: Slow real time prediction, difficult to implement, and complex algorithm.
# note:
# One thing to remember when we use Random Forest is when you use a categorical feature for training it shouldn't have more than 53 categories. 
# Sometimes RandomForest takes numerical data as categorical. To overcome that make sure to convert all categorical as factors using this command.
# df['col_name'] = df['col_name'].astype('category')

# Import Data
df = pd.read_csv('marketing_training.csv')
# df.show()
# there are not that many rows of data, only about 7400
df_clean = df
df_clean.schooling = df_clean.schooling.replace('unknown', np.NaN)
df_clean.default = df_clean.default.replace('unknown', np.NaN)
# there were very few rows of data remaining after I did a straight dropna() so I set the threshold to 2 NaN's across a row and that didn't drop anything ¯\_(ツ)_/¯
# I think that will not be good enough and we will have to drop about half the data. We will see.
df_clean = df.dropna(axis = 'rows', thresh = 2)

# dummies = pd.get_dummies(df['profession', 'marital', 'schooling', 'housing', 'loan', 'contact'])


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

