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

df_clean['schooling']=np.where((df_clean['schooling'] == 'basic.9y') | (df_clean['schooling'] == 'basic.6y') | (df_clean['schooling'] == 'basic.4y'), 'Basic', df_clean['schooling'])

df_clean = df_clean.replace('unknown', np.NaN) # for some reason I still had some "unknown"

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

ax.set_title('Correlation Matrix for Numeric Independent Vars')
# plt.show()

fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2, 4)
fig.suptitle('Numeric Data Histograms for "Yes" and "No" Responses', fontsize=16)

axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
num_cols = ('custAge', 'pdays', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'pastEmail')
i = 0
for col in num_cols:
    x1 = df_clean[col][df_clean['responded'] == 'yes']
    x2 = df_clean[col][df_clean['responded'] == 'no']
    if i < 7:
        axes[i].hist([x1, x2], bins=10, color = ['g', 'r'], alpha = 0.7, label = ['yes', 'no'])
        # axes[i].hist(x2, bins=10, color = 'r')
        # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        axes[i].set_title(col)
    i = i + 1

fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3)
fig.suptitle('Categorical Bar Charts for "Yes" and "No" Responses', fontsize=16)

axes = [ax0, ax1, ax2, ax3, ax4, ax5]
cat_cols = ('profession', 'marital', 'schooling', 'default', 'housing', 'loan')

i = 0
for col in cat_cols:
    total = df_clean[col].count()
    x1 =  df_clean[col][df_clean['responded'] == 'yes']
    x2 = df_clean[col][df_clean['responded'] == 'no']
    x1 = x1.groupby(x1.values).count()
    x2 = x2.groupby(x2.values).count()
    ind = np.arange(len(x1))
    width = 0.35
    rects1 = axes[i].bar(ind, x1, width, color = 'g')
    rects2 = axes[i].bar(ind+width, x2, width, color = 'r')
    xTickMarks = [str(j) for j in x1.index.values]
    xTickNames = axes[i].set_xticklabels(xTickMarks, rotation = 45)
    # axes[i].hist(x2, bins=10, color = 'r')
    # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    axes[i].set_title(col)
    
    i = i + 1


plt.show()

