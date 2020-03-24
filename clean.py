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

#### Import Train Data ####
df = pd.read_csv('marketing_training.csv')
# df.show()
# there are not that many rows of data, only about 7400


#### Import Test Data ####
dft = pd.read_csv('marketing_test.csv')

def clean_the_data(dirty_df):
    clean_df = dirty_df
        
    clean_df = clean_df.replace('unknown', np.NaN)

    clean_df.isna().sum()

    clean_df = clean_df.dropna(axis = 'rows')
    clean_df = clean_df.drop(['pmonths'], axis = 1) #pmonths is just pdays divided by 30 or something, pdays are all 22 or less unless they are 999
    clean_df['p_last_mon'] = clean_df.pdays < 30
    # dummies = pd.get_dummies(df['profession', 'marital', 'schooling', 'housing', 'loan', 'contact'])
    clean_df['custAge'].fillna(clean_df['custAge'].mean(), inplace = True)
    clean_df.loc[clean_df['pdays'] > 31, 'pdays'] = 32
    clean_df['schooling']=np.where((clean_df['schooling'] == 'basic.9y') | (clean_df['schooling'] == 'basic.6y') | (clean_df['schooling'] == 'basic.4y'), 'Basic', clean_df['schooling'])

    clean_df = clean_df.replace('unknown', np.NaN) # for some reason I still had some "unknown"
    
    return clean_df

def make_numeric_cuts_from_scratch(df, numeric_cols):
    bin_holder = dict()
    for col in num_cols:
        df[col], bins = pd.cut(df[col], bins = 10, retbins=True)
        df[col] = df[col].astype('category')
        bin_holder[col] = bins
    return df, bin_holder

def make_numeric_cuts_with_bins(df, numeric_cols, bin_holder):
    for col in num_cols:
        bins = bin_holder[col]
        intind = pd.IntervalIndex.from_breaks(bins)
        df[col] = pd.cut(df[col], bins = intind).astype('category')
        # df[col] = 
    #need a bin holder that has the name of the corresponding numeric column and the bin values, always use 10 bins
    return df

df_clean = clean_the_data(df)
df_clean['responded'] = df_clean['responded'].apply(lambda x: True if x == 'yes' else False)

dft_clean = clean_the_data(dft)
dft_clean = dft_clean.drop('Unnamed: 0', axis = 1)

# df_clean = df_clean.replace('unknown', np.NaN)

# df_clean.isna().sum()

# df_clean = df_clean.dropna(axis = 'rows')
# df_clean = df_clean.drop(['pmonths'], axis = 1) #pmonths is just pdays divided by 30 or something, pdays are all 22 or less unless they are 999
# df_clean['p_last_mon'] = df_clean.pdays < 30
# # dummies = pd.get_dummies(df['profession', 'marital', 'schooling', 'housing', 'loan', 'contact'])
# df_clean['custAge'].fillna(df_clean['custAge'].mean(), inplace = True)

# df_clean['schooling']=np.where((df_clean['schooling'] == 'basic.9y') | (df_clean['schooling'] == 'basic.6y') | (df_clean['schooling'] == 'basic.4y'), 'Basic', df_clean['schooling'])

# df_clean = df_clean.replace('unknown', np.NaN) # for some reason I still had some "unknown"
# df_clean['responded'] = df_clean['responded']
# df_clean['responded'] = df['responded'].apply(lambda x: True if x == 'yes' else False)

# df.show()
# there are not that many rows of data, only about 7400
# dft_clean = dft
# dft_clean = dft_clean.replace('unknown', np.NaN)

# dft_clean.isna().sum()

# dft_clean = df.dropna(axis = 'rows')
# dft_clean = dft_clean.drop(['pmonths'], axis = 1) #pmonths is just pdays divided by 30 or something, pdays are all 22 or less unless they are 999
# dft_clean['p_last_mon'] = dft_clean.pdays < 30
# # dummies = pd.get_dummies(df['profession', 'marital', 'schooling', 'housing', 'loan', 'contact'])
# dft_clean['custAge'].fillna(dft_clean['custAge'].mean(), inplace = True)

# dft_clean['schooling']=np.where((dft_clean['schooling'] == 'basic.9y') | (dft_clean['schooling'] == 'basic.6y') | (dft_clean['schooling'] == 'basic.4y'), 'Basic', dft_clean['schooling'])

# dft_clean = dft_clean.replace('unknown', np.NaN) # for some reason I still had some "unknown"
# dft_clean['responded'] = dft_clean['responded']
# dft_clean['responded'] = df['responded'].apply(lambda x: True if x == 'yes' else False)
# dft = 


#### Exploratory Data Analysis ####
# To choose the right model we will want to have determined if things were independent, so we should plot the correlation matrix

# corr = df_clean.corr()

# mask = np.triu(np.ones_like(corr, dtype=np.bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title('Correlation Matrix for Numeric Independent Vars')

# ax.set_title('Correlation Matrix for Numeric Independent Vars')
# # plt.show()

# fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9)) = plt.subplots(2, 5)
# fig.suptitle('Numeric Data Histograms for "Yes" and "No" Responses', fontsize=16)

# axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
num_cols = ('custAge', 'pdays', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'pastEmail', 'emp.var.rate', 'campaign', 'previous') #there are so many columns
# i = 0
# for col in num_cols:
#     x1 = df_clean[col][df_clean['responded'] == True]
#     x2 = df_clean[col][df_clean['responded'] == False]
#     axes[i].hist([x1, x2], bins=10, color = ['g', 'r'], alpha = 0.5, label = ['yes', 'no'], density = True)
#     axes[i].set_title(col)
#     i = i + 1

# fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3)
# fig.suptitle('Categorical Bar Charts for "Yes" and "No" Responses', fontsize=16)

# axes = [ax0, ax1, ax2, ax3, ax4, ax5]
cat_cols = ('profession', 'marital', 'schooling', 'default', 'housing', 'loan')

# i = 0
# for col in cat_cols:
#     x1 =  df_clean[col][df_clean['responded'] == True]
#     x2 = df_clean[col][df_clean['responded'] == False]
#     if i == 2:
#         x2 = df_clean[col][(df_clean['responded'] == False) & (df_clean['schooling'] != 'illiterate')] #to avoid plotting issues, drop the one illeterate value (from the plot only)
#     x1 = x1.groupby(x1.values).count()
#     if i == 3:
#         x1['yes'] = 0

#     x2 = x2.groupby(x2.values).count()
#     x1 = x1/x1.sum() #normalize for viz
#     x2 = x2/x2.sum() 
#     ind = np.arange(len(x1))
#     width = 0.35
#     rects1 = axes[i].bar(ind, x1, width, color = 'g', alpha = 0.5)
#     rects2 = axes[i].bar(ind+width, x2, width, color = 'r', alpha = 0.5)
#     labels = [str(j) for j in x1.index.values]
#     axes[i].set_xticks(ind)

#     xTickNames = axes[i].set_xticklabels(labels, rotation = 45)
#     axes[i].xaxis.set_tick_params(labelsize=8)
#     axes[i].set_title(col)
        
#     i = i + 1


# plt.show()

#### Data is Explored, now for logistic regression, create dummy variables. Random Forest will need numerics to be listed as categoricals ####
target_col = ['responded']
df_clean = df_clean.reset_index(drop = True)

#### Be uber careful when mixing data. You should never do this but I had to in order to get consistent binning. ####

safe_df_clean_len = len(df_clean)
safe_dft_clean_len = len(dft_clean)
holder_df = df_clean.drop(['responded'], axis = 1)
df_unclean = pd.concat([holder_df, dft_clean])

df_unclean, bin_holder = make_numeric_cuts_from_scratch(df_unclean, num_cols)

df_clean = make_numeric_cuts_with_bins(df_clean, num_cols, bin_holder) #df_clean still has 'responded'
dft_clean = make_numeric_cuts_with_bins(dft_clean, num_cols, bin_holder)

test_df_clean_len = len(df_clean)
test_dft_clean_len = len(dft_clean)

print("Data is safe after binning?")
if (((safe_df_clean_len - test_df_clean_len) == 0) & ((safe_dft_clean_len- test_dft_clean_len) == 0)):
    print("safe")
else:
    print("FAIL")

#### Ok, I think we are out of the woods there
train_dummy_cols = list(set(df_clean.columns) - set(target_col))
df_dum = pd.get_dummies(df_clean, columns= train_dummy_cols)

for col in df_dum.columns:
    print(col)

df_dum['responded'] = df_clean['responded']

df_clean.to_csv('df_clean.csv')

df_dum.to_csv('df_dum.csv', index = False) #I was getting a lot of dummies I think they are indices

### repeat for test set

test_dummy_cols = list(set(dft_clean.columns))
dft_dum = pd.get_dummies(dft_clean, columns= test_dummy_cols)
# dft_dum['responded'] = dft_clean['responded']

dft_clean.to_csv('dft_clean.csv')

dft_dum.to_csv('dft_dum.csv', index= False)

# for col in dft_dum.columns: 
#     if (col in df_dum.columns): 
#         print('Pass', col)
#     else:
#         print('Fail', col)

#     #
# #test if I made the dummy columns right
# for col in dft.columns: 
#     if (col in df.columns): 
#         print('Pass', col)
#     else:
#         print('Fail', col)

    #We discover we have lots of blind test data that is not covered by the categorization of the training data.
    # ['campaign_(0.961, 4.9]', 'nr.employed_(5201.65, 5228.1]', 'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',
    #  'day_of_week_tue', 'day_of_week_wed', 'cons.price.idx_(92.458, 92.714]', 'cons.price.idx_(92.971, 93.227]', 
    # 'pdays_(3.2, 6.4]', 'pdays_(16.0, 19.2]', 'month_may', 'housin'profession_student', 'profession_technician', 
    # 'profession_unemployed', 'schooling_Basic', 'schooling_high.school', 'schooling_professional.cou', 
    # 'custAge_(56.0, 63.6]', 'custAge_(63.6, 71.2]', 'custAge_(78.8, 86.4]']
