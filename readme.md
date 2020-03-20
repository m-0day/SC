# Classification Model to Determine a Likely Customer

## Order of Operations, Generally
[x] Clean Data
[ ] Exploratory Data Analysis and Visualization
[ ] Create and Train Model
[ ] Test Model
[ ] Optimize Parameters
[ ] (Optional) Repeat for other model types and determine best model
[ ] Visualize Results

I conducted some EDA and looked at the types of data. They are mixed categories (strings) and numerics. The perceived output will be a Yes/No on likely customer. 
This binary output leads me to the following model types.

## Model Types
### Naive Bayes
Advantages: This algorithm requires a small amount of training data to estimate the necessary parameters.
Naive Bayes classifiers are extremely fast compared to more sophisticated methods.
Disadvantages: Naive Bayes is is known to be a bad estimator.

### Logistic Regression
Advantages: Logistic regression is designed for this purpose (classification), and is most useful for understanding the influence of several independent variables
on a single outcome variable.
Disadvantages: Works only when the predicted variable is binary, assumes all predictors are independent of each other, and assumes data is free of missing values.

### Random Forest 
Advantages: Reduction in over-fitting and random forest classifier is more accurate than decision trees in most cases.
Another advantage is that I think RFC allows non-numeric values
Disadvantages: Slow real time prediction, difficult to implement, and complex algorithm.
note:
One thing to remember when we use Random Forest is when you use a categorical feature for training it shouldn't have more than 53 categories. 
Sometimes RandomForest takes numerical data as categorical. To overcome that make sure to convert all categorical as factors using this command.
df['col_name'] = df['col_name'].astype('category')

## EDA
Some of the model types listed above assume independence between independent variables. This is not always the case, so we should test for independence with a correlation matrix.
![Correlation Matrix](Corr_Matrix.png)
We can see there are some strong negative correlations between contacts.
