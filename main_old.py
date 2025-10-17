import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Loding the dataset

housing = pd.read_csv("housing.csv")

# 2. Creating a Stratified Test Set

housing['income_cat'] = pd.cut(housing['median_income'],bins = [0.0,1.5,3,4.5,6.0,np.inf], labels = [1,2,3,4,5]) #np.inf -> np.infinity
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index,test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index].drop('income_cat', axis = 1)
    strat_test_set = housing.iloc[test_index].drop('income_cat',axis = 1)

# split.split(X, y) requires:

# X → features (here, entire housing dataframe)

# y → the category column for stratification (here, housing['income_cat'])

# This returns indices of training and test sets.

# train_index → indices of rows for training set

# test_index → indices of rows for test set

## Working on copy of training data

housing = strat_train_set.copy()

# 3. Separating features and labels

housing_labels = housing['median_house_value'].copy()
housing_features = housing.drop('median_house_value', axis = 1)

print(housing_labels,housing_features)

# 4. Listing Numerical and Caategorical Values

num_attributes = housing_features.drop('ocean_proximity', axis = 1).columns.tolist()
cat_attributes = ['ocean_proximity']

# 5 Making The Pipeline
# For Numerical Columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

#For Categorical Columns

cat_pipeline = Pipeline([
    ('OneHot', OneHotEncoder(handle_unknown = 'ignore'))
])

# the parameter handle_unknown='ignore' tells the OneHotEncoder how to deal with categories in the test data that were not seen in the training data.

# Specifically:

# Normally, if the test data contains a category that the encoder never saw during training, it will raise an error.

# With handle_unknown='ignore', the encoder will skip that unknown category and encode it as all zeros in the one-hot vector instead of throwing an error.

#Constructing The Full Pipeline

full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_attributes),
    ('cat',cat_pipeline,cat_attributes)
])

# 6. Transforming The Data

housing_prepared = full_pipeline.fit_transform(housing_features)
print(housing_prepared)
print(housing_prepared.shape)

# 7. Training The Model

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels,lin_preds)

print(f'The Root Mean Squared Error For Linear Regression Is {lin_rmse}')

#Decision Tree Model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_labels,dec_preds)

print(f'The Root Mean Squared Error For Decision Tree Is {dec_rmse}')

#Random Forest Model

random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared,housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
random_forest_rmse = root_mean_squared_error(housing_labels,random_forest_preds)

print(f'The Root Mean Squared Error For Random Forest Is {random_forest_rmse}')

## Training RMSE only shows how well the model fits the training data.
# It does not tell us how well it will perform on unseen data.
# In fact, the Decision Tree and Random Forest may overfit,leading to very low training error but poor generalization.


# Evaluate Decision Tree with cross-validation
# WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# We use minus (-) to convert it back to positive RMSE.

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_labels, scoring = 'neg_root_mean_squared_error', cv = 10)

# print(f'The Root Mean Squared Error For Linear Regression Is {lin_rmse}')
print(pd.Series(lin_rmses).describe())

#Decision Tree Model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_preds = dec_reg.predict(housing_prepared)


dec_rmses = -cross_val_score(dec_reg,housing_prepared,housing_labels, scoring = 'neg_root_mean_squared_error', cv = 10)

print(pd.Series(dec_rmses).describe())
# print(f'The Root Mean Squared Error For Decision Tree Is {dec_rmses}')
#we got a conclusion after applying cross_validation that earlier decision tree model was overfitting the data 
#beacuse after applying cross validation we got to know that for 10 cross validations decision tree giving average error roughly arounf 69000
#from this we can proove that the best regressor for the modelling would be random forest

#Random Forest Model

random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared,housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
random_forest_rmses = -cross_val_score(random_forest_reg,housing_prepared,housing_labels, scoring = 'neg_root_mean_squared_error', cv = 10)

# print(f'The Root Mean Squared Error For Random Forest Is {random_forest_rmses}')
print(pd.Series(random_forest_rmses).describe())

#Conclusion :- 

#Random Forest Regressor Is Performing well compare to other regressors therefore will gonna use randome forest regressor further
