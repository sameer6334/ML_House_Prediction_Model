import os
import joblib
import numpy as np
import pandas as pd

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

#We use .pkl files to
#Save trained ML models: After training a model, you don’t need to retrain it every time. You can save it with pickle and load it whenever needed.

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
        # For Numerical Columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ])

    #For Categorical Columns

    cat_pipeline = Pipeline([
        ('OneHot', OneHotEncoder(handle_unknown = 'ignore'))
    ])

    #Constructing The Full Pipeline

    full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_attributes),
        ('cat',cat_pipeline,cat_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #Lets Train The MOdel

    # Loding the dataset

    housing = pd.read_csv("housing.csv")

    # Creating a Stratified Test Set

    housing['income_cat'] = pd.cut(housing['median_income'],bins = [0.0,1.5,3,4.5,6.0,np.inf], labels = [1,2,3,4,5]) #np.inf -> np.infinity
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index,test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.iloc[train_index].drop('income_cat', axis = 1)
        housing.iloc[test_index].drop('income_cat',axis = 1).to_csv("input.csv", index = False)

    # split.split(X, y) requires:

    # X → features (here, entire housing dataframe)

    # y → the category column for stratification (here, housing['income_cat'])

    # This returns indices of training and test sets.

    # train_index → indices of rows for training set

    # test_index → indices of rows for test set

    housing = strat_train_set.copy()

    # 3. Separating features and labels

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis = 1)

    print(housing_labels,housing_features)

    # 4. Listing Numerical and Caategorical Values

    num_attributes = housing_features.drop('ocean_proximity', axis = 1).columns.tolist()
    cat_attributes = ['ocean_proximity']

    #creating a pipeline
    pipeline = build_pipeline(num_attributes,cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state = 42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model, MODEL_FILE)
    #here joblib will dump the object model to the model_file
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained, Congrats!")

else:
    #Lets do inference 
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions
    input_data.to_csv("output.csv", index = False)
    print("Inference is Completed, result saves to output.csv, Enjoy!")

