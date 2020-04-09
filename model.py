import pandas as pd
import numpy as np
from utils import CombinedAttributesAdder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy import stats 
import pickle

def confidence_interval(predictions, true_labels):
    confidence = 0.95 
    squared_errors = (predictions - true_labels) ** 2
    ci = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))
    return ci

def load_housing_data(file_path):
    data = pd.read_csv(file_path)
    train_data = data.drop("median_house_value", axis=1)
    train_label = data["median_house_value"].copy()
    return train_data, train_label

def data_processing(data, num_attribs, cat_attribs):
    
    num_pipeline = Pipeline([        
        ('imputer', SimpleImputer(strategy="median")),        
        ('attribs_adder', CombinedAttributesAdder()),        
        ('std_scaler', StandardScaler()),    
    ])
    
    full_pipeline = ColumnTransformer([        
        ("num", num_pipeline, num_attribs),        
        ("cat", OneHotEncoder(), cat_attribs),    
    ])
    
    data_prepared = full_pipeline.fit_transform(data)
    
    return full_pipeline, data_prepared

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def train_model(data, label, n_estimators=100, max_features=None ,grid_search=False, bootstrap=False):

    if grid_search:
        param_grid = [    
        {'n_estimators': n_estimators, 'max_features': max_features},    
        {'bootstrap': [bootstrap], 'n_estimators': n_estimators, 'max_features': max_features},  
        ]
    
        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(data, label)
        
        print("Grid search best parameters : ", grid_search.best_params_ )
        print("Grid search best estimator : ", grid_search.best_estimator_ )
        
        cvres = grid_search.cv_results_ 
        
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
            
        return grid_search.best_estimator_
            
    else:
        forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        forest_reg.fit(data, label)
        predictions = forest_reg.predict(data)
        forest_mse = mean_squared_error(label, predictions)
        forest_rmse = np.sqrt(forest_mse)
        print("Root mean square error : ", forest_rmse)
        
        forest_scores = cross_val_score(forest_reg, data, label,
                                scoring="neg_mean_squared_error", cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        display_scores(forest_rmse_scores)
        
        return forest_reg

def save_pickle(file, path):

    pickle.dump(file, open(path, 'wb'))

    
if __name__ == '__main__':
    
    print("Starting to load training file")
    path = './data/train.txt'
    housing, housing_labels = load_housing_data(file_path=path)
    print("Input training dataframe --> \n", housing.head())
    
    num_attribs = ['longitude',
             'latitude',
             'housing_median_age',
             'total_rooms',
             'total_bedrooms',
             'population',
             'households',
             'median_income'] 
    cat_attribs = ["ocean_proximity"]
    
    print("Starting to process training data")
    full_pipeline, housing_prep = data_processing(housing,num_attribs=num_attribs,cat_attribs=cat_attribs)
    print("Training data processing complete")
    
    print("Starting training Random Forest Regressor on the training data")
    model = train_model(housing_prep, housing_labels, n_estimators=[3, 10, 30], max_features=[2, 4, 6, 8] ,grid_search=True, bootstrap=False)
    
    #model = train_model(housing_prep, housing_labels)
    print("Completed training Random Forest Regressor on the training data")
    # Save pipeline 
    
    print("Saving model and pipeline to pickle")
    pipeline_path = "./model/housing_pipeline.pkl"
    save_pickle(full_pipeline, pipeline_path)
    
    # Save model 
    model_path = "./model/housing_model.pkl"
    save_pickle(model, model_path)
    print("Completed saving. All operations done.")
    
    