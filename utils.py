import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):    
    def __init__(self, add_bedrooms_per_room = True):       
        self.add_bedrooms_per_room = add_bedrooms_per_room    
        
    def fit(self, X, y=None):        
        return self 
    
    def transform(self, X, y=None):      
        
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]        
        population_per_household = X[:, population_ix] / X[:, households_ix]  
        
        if self.add_bedrooms_per_room:            
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]    
            
            return np.c_[X, rooms_per_household, population_per_household,                         
                         bedrooms_per_room]        
        else:            
            
            return np.c_[X, rooms_per_household, population_per_household]
            
        
def stratified_shuffle(data, col_name):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

    for train_index, test_index in split.split(data, data[col_name]):    
        strat_train_set = data.loc[train_index]    
        strat_test_set = data.loc[test_index]
    
    return strat_train_set, strat_test_set