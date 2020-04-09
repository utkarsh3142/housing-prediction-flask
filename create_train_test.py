import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_shuffle(data, col_name):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

    for train_index, test_index in split.split(data, data[col_name]):    
        strat_train_set = data.loc[train_index]    
        strat_test_set = data.loc[test_index]
        
    for set_ in (strat_train_set, strat_test_set):    
        set_.drop("income_cat", axis=1, inplace=True)
        
    #strat_train_set = strat_train_set.drop("median_house_value", axis=1) 
    
    return strat_train_set, strat_test_set
    
path = './data/housing.txt'
housing = pd.read_csv(path)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
                               
column = "income_cat"
strat_train_set, strat_test_set = stratified_shuffle(housing, column)

# write training and testing files
train_file = "./data/train.txt"
test_file = "./data/test.txt"

strat_train_set.to_csv(train_file,index=False)
strat_test_set.to_csv(test_file,index=False)