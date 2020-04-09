import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from utils import CombinedAttributesAdder

model = pickle.load(open('./model/housing_model.pkl','rb'))
pipeline = pickle.load(open('./model/housing_pipeline.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/predict', methods=['POST','GET'])
def predict():
    # get data
    data = request.get_json(force=True)
    
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    
    print("----------->", data)
    housing = pd.DataFrame.from_dict(data)

    #print("----------->", housing.head())
	# run the dataframe through data processing pipeline
    housing_prepared = pipeline.transform(housing) 

    
    print("----------->", housing_prepared)
    # predictions
    result = model.predict(housing_prepared)

    # send back to browser
    output = {'house_price': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
# load model
    
    app.run(port = 5000, debug=True)