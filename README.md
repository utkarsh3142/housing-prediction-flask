# housing-prediction-flask
Housing Prediction RESTful API using flask

This is a simple flask application to predict housing prices based on Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems Book by Aurelien Geron tutorial.

RUNNING:

export FLASK_APP=app.py
python -m flask run --host=0.0.0.0 --port=8080 

(default port is 5000)

USAGE: 

URL -> /predict

Methods -> POST or GET 

Data Params -> longitude <DECIMAL>,
		latitude <DECIMAL>, 
		housing_median_age <DECIMAL>, 
		total_rooms <DECIMAL>, 
		total_bedrooms <DECIMAL>, 
		population <DECIMAL>, 
		households <DECIMAL>, 
		median_income <DECIMAL>, 
		ocean_proximity <STRING>
	
Success Response -> code 200, content:{"results":{"house_price":499094}} 

Example Usage - 

curl -X POST http://127.0.0.1:8080/predict -H "Content-Type: application/json" -d '{"longitude":-118.39,"latitude":34.12,"housing_median_age":29.0,"total_rooms":6447.0,"total_bedrooms":1012.0,"population":2184.0,"households":960.0,"median_income":8.2816,"ocean_proximity":"<1H OCEAN"}'

Result:

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   236  100    35  100   201   1944  11166 --:--:-- --:--:-- --:--:-- 13882{"results":{"house_price":499094}}

