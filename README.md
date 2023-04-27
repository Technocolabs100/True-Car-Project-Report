
# Car Prediction

A car price prediction model is a stastical model that predicts the price of a car based on its features.The Features can include the car's make,model,year,mileage,condition,state,city.They can also be used by consumers to get an idea of how much a car is worth.

## Understanding the Dataset
>The Dataset we are working on is a combination of **car_listing** for sales and car that are **sold** to the consumers.

->car_listing dataset contain the names of the cars and model no,make,vin number,mileage ,year bought that are ready for resale.It contain a row of (1216250, 9).

->car dataset cnatain the car list that are sold to the user by the car manufacturer.It contain a row of (852122, 8).

->The target values for our model prediction is yo predict the **price** f the car.

#**EDA**
**Introduction:**
-**merged_data*** data comprises of 545623 rows and 8 columns.
-Dataset comprises of :-
->price - numerical values dtype=int
->make - categorical value dtype = obejct
->model - categorical value dtype = obejct
->VIN number - categorical value dtype = obejct
->year - numerical values dtype=obejct
->state -categorical value dtype = obejct
->city -categorical value dtype = obejct
->mileage - numerical values dtype=float

**Univariate Analysis:**

Plotted hostogram to see the disribution of data most of the curve are left screwed 
-Total no of cities 2553
 CITITES LIKE :
> Houston   :              8810
> San Antonio:             5115
> Louisville  :            4392
> Jacksonville :           4004    
> Orlando       :          3743               

>ARE THE LARGEST  HUB FOR CAR DEALING

-total numbe of car's maker: 58
-total numebr of car's model : 2579
-total numebr of state : 59

**Correlation Plot of Numerical Variables:**
All the contnuous variables are positively correlated with each other with Correlation coeffecient of 0.42 expect price and mleage having a correlation of -0.77 and mileage and year having a correlation of -0.44.

## Model Building

#### Metrics considered for Model Evaluation


->**RMSE**: Root Mean Squared Error is a measurement of the difference between predicted and actual values. It is calculated by taking the square root of the average of the squared differences between the predicted and actual values.
->**R2 Score**: R-Squared (or R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. It is calculated by dividing the explained variance by the total variance.
->**MAE**: Mean Absolute Error is a measurement of the difference between predicted and actual values. It is calculated by taking the average of the absolute differences between the predicted and actual values.
->**MSE**: Mean Squared Error is a measurement of the difference between predicted and actual values. It is calculated by taking the average of the squared differences between the predicted and actual values.

#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

#### Random Forest Regression
-Random Forest Regression is a machine learning algorithm used for regression tasks. It is an ensemble of multiple decision trees, with each tree trained on a random subset of features and samples using bagging and feature randomness techniques. The algorithm aims to create a diverse set of trees that make uncorrelated predictions, and then combine their outputs to achieve better prediction accuracy than any single decision tree. Bagging and feature randomness help to reduce overfitting, increase robustness and make the model less sensitive to noisy data.
- **Bagging and Boosting**: In this method of merging the same type of predictions. Boosting is a method of merging different types of predictions. Bagging decreases variance, not bias, and solves over-fitting issues in a model. Boosting decreases bias, not variance.
- **Feature Randomness**:  In a normal decision tree, when it is time to split a node, we consider every possible feature and pick the one that produces the most separation between the observations in the left node vs. those in the right node. In contrast, each tree in a random forest can pick only from a random subset of features. This forces even more variation amongst the trees in the model and ultimately results in lower correlation across trees and more diversification

### Choosing the features
We know from the EDA that all the features are highly correlated and almost follows the same trend.

when we perform linear regression to predict the price of the car we are getting a score of
Model Linear Regression	R2-0.299133	MAE-4131.940412	MSE-8.820599e-01 

-Random forest regression score of 
Random Forest	R2 - 0.299115	MAE-2420.135677	MSE-2.228450e+07

Ridge Regression R2-0.129785	MAE-2420.135677	MSE-1.606177e+08

#Model Building
-First split the dataset into two set training set and test set with a split ratio of **80%-20%**.
-**x_train,x_test,y_train,y_test**

-Removing the columns that are not required for model prediction

-creating two new columns for our car model prediction **Years_Ago** and **Avg_Mileage_Per_Year**

-Using **frequency OneHotEncoding** on the categorical columns and getting the numerical values for the model prediction

-Using **MinMaxScaler** on the dataset to convert all the larger values and bringing them to a scale range which increases the performance of the model.

### PCA transformation
We reduced the 10 features to be only 3.
~~~
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X_train2)
trained = pca.transform(X_train2)
transformed = pca.transform(X_valid2)
~~~.

#Model Fit 
-we pass the train set to our model for prediction and predicted the price of our car.


## Screenshots

![App Screenshot](https://github.com/Ankitb700/Car-Prediction/blob/main/screensort/Screenshot%20(111).png?raw=true/468x300?text=App+Screenshot+Here)

![App Screenshot](https://github.com/Ankitb700/Car-Prediction/blob/main/screensort/Screenshot%20(110).png?/468x300?text=App+Screenshot+Here)

#** Deployment**
# **Streamlit**
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We write a python code for our app using Streamlit; the app asks the user to enter the following data (**news data**, **Open**, **Close**).
- The output of our app will be 0 or 1 ; 0 indicates that stock price will decrease while 1 means increasing of stock price.
- The app runs on local host.
- To deploy it on the internt we have to deploy it to Heroku.
To deploy this project run

```
  streamlit run preprocess.py
```

