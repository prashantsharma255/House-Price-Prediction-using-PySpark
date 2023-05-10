# House-Price-Prediction-using-PySpark

House prices fluctuate every year, so there is a need for a system to predict house prices in the future. House price prediction can help the developer determine the selling price of a house and can help the customer not only to arrange the right time to purchase a house, but also manage their finances accordingly. The factors taken into consideration that can influence a price of the house are number of baths, beds, location, year built, etc and the square footage of the house. With the real estate industry booming these days it is important for buyers and sellers to have a tool like this at their disposal. Predicting house prices is critical for real estate efficiency. House prices were previously calculated by estimating the purchasing and selling prices in a certain area. As a result, the House Price Prediction Model is critical for closing the information gap and increasing Real Estate efficiency. We will be able to effectively estimate prices using this model.

### Goals

Our project has the below mentioned goals:
●	To predict house prices using linear regression model.
●	Then, calculation of Root Mean Squared Error (RMSE) to detect the accuracy of our predictions.
●	To export our predictions into a csv file as output.


### Problem Setting 

The Linear regression is a supervised machine learning model. This model estimates the relationship between one dependent variable and one or more independent variables using a straight line. In our case the independent variables are sq. footage, no. of beds, no. of baths and the independent variable is price. Using these 4 factors we must train our model to predict our dependent variable ie. house price using our independent variables (sq. footage, no. of beds, no. of baths).


### Linear Regression

Mathematically, we can write the linear relationship as:
              **y = β0 + β1x+ e**
              
Where,
- y is the output variable, e.g. house prices
- x is the input variable, e.g. size of a house in square meters
- β0 is the intercept i.e. the value of y when x=0
- β1 is the coefficient for x and the slope of the regression line
- e is the error term

Our linear regression formula:
**Price = (bath_coe * no_of_baths) - (bed_coe * no_of_beds) + (sq_ft_coe * sq_ft) + regression_intercept**

During the linear regression implementation, the algorithm finds the line of best fit using the model coefficients β0 and β1, such that it is as close as possible to the actual data points. 

After finding the β0 and β1 we can use the model to predict the response.
β0 is the intercept (the value of y when x=0).

There are several applications for linear regression.
The majority of applications fall into one of two categories:
●	Linear regression may be used to fit a prediction model to an observed dataset values of the answer and explanatory variables if the aim is predictions, forecasting, or error reduction. If new values of explanatory variables are obtained without an associated response value after creating such a model, the fitted model may be used to predict the response.
●	If the goal is to measure the strength of relationship between both the response as well as the explanatory variables, and then in specific to determine whether certain explanatory variables have no linear relation with the response at all, or to recognize that what subsets of explanatory variables could contain redundant information, linear regression analysis is used.

### Dataset

A data set represents a collection of information. A data set refers to one or more database tables in the case of tabular data, where each column of a table represents a specific variable, and each row represents a specific record of the data set in question. For each item of the data set, the data set provides values for each one of the variables, such as an object's height and weight. A collection of documents or files can also be included in a data set.

The structure and attributes of a data collection are defined by a number of factors. These include the number and types of characteristics or variables, as well as numerous statistical measures such as standard deviation and kurtosis that may be applied to them.

The values may be numerical data (i.e., not comprising of numerical values), such as a person's height in centimeters, or nominal data (i.e., not consisting of numerical values), such as a person's ethnicity. Values might be of any of the types defined as a level of measurement in general. The values for each variable are usually of the same kind. There may, however, be some missing data that must be communicated in some way.

We are using a dataset with information of 21,500+ houses of **Washington state**.
This dataset provides us **21 columns** representing different house features. 
For this project, we are using **no_of_bedrooms, no_of_bathrooms, and square footage** as features to predict the house price.
