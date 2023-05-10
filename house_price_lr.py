import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
import sys
import argparse
import numpy as np
from pyspark import SparkContext
import pickle
from pprint import pprint

if __name__ == "__main__":
    # Build the SparkSession
    spark = SparkSession.builder \
       .master("local") \
       .appName("Linear Regression Model") \
       .config("spark.executor.memory", "1gb") \
       .getOrCreate()
       
    spark.sparkContext.setLogLevel("ERROR")

    sc = spark.sparkContext


    print('\n\nLoading data set')
    

    rdd = sc.textFile('house-data.csv')
    rdd = rdd.map(lambda line: line.split(","))
    header = rdd.first()
    rdd = rdd.filter(lambda line:line != header)  
    df = rdd.map(lambda line: Row(zip=line[16], beds=line[3], baths=line[4], sqft=line[5], price=line[2])).toDF()

    def cast_columns(df):
        for column in df.columns:
            df = df.withColumn(column, df[column].cast(FloatType()))
        return df    

    df = cast_columns(df)  

    print('\n\nDataset size is:')
    pprint(len(rdd.collect()))

    print('\nDataset sample rows:')
    pprint(df.take(10))   


    print('\n\nDataset mean and standard deviation characteristics:')
    df.describe(['baths', 'beds','price','sqft']).show()    
    

    print('\n\nCleaning up the invalid values from dataset')
    df = df.select('price','baths','beds','sqft')    

    df = df[df.baths > 0]
    df = df[df.beds > 0]
    df = df[df.sqft > 0]
    
    print('\n\nAfter cleaning dataset : mean and standard deviation characteristics:')
    df.describe(['baths','beds','price','sqft']).show()    
    

    temp_data = df.rdd.map(lambda x:(x[0], DenseVector(x[1:])))    
    df2 = spark.createDataFrame(temp_data, ['label','features'])    
    
    print('\n\nSample label and feature vectors:')
    pprint(df2.take(2))    


    print('\n\nScaling the features')

    s_scaler_model = StandardScaler(inputCol='features', outputCol='features_scaled')
    scaler_fn = s_scaler_model.fit(df2)
    scaled_df = scaler_fn.transform(df2)    

    print('\n\nSpliting data set into training and test dat sets')

    train_data, test_data = scaled_df.randomSplit([.80,.20], seed=101) 

    print('\n\nTraining the model with the training data set....')
   
    lr = LinearRegression(labelCol='label', maxIter=20)    
    linear_model = lr.fit(train_data)

    print('\nModel training complete!')

    print('\n\nLinear regression model coefficients:')

    pprint(list(zip(df.columns[1:], linear_model.coefficients)))

    print('\nLinear regression model intercept:')

    pprint(linear_model.intercept)

       
    print('\n\nPredict the house values for testing data set using the linear model')

    predicted = linear_model.transform(test_data)  

    test_predictions = predicted.select('prediction').rdd.map(lambda x:x[0])
    test_labels = predicted.select('label').rdd.map(lambda x:x[0])    

    test_predictions_labels = test_predictions.zip(test_labels)
    test_predictions_labels_df = spark.createDataFrame(test_predictions_labels, 
                                                       ['predictions','labels'])

    test_predictions_labels_df = test_predictions_labels_df.withColumn("delta", abs(col("predictions") - col("labels")))    
	test_predictions_labels_df.write.format('csv').option('header',True).mode('overwrite').option('sep',',').save('file:///home/jsonawane/project/output_file.csv')

    print('\nSample predictions:')

    pprint(test_predictions_labels_df.sort(test_predictions_labels_df.delta).take(100))  
    
    print('\n\nEvaluating model using training and test data set:')

    linear_reg_eval = RegressionEvaluator(predictionCol='predictions', labelCol='labels')    

    linear_reg_eval.evaluate(test_predictions_labels_df)    

    prediction_rmse = linear_reg_eval.evaluate(test_predictions_labels_df, 
                                               {linear_reg_eval.metricName:'rmse'})    

    
    print('\n(training data set root mean squared error, test data set root mean squared error)')
    print((linear_model.summary.rootMeanSquaredError, prediction_rmse))
    
    
    

