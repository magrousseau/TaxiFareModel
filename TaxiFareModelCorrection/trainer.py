# imports
from TaxiFareModelCorrection.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModelCorrection.utils import compute_rmse
from TaxiFareModelCorrection.data import get_data, clean_data
from TaxiFareModelCorrection.params import BUCKET_NAME, GCP_MODEL_PATH

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# import datetime
# import os
# import numpy as np
from google.cloud import storage
import joblib


class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        col_distance = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
        col_time = ['pickup_datetime']
        pipe_distance = Pipeline([('transformer', DistanceTransformer()),
                                  ('scaler', RobustScaler())])

        pipe_time = Pipeline([('transformer', TimeFeaturesEncoder('pickup_datetime')),
                              ('encoder', OneHotEncoder(handle_unknown='ignore'))])

        processing_pipe = ColumnTransformer([('pipe_time', pipe_time, col_time),
                                            ('pipe_distance', pipe_distance, col_distance)])

        self.pipeline = Pipeline([('preprocessing', processing_pipe),
                                  ('regression', LinearRegression())])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)

        return rmse

    # def train(self):
    #     self.run()
    #     rmse = self.evaluate(X_test, y_test)
    #     print(rmse)

    def save_model(self, reg):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        filepath = 'model.joblib'
        joblib.dump(reg, filepath)
        print("saved model.joblib locally")

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        # Then do other things...
        blob = bucket.blob(GCP_MODEL_PATH)
        blob.upload_from_filename(filename=filepath)

        print(f"uploaded model.joblib to gcp cloud storage under \n => {filepath}")


if __name__ == "__main__":
    nrows = 1_000
    df = get_data(nrows=nrows)
    df = clean_data(df)
    print(df.columns)
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)
    reg = trainer.run()
    trainer.save_model(reg)
    print(trainer.evaluate(X_test, y_test))


