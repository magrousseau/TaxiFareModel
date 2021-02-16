# imports
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


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

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.drop('fare_amount', axis=1)
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    print(trainer.evaluate(X_test, y_test))
