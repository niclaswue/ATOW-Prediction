import pandas as pd
from sklearn.base import BaseEstimator
from models.base_model import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class ScikitLearnModel(BaseModel):

    def __init__(self, model_class, model_params=None, test_size=0.2):
        if not issubclass(model_class, BaseEstimator):
            raise ValueError("model_class must be a scikit-learn estimator")

        name = model_class().__repr__()
        name = name.split("(")[0] if "(" in name else name
        super().__init__(name)
        self.model_class = model_class
        self.model_params = model_params or {}
        self.model = None
        self.test_size = test_size
        self.feature_columns = None
        self.target_column = "tow"
        self.metrics = {}

    def _preprocess_df(self, df: pd.DataFrame):
        # drop columnms not useful for prediction
        drop_cols = ["flight_id", "callsign", "name_adep", "name_ades"]
        df = df.drop(columns=drop_cols + [self.target_column])

        # convert objects to categorical
        objs = df.select_dtypes(["object"])
        df[objs.columns] = objs.apply(lambda x: x.astype("category").cat.codes)

        return df

    def train(self, training_df: pd.DataFrame):
        # Separate features and target
        X = self._preprocess_df(training_df)
        y = training_df[self.target_column]

        # Store feature column names
        self.feature_columns = X.columns.tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Initialize and train the model
        self.model = self.model_class(**self.model_params)
        print("model.fit")
        self.model.fit(X_train, y_train)
        print("Done")

        # Calculate metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        self.metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
        }

    def predict(self, input_df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        X = self._preprocess_df(input_df)
        return pd.Series(self.model.predict(X))

    def info(self):
        if self.model is None:
            return {"status": "Model not trained"}

        info = {
            "model_class": self.model_class.__name__,
            "model_params": self.model_params,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }

        # Add model-specific information if available
        if hasattr(self.model, "feature_importances_"):
            info["feature_importances"] = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )
        elif hasattr(self.model, "coef_"):
            info["coefficients"] = dict(zip(self.feature_columns, self.model.coef_))

        return info
