import pandas as pd
from sklearn.base import BaseEstimator
from models.base_model import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class ScikitLearnModel(BaseModel):

    def __init__(self, model_class, model_params=None, test_size=0.2, name: str = None):
        if not issubclass(model_class, BaseEstimator):
            raise ValueError("model_class must be a scikit-learn estimator")

        if not name:
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

    def _preprocess_inputs(self, df: pd.DataFrame):
        # drop columnms not useful for prediction
        drop_cols = ["flight_id", "name_adep", "name_ades"] + [self.target_column]
        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[self.target_column] if self.target_column in df.columns else None

        # convert objects to categorical
        objs = X.select_dtypes(["object"])
        X[objs.columns] = objs.apply(lambda x: x.astype("category").cat.codes)

        return X, y

    def _postprocess_output(self, input_df, output):
        if self.model_params.get("round_output"):
            output = output.round()
        return output

    def train(self, training_df: pd.DataFrame):
        # Separate features and target
        X, y = self._preprocess_inputs(training_df)

        # Store feature column names
        self.feature_columns = X.columns.tolist()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Initialize and train the model
        pipeline = [StandardScaler(), self.model_class(**self.model_params)]
        self.model = make_pipeline(*pipeline)
        print("model.fit")
        self.model.fit(X_train, y_train)
        print("Done")

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Calculate metrics
        self.metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
        }

    def predict(self, input_df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        X, _ = self._preprocess_inputs(input_df)
        y = self.model.predict(X)
        y = self._postprocess_output(X, y)
        return pd.Series(y)

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
