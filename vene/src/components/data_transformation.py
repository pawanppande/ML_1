import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline for numerical and categorical features.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Preprocessing pipelines created successfully.")

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns),
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train and test CSVs, applies preprocessing, 
        saves the preprocessor, and returns transformed arrays.
        """
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded successfully.")

            target_column = "math_score"
            input_features = train_df.drop(columns=[target_column], axis=1)
            target_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_test = test_df[target_column]

            # Get preprocessing object
            preprocessor = self.get_data_transformer_object()

            # Transform data
            logging.info("Applying preprocessing on train and test data.")
            input_train_arr = preprocessor.fit_transform(input_features)
            input_test_arr = preprocessor.transform(input_features_test)

            train_arr = np.c_[input_train_arr, np.array(target_train)]
            test_arr = np.c_[input_test_arr, np.array(target_test)]

            # Ensure artifact directory exists
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            # Save preprocessor object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info(f"Preprocessor object saved at {self.config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    transformation.initiate_data_transformation(train_data, test_data)
