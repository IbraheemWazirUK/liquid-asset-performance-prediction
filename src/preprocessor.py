from typing import Dict, Tuple

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(
        self,
        df_x_train: pd.DataFrame,
        df_y_train: pd.DataFrame,
        df_x_test: pd.DataFrame,
    ) -> None:
        self.df_x_train: pd.DataFrame = df_x_train
        self.df_y_train: pd.DataFrame = df_y_train
        self.df_x_test: pd.DataFrame = df_x_test
        self.target_ids = self.df_x_train["ID_TARGET"].unique()
        self._handle_missing_values()
        self.min_value = {}
        self.max_value = {}
        self._normalise_data()

    def _normalise_data(self, method="minmax") -> None:
        if method == "minmax":
            for column in list(
                filter(
                    lambda x: self.df_x_train[x].dtype == np.float64,
                    self.df_x_train.columns,
                )
            ):
                min_value = self.min_value[column] = self.df_x_train[column].min()
                max_value = self.max_value[column] = self.df_x_train[column].max()
                if max_value > min_value:
                    self.df_x_train[column] = (self.df_x_train[column] - min_value) / (
                        max_value - min_value
                    )
                    self.df_x_test[column] = (self.df_x_test[column] - min_value) / (
                        max_value - min_value
                    )

    def _compute_means(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            column: df[column].mean()
            for column in df.columns
            if df[column].dtype == np.float64
        }

    def _handle_missing_values(
        self, method: str = "mean"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if method == "mean":
            self.df_x_train = self.df_x_train.fillna(
                self._compute_means(self.df_x_train)
            )
            self.df_x_test = self.df_x_test.fillna(self._compute_means(self.df_x_test))

    def get_training_and_test_data(
        self,
    ) -> Tuple[
        Dict[int, Tuple[np.ndarray, np.ndarray]],
        Dict[int, Tuple[np.ndarray, np.ndarray]],
    ]:
        training_data = {}
        test_data = {}
        index_train = self.df_x_train.index
        index_test = self.df_x_test.index
        for target_id in self.target_ids:
            target_indices = index_train[self.df_x_train["ID_TARGET"] == target_id]
            target_x_rows: np.ndarray = self.df_x_train.loc[target_indices].to_numpy()
            target_y_rows: np.ndarray = self.df_y_train.loc[target_indices].to_numpy()
            training_data[target_id] = (target_x_rows[:, 2:-1], target_y_rows[:, 1])
            test_indices = index_test[self.df_x_test["ID_TARGET"] == target_id]
            test_rows: np.ndarray = self.df_x_test.loc[test_indices].to_numpy()
            test_data[target_id] = (test_rows[:, 0], test_rows[:, 2:-1])

        print("Training and test data successfully acquired.")
        return (training_data, test_data)
