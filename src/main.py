import numpy as np
import pandas as pd

from preprocessor import Preprocessor


def get_x_train() -> pd.DataFrame:
    return pd.read_csv("data/X_train.csv")


def get_y_train() -> pd.DataFrame:
    return pd.read_csv("data/Y_train.csv")


def get_x_test() -> pd.DataFrame:
    return pd.read_csv("data/X_test.csv")


def save_to_csv(df: pd.DataFrame) -> None:
    df.to_csv("data/y_test.csv", index=False)


def produce_dataframe(predictions: np.ndarray) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(predictions, columns=["ID", "RET_TARGET"])
    df = df.sort_values(by=["ID"])
    return df.reset_index(drop=True)


def main():
    df_x_train = get_x_train()
    df_y_train = get_x_train()
    df_x_test = get_x_test()
    preprocessor = Preprocessor(df_x_train, df_y_train, df_x_test)
    (training_data, test_data) = preprocessor.get_training_and_test_data()


if __name__ == "__main__":
    main()
