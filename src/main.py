import pandas as pd


def get_x_train() -> pd.DataFrame:
    return pd.read_csv("data/X_train.csv")


def get_y_train() -> pd.DataFrame:
    return pd.read_csv("data/Y_train.csv")


def get_x_test() -> pd.DataFrame:
    return pd.read_csv("data/X_test.csv")


def save_to_csv(df: pd.DataFrame) -> None:
    df.to_csv("data/y_test.csv", index=False)


def main():
    df_x_train = get_x_train()
    df_y_train = get_x_train()
    df_x_test = get_x_test()


if __name__ == "__main__":
    main()
