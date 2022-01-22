import numpy as np
import pandas as pd
import os

from preprocessor import Preprocessor
import keras_tuner as kt
import tensorflow as tf
from trainer import build_model


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
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)
    df_x_train = get_x_train()
    df_y_train = get_x_train()
    df_x_test = get_x_test()
    preprocessor = Preprocessor(df_x_train, df_y_train, df_x_test)
    (training_data, test_data) = preprocessor.get_training_and_test_data()
    MAX_EPOCHS = 20
    count = 1
    predictions = None
    for target_id in training_data:
        print(f"Training for asset {count}")
        count+=1
        (x, y) = training_data[target_id]
        tuner = kt.Hyperband(
            build_model,
            objective="val_accuracy",
            max_epochs=MAX_EPOCHS,
            hyperband_iterations=10,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(x, y, epochs=50, validation_split=0.2, callbacks=[early_stopping])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(x, y, epochs=50, validation_split=0.2)
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        hypermodel = tuner.hypermodel.build(best_hps)
        hypermodel.fit(x, y, epochs=best_epoch, validation_split=0.2)
        (ids, x_test) = test_data[target_id]
        if predictions:
            (prev_ids, preds) = predictions
            predictions = (
                np.append(prev_ids, ids),
                np.append(preds, hypermodel.predict(x_test)),
            )
        else:
            preds = hypermodel.predict(x_test)
            (n, _) = preds.shape
            predictions = (ids, preds.reshape((n,)))

    predictions = np.column_stack(predictions)
    df = produce_dataframe(predictions)
    save_to_csv(df)


if __name__ == "__main__":
    main()
