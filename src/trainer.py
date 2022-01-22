import tensorflow as tf


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(100,)))
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=16, max_value=128, step=16),
                activation="relu",
            )
        )

        model.add(
            tf.keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1))
        )

    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=sign_binary_crossentropy,
        metrics=[custom_weighted_accuracy],
    )
    return model


def custom_weighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    norm_y_acc = tf.reduce_sum(tf.abs(y_true))
    return (1 / norm_y_acc) * tf.reduce_sum(
        tf.abs(y_true) * (y_pred == tf.sign(norm_y_acc))
    )


def sign_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.keras.losses.binary_crossentropy((tf.sign(y_true) + 1) // 2, y_pred)
