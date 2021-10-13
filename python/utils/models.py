import tensorflow as tf
from tensorflow.keras import layers


def get_FCN_model(signal_len=None,):
    input_signal = tf.keras.layers.Input(shape=(signal_len, 23))
    x = input_signal
    x = layers.Conv1D(128, 3, padding="same", activation=None,
                                             kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4))(x)
    bn = layers.BatchNormalization()(x)
    relu = layers.Activation('relu')(bn)
    pool = layers.MaxPooling1D(pool_size=4)(relu)

    num_conv_layers = 2
    for _ in range(num_conv_layers):
        x = layers.Conv1D(128, 3, padding="same", activation=None,
                                             kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4))(pool)
        bn = layers.BatchNormalization()(x)
        relu = layers.Activation('relu')(bn)
        pool = layers.MaxPooling1D(pool_size=4)(relu)

    x = layers.Flatten()(pool)
    x = layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4))(x)
    x = layers.Dropout(rate=0.3)(x)

    final_dense = layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4))(x)

    model = tf.keras.models.Model(inputs=input_signal, outputs=final_dense)
    return model
