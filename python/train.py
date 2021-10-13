import numpy as np
import os
from utils import models
import tensorflow as tf
import logging
from utils.data23 import get_balanced_data, get_test_data
from sklearn.metrics import confusion_matrix, f1_score
from utils.quantization import my_quantized_model, scaled_model


LOG = logging.getLogger(os.path.basename(__file__))
ch = logging.StreamHandler()
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
ch.setFormatter(logging.Formatter(log_fmt))
LOG.addHandler(ch)
LOG.setLevel(logging.INFO)

SEG_LENGTH = 1024
SEQ_LEN = 899
NUM_FRACTIONS_B = 5
NUM_FRACTIONS_W = 8
BITS = 8


def train_model():
    arch = 'FCN_v1'
    subdirname = "../models/{}".format(arch)
    if not os.path.exists(subdirname):
        os.makedirs(subdirname)

    for test_id in range(1, 24):
        fcn_model = models.get_FCN_model(signal_len=SEG_LENGTH,)
        print(fcn_model.summary())
        fcn_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics='accuracy')

        savedir = '{}/test_{}/saved_model/'.format(subdirname, test_id)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for iter in range(200):
            try:
                train_data, train_label = get_balanced_data(test_id, ictal_ratio=0.05, inter_ratio=0.05, non_ratio=0.05)
                train_data = np.clip(train_data, a_min=-250, a_max=250)
                train_data = train_data / 250
                fcn_model.fit(x=train_data, y=tf.keras.utils.to_categorical(train_label, 2), batch_size=32,
                                  initial_epoch= iter*5, epochs=(iter+1)*5)
            except MemoryError:
                print("Memory Error")
                continue

        fcn_model.save(savedir)


def retrain_model():
    '''
    This function is for retraining the model for quantization. It loads the models from load_dir
     and saves the quantized models in save_dir
    '''
    arch = 'FCN_v1'
    subdirname = "../models/{}".format(arch)

    for test_id in range(1, 24):
        load_dir = '{}/test_{}/saved_model/'.format(subdirname, test_id)
        fcn_model = tf.keras.models.load_model(load_dir)
        print(fcn_model.summary())

        scaled_model(fcn_model)
        for trainable_index in [1, 3, 6, 8, 11, 13, 17]:
            fcn_model = my_quantized_model(fcn_model, NUM_FRACTIONS_W, NUM_FRACTIONS_B, BITS)
            for layers_behind in range(trainable_index+1):
                fcn_model.layers[layers_behind].trainable = False
            fcn_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                                  loss='categorical_crossentropy',
                                  metrics='accuracy')
            for _ in range(15):
                try:
                    train_data, train_label = get_balanced_data(test_id, ictal_ratio=0.05, inter_ratio=0.05,
                                                                non_ratio=0.05)
                    train_data = np.clip(train_data, a_min=-250, a_max=250) / 250.0
                    fcn_model.fit(x=train_data, y=tf.keras.utils.to_categorical(train_label, 2), batch_size=32,
                                      epochs=2)
                except MemoryError:
                    print("Memory Error")

        save_dir = '{}/quantized/q_{}_test_{}/saved_model/'.format(subdirname, BITS, test_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fcn_model.save(save_dir)


def inference(test_patient:int, subdirname:str):
    """
    To evaluate the model on the test dataset
    :param test_patient: int, number of test patient. Ex, 1 -> CHB_01
    :param subdirname: str, If TF model is not passed, the method takes a saved model from this address
    :return: confusion matrix and f1-score
    """

    X_test, y_test = get_test_data(test_patient)
    X_test = np.clip(X_test, a_min=-250, a_max=250)
    X_test = X_test/250

    # Load the trained model
    save_path = '{}/model/q_8_test_{}/saved_model/'.format(subdirname, test_patient)
    trained_model = tf.keras.models.load_model(save_path)

    # Evaluate for every sessions in the test dataset

    predicted = trained_model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Pat : {}\n Conf Mat : {}".format(test_patient, conf_mat))
    print("F1-score : {}".format(f1_score(y_test, y_pred)))

    return conf_mat, f1_score(y_test, y_pred)


def get_results():
    """
    This method is only for evaluation a saved model
    """
    arch = 'FCN_v1'
    subdirname = "../models/{}".format(arch)
    f1_scores = []
    for pat_id in range(1, 24):
        pat = pat_id
        pat_conf_mat, f1 = inference(pat, subdirname)
        f1_scores.append(f1)
    print("F1 scores : {}".format(f1_scores))


if __name__ == "__main__":
    train_model()
    # get_results()
    # retrain_model()

