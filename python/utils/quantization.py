import tensorflow as tf
import numpy as np


def get_fxp_var(std, NUM_FRACTION, BIT_RANGE):
    var_inv = 1 / np.sqrt(np.add(std, 0.001))
    w_new = np.round(np.array(var_inv * (1 << NUM_FRACTION))).astype(np.int)
    w_new = np.clip(w_new, a_min=-(1 << BIT_RANGE) + 1, a_max=(1 << BIT_RANGE) - 1)
    w_new = np.where(w_new == 0, w_new+1, w_new)
    w_new = w_new.astype(np.float)/(1 << NUM_FRACTION)
    new_var = 1.0/(w_new ** 2) - 0.001
    return new_var


def my_quantized_model(model, num_fraction_w, num_fraction_b, bits):
    bit_range = bits - 1
    for layer in model.layers:
        # print(layer.name)
        if len(layer.get_weights()) > 0:
            new_weights = []
            for idx, w in enumerate(layer.get_weights()):
                if idx == 3:  # Gamma in abatch normalization layer
                    new_var = get_fxp_var(w, num_fraction_b, bit_range)
                    new_weights.append(new_var)
                    # new_weights.append(w)
                else:
                    num_fraction = num_fraction_b if (layer.name).startswith('batch') else num_fraction_w
                    w_new = np.round(np.array(w * (1 << num_fraction))).astype(np.int)
                    w_new = np.clip(w_new, a_min=-(1 << bit_range) + 1, a_max= (1 << bit_range) - 1)
                    new_weights.append(w_new.astype(np.float)/(1 << num_fraction))
            layer.set_weights(new_weights)

    return model


def scaled_model(model:tf.keras.models.Model):
    scaled = model
    first_conv_name = [layer.name for layer in model.layers if layer.name.startswith('conv')][0]
    first_batch_name = [layer.name for layer in model.layers if layer.name.startswith('batch')][0]
    w =  model.get_layer(first_conv_name).get_weights()[0]
    bias = model.get_layer(first_conv_name).get_weights()[1]
    new_weight = [w, bias/250]
    scaled.get_layer(first_conv_name).set_weights(new_weight)

    gamma = model.get_layer(first_batch_name).get_weights()[0]
    beta = model.get_layer(first_batch_name).get_weights()[1]
    mean = model.get_layer(first_batch_name).get_weights()[2]
    var = model.get_layer(first_batch_name).get_weights()[3]
    new_weight = [gamma, beta, mean/250, var/(250*250)]
    scaled.get_layer(first_batch_name).set_weights(new_weight)
    return scaled
