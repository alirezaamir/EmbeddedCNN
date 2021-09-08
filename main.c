#include "gan.h"


// ===========================> Functions Prototype <===============================
void conv1d(const __int16_t *data, const __int8_t *filter, __int16_t *map_out, const __int8_t *bias, __int32_t filter_size,
            __int32_t input_len, __int32_t input_depth, __int32_t output_len, __int32_t n_filter, __int32_t strides, __int32_t relu, __int32_t padding);

void conv_max1d(const __int16_t *data, const __int8_t *filter, __int16_t *map_out, const __int8_t *bias, __int32_t filter_size,
            __int32_t input_len, __int32_t input_depth, __int32_t output_len, __int32_t n_filter, __int32_t strides, __int32_t relu, __int32_t padding, __int32_t pool_size);

void batch_normalization(const __int16_t *data, const __int8_t *gamma, const __int8_t *beta, const __int8_t *mean, const __int8_t *var,
                         __int16_t *map_out, __int32_t input_len, __int32_t input_depth);

__int16_t forward_propagation(__int16_t *data);

// =================================================================================

__int32_t main() {
    printf("Input Array : %x\n", input_array[0]);
    __int16_t* eeg_input = input_array;
    __int16_t predict = forward_propagation(eeg_input);
    printf("Prediction : %d", predict);
    return 0;
}


void conv1d(const __int16_t *data, const __int8_t *filter, __int16_t *map_out, const __int8_t *bias, const __int32_t filter_size,
            const __int32_t input_len, const __int32_t input_depth, const __int32_t output_len, const __int32_t n_filter, const __int32_t strides,
            const __int32_t relu, const __int32_t padding) {
    __int32_t sum;
    __int32_t mult;
    __int32_t pad = padding ? filter_size / 2 : 0;
    for (__int32_t w_n = 0; w_n < n_filter; w_n++) {
        for (__int32_t start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (__int32_t w_i = 0; w_i < filter_size; w_i++) {
                for (__int32_t w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - pad < input_len && start_index + w_i - pad > -1) {
                        mult = MUL(mem3d(filter, filter_size, input_depth, w_n, w_j, w_i),
                                   mem2d(data, input_len, w_j, start_index + w_i - pad), NUM_FRACTION_CNV_FC);
                        sum += mult;
                    }
                }
            }
            sum += bias[w_n];
            if (sum < 0 && relu)
                sum = 0; // Relu
            if (sum > (1 << 15) ){
                printf("Overflow %d\n", sum);
                sum = (1<<15) -1;
            }else if (sum < -(1 << 15)){
                printf("Overflow %d\n", sum);
                sum = -(1<<15) +1;
            }
            mem2d(map_out, output_len, w_n, start_index / strides) = (__int16_t) sum;
        }
    }
}


void conv_max1d(const __int16_t *data, const __int8_t *filter, __int16_t *map_out, const __int8_t *bias, const __int32_t filter_size,
            const __int32_t input_len, const __int32_t input_depth, const __int32_t output_len, const __int32_t n_filter, const __int32_t strides,
            const __int32_t relu, const __int32_t padding, const __int32_t pool_size) {
    __int32_t sum;
    __int32_t mult;
    __int32_t pad = padding ? filter_size / 2 : 0;
    for (__int32_t w_n = 0; w_n < n_filter; w_n++) {
        __int32_t maximum = NEG_INF;
        for (__int32_t start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (__int32_t w_i = 0; w_i < filter_size; w_i++) {
                for (__int32_t w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - pad < input_len && start_index + w_i - pad > -1) {
                        mult = MUL(mem3d(filter, filter_size, input_depth, w_n, w_j, w_i),
                                   mem2d(data, input_len, w_j, start_index + w_i - pad), NUM_FRACTION_CNV_FC);
                        sum += mult;
                    }
                }
            }
            sum += bias[w_n];
            if (sum < 0 && relu)
                sum = 0; // Relu
            if (sum > (1 << 15) ){
                printf("Overflow %d\n", sum);
                sum = (1<<15) -1;
            }else if (sum < -(1 << 15)){
                printf("Overflow %d\n", sum);
                sum = -(1<<15) +1;
            }
            if (sum > maximum) {
                maximum = sum;
            }
            if (start_index % pool_size == pool_size-1) {
                mem2d(map_out, output_len, w_n, start_index / (strides * pool_size)) = (__int16_t) maximum;
                maximum = NEG_INF;
            }
        }
    }
}

void batch_normalization(const __int16_t *data, const __int8_t *gamma, const __int8_t *beta, const __int8_t *mean, const __int8_t *var,
                         __int16_t *map_out, const __int32_t input_len, const __int32_t input_depth) {
    for (__int32_t start_index = 0; start_index < input_len; start_index++) {
        for (__int32_t w_j = 0; w_j < input_depth; w_j++) {
            __int16_t normalized = mem2d(data, input_len, w_j, start_index) -
                               ((__int16_t) mean[w_j] << (NUM_FRACTION_DATA - NUM_FRACTION_BN));
            __int16_t standardized = MUL(normalized, var[w_j], NUM_FRACTION_BN);
            __int16_t new_standardized = MUL(standardized, gamma[w_j], NUM_FRACTION_BN);
            mem2d(map_out, input_len, w_j, start_index) = (__int16_t) (new_standardized +
                                                                   ((__int16_t) beta[w_j]
                                                                           << (NUM_FRACTION_DATA - NUM_FRACTION_BN)));
        }
    }
}

void relu(const __int16_t *data, __int16_t *map_out, __int32_t input_len) {
    for (__int32_t i = 0; i < input_len; i++)
        map_out[i] = (data[i] < 0) ? 0 : data[i];
}

void conv_block(__int32_t block, __int16_t* layer_in, __int16_t* conv1d_out){
    __int32_t depth_size[6] = {23, 128, 128, 128, 100, 2};
    __int32_t map_size[6] = {1024, 256, 64, 16, 1, 1};
    int8_t* filter = conv1d_w[block];
    __int8_t* bias = conv1d_b[block];
    conv_max1d(layer_in, filter, conv1d_out, bias, 3, map_size[block], depth_size[block], map_size[block+1],
               depth_size[block + 1], 1, 0, 1, 4);

    batch_normalization(conv1d_out, bn[block * 4], bn[block * 4 + 1], bn[block * 4 + 2], bn[block * 4 + 3],
                        conv1d_out, map_size[block+1], depth_size[block + 1]);

    relu(conv1d_out, conv1d_out, map_size[block+1] * depth_size[block + 1]);
}

__int16_t forward_propagation(__int16_t *data) {
    __int32_t fc_depth_size[6] = {128, 100, 2};
    __int32_t fc_map_size[6] = {16, 1, 1};
    __int16_t intermediate_map0[256*128];
    __int16_t* intermediate_map1 = data;

    //  ************  BLOCK 0  ************ //
    __int16_t *layer_out = intermediate_map0;
    __int16_t *layer_in = (__int16_t *) data;
    conv_block(0, layer_in, layer_out);

    //  ************  BLOCK 1  ************ //
    layer_out = intermediate_map1;
    layer_in = intermediate_map0;
    conv_block(1, layer_in, layer_out);

    //  ************  BLOCK 2  ************ //
    layer_out = intermediate_map0;
    layer_in = intermediate_map1;
    conv_block(2, layer_in, layer_out);

    //  ************  FC 0  ************ //
    layer_out = intermediate_map1;
    layer_in = intermediate_map0;
    conv1d(layer_in, dense_w[0], layer_out, dense_b[0], fc_map_size[0],fc_map_size[0],
           fc_depth_size[0], fc_map_size[1], fc_depth_size[1], fc_map_size[0],
           1, 0);

    //  ************  FC 1  ************ //
    layer_out = intermediate_map0;
    layer_in = intermediate_map1;
    conv1d(layer_in, dense_w[1], layer_out, dense_b[1], fc_map_size[1],fc_map_size[1],
           fc_depth_size[1], fc_map_size[2], fc_depth_size[2], fc_map_size[1],
           0, 0);

    if (layer_out[0] > layer_out[1])
        return 0;
    else
        return 1;
}
