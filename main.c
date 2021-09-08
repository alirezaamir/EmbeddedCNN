#include <stdio.h>
#include <stdlib.h>
#include "gan.h"


// ===========================> Functions Prototype <===============================
void conv1d(const short *data, const char *filter, short *map_out, const char *bias, int filter_size,
            int input_len, int input_depth, int output_len, int n_filter, int strides, int relu, int padding);

void conv_max1d(const short *data, const char *filter, short *map_out, const char *bias, int filter_size,
            int input_len, int input_depth, int output_len, int n_filter, int strides, int relu, int padding, int pool_size);

void batch_normalization(const short *data, const char *gamma, const char *beta, const char *mean, const char *var,
                         short *map_out, int input_len, int input_depth);

short forward_propagation(short *data);

// =================================================================================

int main() {
    short* eeg_input = input_array;
    short predict = forward_propagation(eeg_input);
    printf("Prediction : %d", predict);
    return 0;
}


void conv1d(const short *data, const char *filter, short *map_out, const char *bias, const int filter_size,
            const int input_len, const int input_depth, const int output_len, const int n_filter, const int strides,
            const int relu, const int padding) {
    int sum;
    int mult;
    int pad = padding ? filter_size / 2 : 0;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (int w_i = 0; w_i < filter_size; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
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
            mem2d(map_out, output_len, w_n, start_index / strides) = (short) sum;
        }
    }
}


void conv_max1d(const short *data, const char *filter, short *map_out, const char *bias, const int filter_size,
            const int input_len, const int input_depth, const int output_len, const int n_filter, const int strides,
            const int relu, const int padding, const int pool_size) {
    int sum;
    int mult;
    int pad = padding ? filter_size / 2 : 0;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        int maximum = NEG_INF;
        for (int start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (int w_i = 0; w_i < filter_size; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
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
                mem2d(map_out, output_len, w_n, start_index / (strides * pool_size)) = (short) maximum;
                maximum = NEG_INF;
            }
        }
    }
}

void batch_normalization(const short *data, const char *gamma, const char *beta, const char *mean, const char *var,
                         short *map_out, const int input_len, const int input_depth) {
    for (int start_index = 0; start_index < input_len; start_index++) {
        for (int w_j = 0; w_j < input_depth; w_j++) {
            short normalized = mem2d(data, input_len, w_j, start_index) -
                               ((short) mean[w_j] << (NUM_FRACTION_DATA - NUM_FRACTION_BN));
            short standardized = MUL(normalized, var[w_j], NUM_FRACTION_BN);
            short new_standardized = MUL(standardized, gamma[w_j], NUM_FRACTION_BN);
            mem2d(map_out, input_len, w_j, start_index) = (short) (new_standardized +
                                                                   ((short) beta[w_j]
                                                                           << (NUM_FRACTION_DATA - NUM_FRACTION_BN)));
        }
    }
}

void relu(const short *data, short *map_out, int input_len) {
    for (int i = 0; i < input_len; i++)
        map_out[i] = (data[i] < 0) ? 0 : data[i];
}

void conv_block(int block, short* layer_in, short* conv1d_out){
    int depth_size[6] = {23, 128, 128, 128, 100, 2};
    int map_size[6] = {1024, 256, 64, 16, 1, 1};
    char* filter = conv1d_w[block];
    char* bias = conv1d_b[block];
    conv_max1d(layer_in, filter, conv1d_out, bias, 3, map_size[block], depth_size[block], map_size[block+1],
               depth_size[block + 1], 1, 0, 1, 4);

    batch_normalization(conv1d_out, bn[block * 4], bn[block * 4 + 1], bn[block * 4 + 2], bn[block * 4 + 3],
                        conv1d_out, map_size[block+1], depth_size[block + 1]);

    relu(conv1d_out, conv1d_out, map_size[block+1] * depth_size[block + 1]);
}

short forward_propagation(short *data) {
    int fc_depth_size[6] = {128, 100, 2};
    int fc_map_size[6] = {16, 1, 1};
    short intermediate_map0[256*128];
    short intermediate_map1[64*128];

    //  ************  BLOCK 0  ************ //
    short *layer_out = intermediate_map0;
    short *layer_in = (short *) data;
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

    if (layer_in[0] > layer_in[1])
        return 0;
    else
        return 1;
}
