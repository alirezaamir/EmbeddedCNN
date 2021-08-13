#include <stdio.h>
#include <stdlib.h>
#include "gan.h"


// ===========================> Functions Prototype <===============================
//void conv1d(int *data, int *filter, int *map_out, int *bias, int filter_size,
//            int input_len, int input_depth, int n_filter, int strides);
void conv1d(const int *data, const int *filter, int *map_out, const int *bias, int filter_size,
            int input_len, int input_depth, int output_len, int n_filter, int strides);

        void deconv1d(int *data, const int *filter, int *map_out,
              int input_len, int input_depth, int n_filter);

void skip_add(const int *data, int *map_out, int input_len);

void forward_propagation(int *data);
// =================================================================================

int main() {
    // allocate memory in CPU for calculation
    int *ecg_input;
    ecg_input = input_array;
    forward_propagation(ecg_input);
    return 0;
}


void conv1d(const int *data, const int *filter, int *map_out,const int *bias, const int filter_size,
            const int input_len, const int input_depth, const int output_len, const int n_filter, const int strides) {
    int sum;
    int mult;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index <= input_len-filter_size; start_index += strides) {
            sum = 0;
            for (int w_i = 0; w_i < filter_size; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    mult = MUL(mem3d(filter, filter_size, input_depth, w_n, w_j, w_i),
                               mem2d(data, input_len, w_j, start_index + w_i));
                    sum += mult;
                }
            }
            if (sum < 0)
                sum = 0; // Relu
            mem2d(map_out, output_len, w_n, start_index/strides) = sum + bias[w_n];
        }
    }
}

void max1d(const int *data, int *map_out, const int input_len, const int input_depth,
           const int pool_size, const int strides) {
    int out_len = (input_len - pool_size / 2) / strides;
    for (int start_index = 0; start_index < input_len - pool_size; start_index += strides) {
        for (int w_j = 0; w_j < input_depth; w_j++) {
            int maximum = mem2d(data, input_len, w_j, start_index);
            for (int w_i = 1; w_i < pool_size; w_i++) {
                if (mem2d(data, input_len, w_j, start_index + w_i) > maximum)
                    maximum = mem2d(data, input_len, w_j, start_index + w_i);
            }
            mem2d(map_out, out_len, w_j, start_index / strides) = maximum;
        }
    }
}

void skip_add(const int *data, int *map_out, int input_len) {
    for (int i = 0; i < input_len; i++)
        map_out[i] += data[i];
}


void save_file(int *data, char *filename, int input_len) {
    FILE *fp;
    fp = fopen(filename, "w");
    for (int wi = 0; wi < input_len; wi++)
        fprintf(fp, "%d\n", data[wi]);

    fclose(fp);
}

void forward_propagation(int *data) {
    int depth_size[11] = {1, 64, 64, 64, 64, 128, 128, 256, 256, 512, 512};
    int map_size[11] = {768, 381, 190,  94, 92, 45, 43, 21, 19, 9, 7};

    int *filter;
    int *bias;
    int *layer_in = (int *) data;
    save_file(data, "input_data.txt", 768);

    //
    int *first_conv = (int *) malloc(map_size[1] * depth_size[1] * sizeof(int));
    filter = enc_w[0];
    bias = bias_w[0];
    conv1d(layer_in, filter, first_conv, bias, 7, map_size[0], depth_size[0], map_size[1],
           depth_size[1], 2);
    int *max_pool = (int *) malloc(map_size[2] * depth_size[2] * sizeof(int));
    max1d(first_conv, max_pool, map_size[1], depth_size[1], 3, 2);
    save_file(max_pool, "max_pool1.txt", map_size[2] * depth_size[2]);
    //free(first_conv);

    layer_in = max_pool;
    for (int block = 0; block < 4; block++) {
        int index = block * 2 + 2;
        int* conv1d_1 = (int *) malloc(map_size[index+1] * depth_size[index+1] * sizeof(int));
        filter = enc_w[index-1];
        bias = bias_w[index-1];
        conv1d(layer_in, filter, conv1d_1, bias, 3, map_size[index], depth_size[index], map_size[index+1],
               depth_size[index + 1], 2);
        char out_filename[17];
        sprintf(out_filename, "conv1d_b%d_1.txt", block);
        save_file(conv1d_1, out_filename, map_size[index + 1] * depth_size[index + 1]);

        int *conv1d_2 = (int *) malloc(map_size[index+2] * depth_size[index+2] * sizeof(int));
        filter = enc_w[index];
        bias = bias_w[index];
        conv1d(conv1d_1, filter, conv1d_2, bias, 3, map_size[index + 1], depth_size[index+1], map_size[index + 2],
               depth_size[index + 2], 1);
        sprintf(out_filename, "conv1d_b%d_2.txt", block);
        save_file(conv1d_2, out_filename, map_size[index + 2] * depth_size[index + 2]);

        int* conv1d_skip = (int *) malloc(map_size[index + 2] * depth_size[index + 2] * sizeof(int));
        filter = skip_w[block];
        bias = skip_bias[block];
        conv1d(layer_in, filter, conv1d_skip, bias, 7, map_size[index], depth_size[index], map_size[index + 2],
               depth_size[index + 2], 2);

        skip_add(conv1d_2, conv1d_skip, map_size[index + 2] * depth_size[index + 2]);
        sprintf(out_filename, "conv1d_b%d_3.txt", block);
        save_file(conv1d_skip, out_filename, map_size[index + 2] * depth_size[index + 2]);
        layer_in = conv1d_skip;
    }
    int *fully_connected = malloc(2 * sizeof(int));
    conv1d(layer_in, fc_weights_1, fully_connected, fc_bias_1, 7, map_size[10], depth_size[10],
           1, 2, 1);

    save_file(fully_connected, "fxp_output.txt", 2);
}
