#include <stdio.h>
#include <stdlib.h>
#include "gan.h"


// ===========================> Functions Prototype <===============================
void conv1d(const int *data, const int *filter, int *map_out, const int *bias, int filter_size,
            int input_len, int input_depth, int output_len, int n_filter, int strides, int relu, int padding);

void batch_normalization(const int *data, const int *gamma, const int *beta, const int *mean, const int *var,
                         int *map_out, int input_len, int input_depth);

void skip_add(const int *data, int *map_out, int input_len);

int forward_propagation(int *data);
void save_file(int *data, char *filename, int input_len) ;
// =================================================================================

int main() {
    // allocate memory in CPU for calculation
    int *eeg_input;
    int prediction[2000];
    eeg_input = input_array;
    for (int sample=0; sample< 2000; sample++){
        int predict = forward_propagation(eeg_input+sample*23*1024);
        printf("%d: %d\n", sample, predict);
        prediction[sample] = predict;
    }
    save_file(prediction, "2000.txt", 2000);

    return 0;
}


void conv1d(const int *data, const int *filter, int *map_out,const int *bias, const int filter_size,
            const int input_len, const int input_depth, const int output_len, const int n_filter, const int strides,
            const int relu, const int padding) {
    int sum;
    int mult;
    int pad = padding? filter_size / 2 : 0;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (int w_i = 0; w_i < filter_size; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - pad < input_len && start_index + w_i - pad > -1) {
                        mult = MUL(mem3d(filter, filter_size, input_depth, w_n, w_j, w_i),
                                   mem2d(data, input_len, w_j, start_index + w_i - pad));
                        sum += mult;
                    }
                }
            }
            sum += bias[w_n];
            if (sum < 0 && relu)
                sum = 0; // Relu
            mem2d(map_out, output_len, w_n, start_index/strides) = (int) sum;
        }
    }
}

void max1d(const int *data, int *map_out, const int input_len, const int input_depth,
           const int pool_size, const int strides) {
    int out_len = input_len / strides;
    for (int start_index = 0; start_index < input_len; start_index += strides) {
        for (int w_j = 0; w_j < input_depth; w_j++) {
            int maximum = mem2d(data, input_len, w_j, start_index);
            for (int w_i = 1; w_i < pool_size; w_i++) {
                if (mem2d(data, input_len, w_j, start_index + w_i) > maximum)
                    maximum = mem2d(data, input_len, w_j, start_index + w_i);
            }
            mem2d(map_out, out_len, w_j, start_index / strides) = (int) maximum;
        }
    }
}

void batch_normalization(const int *data, const int *gamma, const int *beta, const int *mean, const int *var,
                         int *map_out, const int input_len, const int input_depth){
    for (int start_index = 0; start_index < input_len; start_index ++) {
        for (int w_j = 0; w_j < input_depth; w_j++) {
            int normalized = MUL(mem2d(data, input_len, w_j, start_index) - mean[w_j] , var[w_j]);
            mem2d(map_out, input_len, w_j, start_index) = MUL(normalized , gamma[w_j]) + beta[w_j];
        }
    }
}

void relu(const int *data, int *map_out, int input_len) {
    for (int i = 0; i < input_len; i++)
        map_out[i] = (data[i] < 0) ? 0 : data[i];
}


void save_file(int *data, char *filename, int input_len) {
    FILE *fp;
    fp = fopen(filename, "w");
    for (int wi = 0; wi < input_len; wi++)
        fprintf(fp, "%d\n", data[wi]);

    fclose(fp);
}

int forward_propagation(int *data) {
    int depth_size[6] = {23, 128, 128, 128, 100, 2};
    int map_size[6] = {1024, 256, 64, 16, 1, 1};
    int *filter;
    int *bias;
    int *layer_in = (int *) data;
//    save_file(data, "input_data.txt", map_size[0] * depth_size[0]);

    for (int block = 0; block < 3; block++) {
        int* conv1d_out = (int *) malloc(map_size[block] * depth_size[block + 1] * sizeof(int));
        filter = conv1d_w[block];
        bias = conv1d_b[block];
        conv1d(layer_in, filter, conv1d_out, bias, 3, map_size[block], depth_size[block], map_size[block],
               depth_size[block + 1], 1, 0, 1);
//        char out_filename[17];
//        sprintf(out_filename, "conv1d_%d.txt", block);
//        save_file(conv1d_out, out_filename, map_size[block] * depth_size[block + 1]);

        batch_normalization(conv1d_out, bn[block*4], bn[block*4+1], bn[block*4+2], bn[block*4+3],
                            conv1d_out, map_size[block], depth_size[block+1]);
//        sprintf(out_filename, "bn_%d.txt", block);
//        save_file(conv1d_out, out_filename, map_size[block] * depth_size[block + 1]);

        relu(conv1d_out, conv1d_out, map_size[block] * depth_size[block + 1]);

        int* max_out = (int *) malloc(map_size[block+1] * depth_size[block + 1] * sizeof(int));
        max1d(conv1d_out, max_out, map_size[block], depth_size[block+1], 4, 4);
//        sprintf(out_filename, "max_pool_%d.txt", block);
//        save_file(max_out, out_filename, map_size[block + 1] * depth_size[block + 1]);
        layer_in = max_out;
    }
    for (int fc_index=0; fc_index<2; fc_index++) {
        int *fully_connected = malloc(map_size[4+fc_index] * depth_size[4+fc_index] * sizeof(int));
        conv1d(layer_in, dense_w[fc_index], fully_connected, dense_b[fc_index], map_size[3+fc_index],
               map_size[3+fc_index], depth_size[3+fc_index],
               map_size[4+fc_index], depth_size[4+fc_index], map_size[3+fc_index],
               fc_index<1 ? 1: 0, 0);
        char out_filename[17];
//        sprintf(out_filename, "dense_%d.txt", fc_index);
//        save_file(fully_connected, out_filename, map_size[4+fc_index] * depth_size[4+fc_index]);
        layer_in = fully_connected;
    }
//    printf("%d, %d\n", layer_in[0], layer_in[1]);
    if (layer_in[0] > layer_in[1])
        return 0;
    else
        return 1;
}
