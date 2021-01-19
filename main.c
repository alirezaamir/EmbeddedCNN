#include <stdio.h>
#include <stdlib.h>
#include "gan.h"
//#include "gan.c"

#define FILTER_SIZE 31


// ===========================> Functions Prototype <===============================
void fill_map(float *data, int size, int depth);

void fill_filter(float *filter, int layer_num);
void fill_dec(float *filter, int layer_num);

void fill_skip(float *filter, int size, int depth);

void conv1d(const float *data, const float *filter, float *map_out,
            int input_len, int input_depth, int n_filter);

void deconv1d(float *data, const float *filter, float *map_out,
              int input_len, int input_depth, int n_filter);

void skip_add(float *data, float *filter, float *map_out,
              int input_len, int input_depth);

void forward_propagation(float *data);
// =================================================================================

int main() {
    printf("EpilepsyGAN!\n");

    // allocate memory in CPU for calculation
    float *eeg_input;
    const int in_len = 2048;
    eeg_input = (float *) malloc(in_len * sizeof(float));
//    filter      = (float*)malloc( * sizeof(float));
//    c_serial = (float*)malloc(n*n * sizeof(float));
//    c        = (float*)malloc(n*n * sizeof(float));

    // fill a, b matrices with random values between -16.0f and 16.0f
//    srand(0);
    fill_map(eeg_input, in_len, 1);
    forward_propagation(eeg_input);
    free(eeg_input);
    return 0;
}


void fill_map(float *data, int size, int depth) {
//    for (int i = 0; i < size * depth; ++i) {
//        data[i] = (float) (rand() % 17 - 8);
//    }
data = input_array;
}

void fill_filter(float *filter, int layer_num) {
//    for (int i = 0; i < FILTER_SIZE * depth * n; ++i)
//        filter[i] = (float) (rand() % 17 - 8);
filter = enc_w[layer_num];
}

void fill_dec(float *filter, int layer_num){
    filter = dec_w[layer_num];
}

void fill_skip(float *filter, int size, int depth){
    for (int i = 0; i < size * depth; ++i)
        filter[i] = (float) (rand() % 17 - 8);
}

void conv1d(const float *data, const float *filter, float *map_out,
            const int input_len, const int input_depth, const int n_filter) {
    float sum;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 1; start_index < input_len; start_index += 2) {
            sum = 0;
            for (int w_i = 0; w_i < FILTER_SIZE; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - 15 < input_len && start_index + w_i - 15 > -1)
                        sum += mem3d(filter, FILTER_SIZE, input_depth, w_n, w_j, w_i) *
                               mem2d(data, input_len, w_j, start_index + w_i - 15);
                }
            }
            if (sum < 0)
                sum *= 0.3f; // Leaky Relu
            mem2d(map_out, input_len >> 1, w_n, start_index >> 1) = sum;
        }
    }
}

void deconv1d(float *data, const float *filter, float *map_out,
              int input_len, const int input_depth, const int n_filter) {
    // Upsampling
    input_len *= 2; // Update input len to the upsampled one
    float* upsampled = (float *) malloc( input_len * input_depth * sizeof(float));
    for (int i=0; i<input_len; i++){
        if (i%2 ==1)
            upsampled[i] = data[i>>1];
        else
            upsampled[i] = 0;
    }
//    free(data);

    // Convolution
    float sum;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index < input_len; start_index ++) {
            sum = 0;
            for (int w_i = 0; w_i < FILTER_SIZE; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - 15 < input_len && start_index + w_i - 15 > -1)
                        sum += mem3d(filter, FILTER_SIZE, input_depth, w_n, w_j, FILTER_SIZE - w_i -1) *
                               mem2d(upsampled, input_len, w_j, start_index + w_i - 15);
                }
            }
            if (sum < 0)
                sum *= 0.3f; // Leaky Relu
            mem2d(map_out, input_len >> 1, w_n, start_index >> 1) = sum;
        }
    }
}


void concatenate(const float* data, const float* z, float* map_out,
                 int input_len, int input_depth){
    for (int w_j = 0; w_j < input_depth; w_j++) {
        for (int w_i = 0; w_i < input_len; w_i++) {
            mem2d(map_out, input_len, w_j, w_i) = mem2d(data, input_len, w_j, w_i);
            mem2d(map_out, input_len, w_j+input_depth, w_i) = mem2d(z, input_len, w_j, w_i);
        }
    }
}


void skip_add(float *data, float *filter, float *map_out,
              int input_len, const int input_depth){
    for (int w_j = 0; w_j < input_depth; w_j++) {
        for (int w_i = 0; w_i < input_len; w_i++) {
            float add = mem2d(data, input_len, w_j, w_i) * mem2d(filter, input_len, w_j, w_i);
            mem2d(map_out, input_len, w_j, w_i) += add;
        }
    }
//    free(data);
//    free(filter);
}

void forward_propagation(float *data) {
    float *encoder_layers_out[8] = {0};
    int depth_size[9] = {1, 64, 64, 128, 128, 256, 256, 512, 1024};
    int map_size[9] = {2048, 1024, 512, 256, 128, 64, 32, 16, 8};

    float *filter;
    float *layer_in = data;

    // Encoder
    for (int layer = 0; layer < 2; layer++) {
        printf("Encoder Layer %d\n", layer);
        encoder_layers_out[layer] = (float *) malloc(map_size[layer + 1] * depth_size[layer + 1] * sizeof(float));
//        filter = (float *) malloc(FILTER_SIZE * depth_size[layer] * depth_size[layer + 1] * sizeof(float));
//        fill_filter(filter, depth_size[layer], depth_size[layer + 1]);
//        fill_filter(filter, layer);
        filter = enc_w[layer];
        conv1d(layer_in, filter, encoder_layers_out[layer], map_size[layer], depth_size[layer], depth_size[layer + 1]);
        layer_in = encoder_layers_out[layer];
//        free(filter);
    }

    // Concatenation
//    layer_in = (float *) malloc(map_size[8] * depth_size[8] * 2 * sizeof(float));
//    concatenate(encoder_layers_out[7], Z_array, layer_in, map_size[8], depth_size[8]);

    // Decoder
    float * decoder_layers_out;
    float * skip;
    for (int layer = 1; layer >=0; layer--) {
        printf("Decoder Layer %d\n", layer);
        decoder_layers_out = (float *) malloc(map_size[layer] * depth_size[layer] * sizeof(float));
//        filter = (float *) malloc(FILTER_SIZE * depth_size[layer] * depth_size[layer - 1] * sizeof(float));
        skip = (float *) malloc(map_size[layer] * depth_size[layer] * sizeof(float));
//        fill_filter(filter, depth_size[layer], depth_size[layer - 1]);
//        fill_filter(filter, 2 - layer);
//        fill_dec(filter, 2-layer);
        filter = dec_w[1 - layer];
        deconv1d(layer_in, filter, decoder_layers_out, map_size[layer+1], depth_size[layer+1], depth_size[layer]);
        if (layer != 0) {
            fill_skip(skip, map_size[layer], depth_size[layer]);
            skip_add(encoder_layers_out[layer], skip, decoder_layers_out, map_size[layer],
                     depth_size[layer]);
        }
        layer_in = decoder_layers_out;
//        free(filter);
    }
}