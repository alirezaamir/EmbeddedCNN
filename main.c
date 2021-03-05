#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "gan.h"
#include <time.h>

#define FILTER_SIZE 31
#define NUM_THREADS 4


// ===========================> Functions Prototype <===============================
void conv1d(const int *data, const short *filter, int *map_out, int *map_skip,
            int input_len, int input_depth, int n_filter, int shift);

void deconv1d(int *data, const short *filter, int *map_out,
              int input_len, int input_depth, int n_filter, int shift);

void skip_add(int *data, const short *filter, int *map_out,
              int input_len, int input_depth, int shift);

void forward_propagation(int *data);
// =================================================================================

int main() {
    omp_set_num_threads(NUM_THREADS);
    // allocate memory in CPU for calculation
    int *eeg_input;
    eeg_input = input_array;
    forward_propagation(eeg_input);
    return 0;
}


void conv1d(const int *data, const short *filter, int *map_out, int *map_skip,
            const int input_len, const int input_depth, const int n_filter, const int shift_bits) {
#pragma omp parallel for
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 1; start_index < input_len; start_index += 2) {
            int sum = 0;
            for (int w_i = 0; w_i < FILTER_SIZE; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - 15 < input_len && start_index + w_i - 15 > -1) {
                        sum += MUL(mem3d(filter, FILTER_SIZE, input_depth, w_n, w_j, w_i),
                                   mem2d(data, input_len, w_j, start_index + w_i - 15), shift_bits);
                    }
                }
            }
            mem2d(map_skip, input_len >> 1, w_n, start_index >> 1) = sum;
            if (sum < 0)
                sum = MUL(sum, LEAKY_RATIO, 16); // Leaky Relu
            mem2d(map_out, input_len >> 1, w_n, start_index >> 1) = sum;
        }
    }
}

void deconv1d(int *data, const short *filter, int *map_out,
              int input_len, const int input_depth, const int n_filter, const int shift_bits) {
    // Upsampling
    input_len *= 2; // Update input len to the upsampled one
    int *upsampled = (int *) malloc(input_len * input_depth * sizeof(int));
    for (int w_j = 0; w_j < input_depth; w_j++)
        for (int i = 0; i < input_len; i++) {
            if (i % 2 == 1) {
                mem2d(upsampled, input_len, w_j, i) = mem2d(data, input_len >> 1, w_j, i >> 1);
            } else
                upsampled[i] = 0;
        }

    // Convolution
#pragma omp parallel for
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index < input_len; start_index++) {
            int sum = 0;

            for (int w_j = 0; w_j < input_depth; w_j++) {
                for (int w_i = 0; w_i < FILTER_SIZE; w_i++) {
                    if (start_index + w_i - 15 < input_len && start_index + w_i - 15 > -1) {
                        sum += MUL(mem3d(filter, FILTER_SIZE, n_filter, w_j, w_n, FILTER_SIZE - w_i - 1),
                                   mem2d(upsampled, input_len, w_j, start_index + w_i - 15), shift_bits);
                    }
                }
            }
            if (sum < 0 && n_filter != 1)
                sum = MUL(sum, LEAKY_RATIO, 16); // Leaky Relu
            mem2d(map_out, input_len, w_n, start_index) = sum;
        }
    }
}


void concatenate(const int *data, const int *z, int *map_out,
                 int input_len, int input_depth) {
    for (int w_j = 0; w_j < input_depth; w_j++) {
        for (int w_i = 0; w_i < input_len; w_i++) {
            mem2d(map_out, input_len, w_j, w_i) = 0;
//            mem2d(map_out, input_len, w_j + input_depth, w_i) = mem2d(z, input_len, w_j, w_i);
            mem2d(map_out, input_len, w_j + input_depth, w_i) = mem2d(data, input_len, w_j, w_i);
        }
    }
}


void skip_add(int *data, const short *filter, int *map_out,
              int input_len, const int input_depth, const int shift_bits) {
    for (int w_j = 0; w_j < input_depth; w_j++) {
        for (int w_i = 0; w_i < input_len; w_i++) {
            int add = MUL(mem2d(data, input_len, w_j, w_i), mem2d(filter, input_len, w_j, w_i), shift_bits);
            add += mem2d(data, input_len, w_j, w_i);
//            if (add <0)
//                add = MUL(add,  INV_LEAKY_RATIO, NUM_FRACTION_BITS);
            mem2d(map_out, input_len, w_j, w_i) += add;
        }
    }
}


void save_file(int *data, char *filename, int input_len) {
    FILE *fp;
    fp = fopen(filename, "w");
    for (int wi = 0; wi < input_len; wi++)
        fprintf(fp, "%d\n", data[wi]);

    fclose(fp);
}

void forward_propagation(int *data) {
    int *encoder_layers_out[8] = {0};
    const int depth_size[9] = {1, 64, 64, 128, 128, 256, 256, 512, 1024};
    const int map_size[9] = {2048, 1024, 512, 256, 128, 64, 32, 16, 8};

    const int shift_bits_enc[8] = {18, 21, 21, 21, 22, 22, 22, 23};
    const int shift_bits_dec[8] = {23, 22, 21, 20, 20, 20, 19, 18};
    const int shift_bits_skp[7] = {20, 20, 20, 20, 20, 20, 20};

    short *filter;
    int *layer_in = (int *) data;
    int *layer_out;

    double time_spent = 0.0;

    clock_t begin = clock();
    // Encoder
    for (int layer = 0; layer <= 7; layer++) {
        encoder_layers_out[layer] = (int *) malloc(map_size[layer + 1] * depth_size[layer + 1] * sizeof(int));
        layer_out = (int *) malloc(map_size[layer + 1] * depth_size[layer + 1] * sizeof(int));
        filter = enc_w[layer];
        conv1d(layer_in, filter, layer_out, encoder_layers_out[layer],
               map_size[layer], depth_size[layer], depth_size[layer + 1], shift_bits_enc[layer]);
        layer_in = layer_out;

        char out_filename[17];
        sprintf(out_filename, "fxp_enc_out%d.txt", layer);
        save_file(layer_out, out_filename, map_size[layer + 1] * depth_size[layer + 1]);
    }

    // Concatenation
    layer_in = (int *) malloc(map_size[8] * depth_size[8] * 2 * sizeof(int));
    concatenate(encoder_layers_out[7], Z_array, layer_in, map_size[8], depth_size[8]);

    // Decoder
    int *decoder_layers_out;
    short *skip;
    for (int layer = 7; layer >= 0; layer--) {
        decoder_layers_out = (int *) malloc(map_size[layer] * depth_size[layer] * sizeof(int));
        filter = dec_w[7 - layer];
        deconv1d(layer_in, filter, decoder_layers_out, map_size[layer + 1],
                 depth_size[layer + 1] * (layer == 7 ? 2 : 1), depth_size[layer],
                 shift_bits_dec[7 - layer]);
        if (layer != 0) {
            skip = A_w[layer - 1];
            skip_add(encoder_layers_out[layer - 1], skip, decoder_layers_out, map_size[layer], depth_size[layer],
                     shift_bits_skp[layer - 1]);
        }
        layer_in = decoder_layers_out;

        char out_filename[17];
        sprintf(out_filename, "fxp_dec_out%d.txt", 7 - layer);
        save_file(decoder_layers_out, out_filename, map_size[layer] * depth_size[layer]);
    }
    clock_t end = clock();

    // calculate elapsed time by finding difference (end - begin) and
    // dividing the difference by CLOCKS_PER_SEC to convert to seconds
    time_spent += (double) (end - begin) / CLOCKS_PER_SEC;
    printf("Time elapsed is %f seconds", time_spent);

//    save_file(layer_in, "fxp_output.txt", map_size[0]);
}
