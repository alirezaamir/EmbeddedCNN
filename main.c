#include <stdio.h>
#include <stdlib.h>
#include "gan.h"
//#include "gan.c"

#define FILTER_SIZE 31


// ===========================> Functions Prototype <===============================
void conv1d(const int *data, const int *filter, int *map_out,
            int input_len, int input_depth, int n_filter);

void deconv1d(int *data, const int *filter, int *map_out,
              int input_len, int input_depth, int n_filter);

void skip_add(int *data, int *filter, int *map_out,
              int input_len, int input_depth);

void forward_propagation(int *data);
// =================================================================================

int main() {
    printf("EpilepsyGAN!\n");

    // allocate memory in CPU for calculation
    int *eeg_input;
    const int in_len = 2048;
    // fill a, b matrices with random values between -16.0f and 16.0f
//    srand(0);
//    fill_map(eeg_input, in_len, 1);
    eeg_input = input_array;
    forward_propagation(eeg_input);
//    free(eeg_input);
    return 0;
}


void conv1d(const int *data, const int *filter, int *map_out,
            const int input_len, const int input_depth, const int n_filter) {
    int sum;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 1; start_index < input_len; start_index += 2) {
            sum = 0;
            for (int w_i = 0; w_i < FILTER_SIZE; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    if (start_index + w_i - 15 < input_len && start_index + w_i - 15 > -1) {
                        sum += MUL(mem3d(filter, FILTER_SIZE, input_depth, w_n, w_j, w_i),
                               mem2d(data, input_len, w_j, start_index + w_i - 15));
//                        printf("weight %d,%d,%d X in %d, %d: %f X %f = %f\n",
//                               w_i, w_j, w_n, start_index + w_i - 15,  w_j,
//                               mem3d(filter, FILTER_SIZE, input_depth, w_n, w_j, w_i),
//                               mem2d(data, input_len, w_j, start_index + w_i - 15),
//                               sum);
                    }
                }
            }
            if (sum < 0)
                sum = MUL(sum, LEAKY_RATIO); // Leaky Relu
            mem2d(map_out, input_len >> 1, w_n, start_index >> 1) = sum;
//            printf("Wrote %f in %d, %d\n", sum, start_index >> 1, w_n );
//            if (input_depth > 1)
//                getchar();
        }
    }
}

void deconv1d(int *data, const int *filter, int *map_out,
              int input_len, const int input_depth, const int n_filter) {
    // Upsampling
    input_len *= 2; // Update input len to the upsampled one
    int *upsampled = (int *) malloc(input_len * input_depth * sizeof(int));
    for (int w_j= 0; w_j< input_depth; w_j ++)
        for (int i = 0; i < input_len; i++) {
            if (i % 2 == 1) {
                mem2d(upsampled, input_len, w_j, i) = mem2d(data, input_len >> 1, w_j, i >> 1);
            }
            else
                upsampled[i] = 0;
        }
//    free(data);

    // Convolution
    int sum;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index < input_len; start_index++) {
            sum = 0;

            for (int w_j = 0; w_j < input_depth; w_j++) {
                for (int w_i = 0; w_i < FILTER_SIZE; w_i++) {
                    if (start_index + w_i - 15 < input_len && start_index + w_i - 15 > -1) {
                        sum += MUL(mem3d(filter, FILTER_SIZE, n_filter, w_j, w_n, FILTER_SIZE - w_i - 1) ,
                               mem2d(upsampled, input_len, w_j, start_index + w_i - 15));
//                        printf("weight %d,%d,%d X in %d, %d: %f X %f = %f\n",
//                               FILTER_SIZE - w_i - 1, w_j, w_n, start_index + w_i - 15, w_j,
//                               mem3d(filter, FILTER_SIZE, n_filter, w_j, w_n, FILTER_SIZE - w_i - 1),
//                               mem2d(upsampled, input_len, w_j, start_index + w_i - 15),
//                               sum);
                    }
                }
            }
            if (sum < 0 && n_filter != 1)
                sum = MUL(sum, LEAKY_RATIO); // Leaky Relu
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


void skip_add(int *data, int *filter, int *map_out,
              int input_len, const int input_depth) {
    for (int w_j = 0; w_j < input_depth; w_j++) {
        for (int w_i = 0; w_i < input_len; w_i++) {
//            printf("%d, %d\n", w_j, w_i);
//            printf("enc: %f\n", mem2d(data, input_len, w_j, w_i));
//            printf("A: %f\n", mem2d(filter, input_len, w_j, w_i));
//            printf("dec: %f\n", mem2d(map_out, input_len, w_j, w_i));
            int add = MUL(mem2d(data, input_len, w_j, w_i), mem2d(filter, input_len, w_j, w_i));
            if (add <0)
                add = MUL(add,  INV_LEAKY_RATIO);
            mem2d(map_out, input_len, w_j, w_i) += add;
//            getchar();
        }
    }
//    free(data);
//    free(filter);
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
    int depth_size[9] = {1, 64, 64, 128, 128, 256, 256, 512, 1024};
    int map_size[9] = {2048, 1024, 512, 256, 128, 64, 32, 16, 8};
//    float enc_ratio[8] = {1.2602388f, 0.2659299f, 0.1959376f, 0.13729893f, 0.1064626f, 0.0913380f, 0.08377741f, 0.05075028f};
//    float dec_ratio[8] = {0.06047365f, 0.06849701f, 0.19342509f, 0.3388827f, 0.4225882f, 0.4932240f, 0.5388081f, 2.3255081f};

    int *filter;
    int *layer_in = (int *) data;
    save_file(layer_in, "input.txt", map_size[0]);

    // Encoder
    for (int layer = 0; layer <= 7; layer++) {
        printf("Encoder Layer %d\n", layer);
        encoder_layers_out[layer] = (int *) malloc(map_size[layer + 1] * depth_size[layer + 1] * sizeof(int));
        filter = enc_w[layer];
        conv1d(layer_in, filter, encoder_layers_out[layer], map_size[layer], depth_size[layer],
               depth_size[layer + 1]);
        layer_in = encoder_layers_out[layer];
//        free(filter);
        char out_filename[17];
        sprintf(out_filename, "fxd_enc_out%d.txt", layer);
        save_file(encoder_layers_out[layer], out_filename, map_size[layer + 1] * depth_size[layer + 1]);
    }

    // Concatenation
    layer_in = (int *) malloc(map_size[8] * depth_size[8] * 2 * sizeof(int));
    concatenate(encoder_layers_out[7], Z_array, layer_in, map_size[8], depth_size[8]);
    save_file(layer_in, "fxd_concatenate.txt", map_size[8] * depth_size[8] * 2 );

    // Decoder
    int *decoder_layers_out;
    int *skip;
    for (int layer = 7; layer >= 0; layer--) {
        printf("Decoder Layer %d\n", layer);
        decoder_layers_out = (int *) malloc(map_size[layer] * depth_size[layer] * sizeof(int));
        filter = dec_w[7 - layer];
        deconv1d(layer_in, filter, decoder_layers_out, map_size[layer + 1],
                 depth_size[layer + 1] * (layer == 7 ? 2 :1), depth_size[layer]);
        if (layer != 0) {
            skip = A_w[layer - 1];
            skip_add(encoder_layers_out[layer-1], skip, decoder_layers_out, map_size[layer], depth_size[layer]);
        }
        layer_in = decoder_layers_out;
        char out_filename[17];
        sprintf(out_filename, "fxd_dec_out%d.txt", 7 - layer);
        save_file(decoder_layers_out, out_filename, map_size[layer] * depth_size[layer]);
//        free(filter);
    }

    save_file(layer_in, "fxd_output.txt", map_size[0]);
}
