#include <stdio.h>
#include <stdlib.h>
#include "gan.h"


// ===========================> Functions Prototype <===============================
void conv1d(const short *data, const short *filter, short *map_out, const short *bias, int filter_size,
            int input_len, int input_depth, int output_len, int n_filter, int strides, int relu);

void skip_add(const short *data, short *map_out, int input_len);

short forward_propagation(short *data);
short add_overflow_free(int var0, short var1);
void save_file(short *data, char *filename, int input_len) ;
void pre_processing(short *data, int input_len);
// =================================================================================

int main() {
    // allocate memory in CPU for calculation
    short *ecg_input;
    ecg_input = input_array;
    short prediction[1568];
    for (int sample = 0; sample < 1568; sample++) {
//        pre_processing(ecg_input + sample * 768, 768);
        short predict = forward_propagation(ecg_input + sample * 768);
        printf("%d: %d\n", sample, predict);
        prediction[sample] = predict;
    }
    save_file(prediction, "1568_short.txt", 1568);
    return 0;
}


void pre_processing(short* data, int input_len){
    BWBandPass * filter = create_bw_band_pass_filter(6, 256, 0.67f, 40);
    for (int i=0; i<input_len; i++)
        printf("Input %d, Output %f\n", data[i], bw_band_pass(filter, (float)data[i]));

}

void conv1d(const short *data, const short *filter, short *map_out,const short *bias, const int filter_size,
            const int input_len, const int input_depth, const int output_len, const int n_filter, const int strides, const int relu) {
    int sum;
    int mult;
    int total=0;
    for (int w_n = 0; w_n < n_filter; w_n++) {
        for (int start_index = 0; start_index <= input_len-filter_size; start_index += strides) {
            sum = 0;
            for (int w_i = 0; w_i < filter_size; w_i++) {
                for (int w_j = 0; w_j < input_depth; w_j++) {
                    mult = MUL(mem3d(filter, filter_size, input_depth, w_n, w_j, w_i),
                               mem2d(data, input_len, w_j, start_index + w_i));
                    total += 1;
                    sum += mult;
                }
            }
            sum = add_overflow_free(sum, bias[w_n]);
            total += 1;
            if (sum < 0 && relu)
                sum = 0; // Relu
            mem2d(map_out, output_len, w_n, start_index/strides) = (short) sum;
        }
    }
//    printf("MAC %d\n", total);
}

void max1d(const short *data, short *map_out, const int input_len, const int input_depth,
           const int pool_size, const int strides) {
    int out_len = (input_len - pool_size / 2) / strides;
    for (int start_index = 0; start_index < input_len - pool_size; start_index += strides) {
        for (int w_j = 0; w_j < input_depth; w_j++) {
            short maximum = mem2d(data, input_len, w_j, start_index);
            for (int w_i = 1; w_i < pool_size; w_i++) {
                if (mem2d(data, input_len, w_j, start_index + w_i) > maximum)
                    maximum = mem2d(data, input_len, w_j, start_index + w_i);
            }
            mem2d(map_out, out_len, w_j, start_index / strides) = maximum;
        }
    }
}

void skip_add(const short *data, short *map_out, int input_len) {
    for (int i = 0; i < input_len; i++){
        map_out[i] = add_overflow_free(map_out[i], data[i]);
    }

}

short add_overflow_free(int var0, short var1){
    int sum = var0 + var1;
    if (sum > (1<<15)){
//        printf("Overflow %d\n", sum);
        return (1<<15) -1;
    }else if (sum < -(1<< 15)){
//        printf("Overflow %d\n", sum);
        return -(1<<15) +1;
    }
    else{
        return (short) sum;
    }
}


void save_file(short *data, char *filename, int input_len) {
    FILE *fp;
    fp = fopen(filename, "w");
    for (int wi = 0; wi < input_len; wi++)
        fprintf(fp, "%d\n", data[wi]);

    fclose(fp);
}

short forward_propagation(short *data) {
    int depth_size[11] = {1, 64, 64, 64, 64, 128, 128, 256, 256, 512, 512};
    int map_size[11] = {768, 381, 190,  94, 92, 45, 43, 21, 19, 9, 7};

    short *filter;
    short *bias;
    short *layer_in = (short *) data;
    save_file(data, "input_data.txt", 768);

    //
    short *first_conv = (short *) malloc(map_size[1] * depth_size[1] * sizeof(short));
    filter = enc_w[0];
    bias = bias_w[0];
    conv1d(layer_in, filter, first_conv, bias, 7, map_size[0], depth_size[0], map_size[1],
           depth_size[1], 2, 1);
    short *max_pool = (short *) malloc(map_size[2] * depth_size[2] * sizeof(short));
    max1d(first_conv, max_pool, map_size[1], depth_size[1], 3, 2);
    save_file(first_conv, "first_conv.txt", map_size[1] * depth_size[1]);
    //free(first_conv);

    layer_in = max_pool;
    for (int block = 0; block < 4; block++) {
        int index = block * 2 + 2;
        short* conv1d_1 = (short *) malloc(map_size[index+1] * depth_size[index+1] * sizeof(short));
        filter = enc_w[index-1];
        bias = bias_w[index-1];
        conv1d(layer_in, filter, conv1d_1, bias, 3, map_size[index], depth_size[index], map_size[index+1],
               depth_size[index + 1], 2, 1);
        char out_filename[17];
        sprintf(out_filename, "conv1d_b%d_1.txt", block);
        save_file(conv1d_1, out_filename, map_size[index + 1] * depth_size[index + 1]);

        short *conv1d_2 = (short *) malloc(map_size[index+2] * depth_size[index+2] * sizeof(short));
        filter = enc_w[index];
        bias = bias_w[index];
        conv1d(conv1d_1, filter, conv1d_2, bias, 3, map_size[index + 1], depth_size[index+1], map_size[index + 2],
               depth_size[index + 2], 1, 1);
        sprintf(out_filename, "conv1d_b%d_2.txt", block);
        save_file(conv1d_2, out_filename, map_size[index + 2] * depth_size[index + 2]);

        short* conv1d_skip = (short *) malloc(map_size[index + 2] * depth_size[index + 2] * sizeof(short));
        filter = skip_w[block];
        bias = skip_bias[block];
        conv1d(layer_in, filter, conv1d_skip, bias, 7, map_size[index], depth_size[index], map_size[index + 2],
               depth_size[index + 2], 2, 1);

        skip_add(conv1d_2, conv1d_skip, map_size[index + 2] * depth_size[index + 2]);
        sprintf(out_filename, "conv1d_b%d_3.txt", block);
        save_file(conv1d_skip, out_filename, map_size[index + 2] * depth_size[index + 2]);
        layer_in = conv1d_skip;
    }
    short *fully_connected = malloc(2 * sizeof(short));
    conv1d(layer_in, fc_weights_1, fully_connected, fc_bias_1, 7, map_size[10], depth_size[10],
           1, 2, 1, 0);

    save_file(fully_connected, "fxp_output.txt", 2);

    if (fully_connected[0] > fully_connected[1])
        return 0;
    else
        return 1;
}
