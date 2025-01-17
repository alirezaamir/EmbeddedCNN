#include "main.h"

RT_L2_DATA int16_t intermediate_map[256*128];
#ifdef PULP
    RT_L2_DATA rt_perf_t* perf;
#endif
#ifdef HEEP
    int kResultsIdx = 0;
#endif
// ===========================> Functions Prototype <===============================
void conv1d(const int16_t *data, const signed char *filter, int16_t *map_out, const signed char *bias, int32_t filter_size,
            int32_t input_len, int32_t input_depth, int32_t output_len, int32_t n_filter, int32_t strides, int32_t relu, int32_t padding);

void conv_max1d(const int16_t *data, const signed char *filter, int16_t *map_out, const signed char *bias, int32_t filter_size,
                int32_t input_len, int32_t input_depth, int32_t output_len, int32_t n_filter, int32_t strides, int32_t relu, int32_t padding, int32_t pool_size);

void batch_normalization(const int16_t *data, const signed char *gamma, const signed char *beta, const signed char *mean, const signed char *var,
                         int16_t *map_out, int32_t input_len, int32_t input_depth);

int16_t forward_propagation(int16_t *data, int16_t *intermediate);

// =================================================================================

int main()
{
#ifdef HEEP
    //heep_ResetStatusRegister();
#endif
#ifdef SERIAL_AVAILABLE
    printf("Input Array : %x\n", input_array[0]);
#endif

#ifdef PULP
    profile_start(perf);
#endif

    // int16_t* eeg_input = input_array;
#ifdef HEEP
//    #ifdef DATA_ACQUISITION
//        heep_fDmaCaptureFromAdc(
//            input_array, INPUT_LEN*sizeof(int16_t), CAPTURE_IDLE_CYCLES
//        );
//        heep_fClockgate(heep_Eventunit_kDmaIntBit);
//    #endif

	#ifdef DATA_ACQUISITION
	    // First capture
	    heep_ClockgatedDmaCaptureFromAdc(
	        input_array,
	        INPUT_LEN*sizeof(int16_t),
	        CAPTURE_IDLE_CYCLES
	    );
	heep_kResults[kResultsIdx++] = 10;
        // Clear the previous interrupts
        heep_Eventunit_ClearInterrupts(
            heep_Eventunit_kDmaIntBit | heep_Eventunit_kTimerIntBit
        );
	heep_kResults[kResultsIdx++] = 100;
	// Set a timer as watchdog
	int dim_seconds = 4;
        heep_StartTimer((dim_seconds + 1) * heep_kCpuFreq);
	heep_kResults[kResultsIdx++] = 1000;
        // Start capturing next window
        //heep_DmaCaptureFromAdc(
            // &ecg_3l[overlap][0],
            // NLEADS*(dim - overlap)*sizeof(int16_t),
        //    input_array,
        //    INPUT_LEN*sizeof(int16_t),
        //    CAPTURE_IDLE_CYCLES
        //);
	heep_kResults[kResultsIdx++] = 10000;
	//for (int heep_idx = -1; heep_idx < 90; heep_idx ++)
                //heep_kResults[kResultsIdx++] = heep_idx;
	#endif
#endif
	//for (int heep_idx = -1; heep_idx < 90; heep_idx ++)
	        //heep_kResults[kResultsIdx++] = heep_idx;
    heep_kResults[kResultsIdx++] = 100000;
    int16_t predict = forward_propagation(input_array, intermediate_map);

#ifdef PULP
    profile_stop(perf);
#endif

#ifdef PRINT_PREDICTION
    printf("Prediction : %d\n", predict);
#endif

#ifdef HEEP
    heep_kResults[kResultsIdx++] = predict + 100;
    heep_SetStatusRegister();

    #ifdef DATA_ACQUISITION
        #ifndef FAST_DATA_ACQUISITION
            // Clockgate until the transfer is complete
            uint32_t awokenBits = heep_Clockgate(
                heep_Eventunit_kDmaIntBit | heep_Eventunit_kTimerIntBit
            );
            if (!(awokenBits & heep_Eventunit_kDmaIntBit)) {
                // There was a problem with the adc
                // Do something
                //heep_kResults[kResultsIdx++] = -2;
                //heep_kResults[kResultsIdx++] = awokenBits;
                exit(2);
            }
        #endif
    #endif
#endif


    return 0;
}


void conv1d(const int16_t *data, const signed char *filter, int16_t *map_out, const signed char *bias, const int32_t filter_size,
            const int32_t input_len, const int32_t input_depth, const int32_t output_len, const int32_t n_filter, const int32_t strides,
            const int32_t relu, const int32_t padding) {
    int32_t sum;
    int32_t mult;
    int32_t pad = padding ? filter_size / 2 : 0;
    for (int32_t w_n = 0; w_n < n_filter; w_n++) {
        for (int32_t start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (int32_t w_i = 0; w_i < filter_size; w_i++) {
                for (int32_t w_j = 0; w_j < input_depth; w_j++) {
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
                #ifdef PRINT_OVERFLOW
                    printf("Overflow %d\n", sum);
                #endif
                #ifdef HEEP
                    //heep_kResults[1000];
                #endif
                    sum = (1<<15) -1;
            }else if (sum < -(1 << 15)){
                #ifdef PRINT_OVERFLOW
                    printf("Overflow %d\n", sum);
                #endif
                #ifdef HEEP
                    //heep_kResults[1000];
                #endif
                    sum = -(1<<15) +1;
            }
            mem2d(map_out, output_len, w_n, start_index / strides) = (int16_t) sum;
            #ifdef SERIAL_AVAILABLE
                printf("FC out %d : %x\n", w_n, sum);
            #endif
            #ifdef HEEP
                //heep_kResults[kResultsIdx++] = sum;
            #endif
        }
    }
}


void conv_max1d(const int16_t *data, const signed char *filter, int16_t *map_out, const signed char *bias, const int32_t filter_size,
                const int32_t input_len, const int32_t input_depth, const int32_t output_len, const int32_t n_filter, const int32_t strides,
                const int32_t relu, const int32_t padding, const int32_t pool_size) {
    int32_t sum;
    int32_t mult;
    int32_t pad = padding ? filter_size / 2 : 0;
    for (int32_t w_n = 0; w_n < n_filter; w_n++) {
        int32_t maximum = NEG_INF;
        for (int32_t start_index = 0; start_index < input_len; start_index += strides) {
            sum = 0;
            for (int32_t w_i = 0; w_i < filter_size; w_i++) {
                for (int32_t w_j = 0; w_j < input_depth; w_j++) {
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
                #ifdef PRINT_OVERFLOW
                    printf("Overflow %d\n", sum);
                #endif
                    sum = (1<<15) -1;
            }else if (sum < -(1 << 15)){
                #ifdef PRINT_OVERFLOW
                    printf("Overflow %d\n", sum);
                #endif
                sum = -(1<<15) +1;
            }
            if (sum > maximum) {
                maximum = sum;
            }
            if (start_index % pool_size == pool_size-1) {
                mem2d(map_out, output_len, w_n, start_index / (strides * pool_size)) = (int16_t) maximum;
                maximum = NEG_INF;
            }
        }
    }
}

void batch_normalization(const int16_t *data, const signed char *gamma, const signed char *beta, const signed char *mean, const signed char *var,
                         int16_t *map_out, const int32_t input_len, const int32_t input_depth) {
    for (int32_t start_index = 0; start_index < input_len; start_index++) {
        for (int32_t w_j = 0; w_j < input_depth; w_j++) {
            int16_t normalized = mem2d(data, input_len, w_j, start_index) -
            ((int16_t) mean[w_j] << (NUM_FRACTION_DATA - NUM_FRACTION_BN));
            int16_t standardized = MUL(normalized, var[w_j], NUM_FRACTION_BN);
            int16_t new_standardized = MUL(standardized, gamma[w_j], NUM_FRACTION_BN);
            mem2d(map_out, input_len, w_j, start_index) =
                    (int16_t) (new_standardized + ((int16_t) beta[w_j] << (NUM_FRACTION_DATA - NUM_FRACTION_BN)));
        }
    }
}

void relu(const int16_t *data, int16_t *map_out, int32_t input_len) {
    for (int32_t i = 0; i < input_len; i++)
        map_out[i] = (data[i] < 0) ? 0 : data[i];
}

void conv_block(int32_t block, int16_t* layer_in, int16_t* conv1d_out){
    int32_t depth_size[6] = {23, 128, 128, 128, 100, 2};
    int32_t map_size[6] = {1024, 256, 64, 16, 1, 1};
    signed char* filter = conv1d_w[block];
    signed char* bias = conv1d_b[block];
#ifdef SERIAL_AVAILABLE
    printf("Block %d\n", block);
#endif
    conv_max1d(layer_in, filter, conv1d_out, bias, 3, map_size[block], depth_size[block], map_size[block+1],
               depth_size[block + 1], 1, 0, 1, 4);

    batch_normalization(conv1d_out, bn[block * 4], bn[block * 4 + 1], bn[block * 4 + 2], bn[block * 4 + 3],
                        conv1d_out, map_size[block+1], depth_size[block + 1]);

    relu(conv1d_out, conv1d_out, map_size[block+1] * depth_size[block + 1]);
}

int16_t forward_propagation(int16_t *data, int16_t *intermediate) {
    int32_t fc_depth_size[6] = {128, 100, 2};
    int32_t fc_map_size[6] = {16, 1, 1};
    int16_t* intermediate_map0 = intermediate;
    int16_t* intermediate_map1 = data;
    //heep_kResults[kResultsIdx++] = 11;

    //  ************  BLOCK 0  ************ //
    int16_t *layer_out = intermediate_map0;
    int16_t *layer_in = intermediate_map1;
    conv_block(0, layer_in, layer_out);
    //heep_kResults[kResultsIdx++] = 110;

    //  ************  BLOCK 1  ************ //
    layer_out = intermediate_map1;
    layer_in = intermediate_map0;
    conv_block(1, layer_in, layer_out);
    //heep_kResults[kResultsIdx++] = 1110;

    //  ************  BLOCK 2  ************ //
    layer_out = intermediate_map0;
    layer_in = intermediate_map1;
    conv_block(2, layer_in, layer_out);
    //heep_kResults[kResultsIdx++] = 11110;

    //  ************  FC 0  ************ //
    layer_out = intermediate_map1;
    layer_in = intermediate_map0;
    conv1d(layer_in, dense_w[0], layer_out, dense_b[0], fc_map_size[0],fc_map_size[0],
           fc_depth_size[0], fc_map_size[1], fc_depth_size[1], fc_map_size[0],
           1, 0);
    for(int i=0; i< fc_depth_size[1]; i++)
	    heep_kResults[kResultsIdx++] = intermediate_map1[i];

    //  ************  FC 1  ************ //
    layer_out = intermediate_map0;
    layer_in = intermediate_map1;
    conv1d(layer_in, dense_w[1], layer_out, dense_b[1], fc_map_size[1],fc_map_size[1],
           fc_depth_size[1], fc_map_size[2], fc_depth_size[2], fc_map_size[1],
           0, 0);
    heep_kResults[kResultsIdx++] = 111100;

    if (layer_out[0] > layer_out[1])
        return 0;
    else
        return 1;
}
