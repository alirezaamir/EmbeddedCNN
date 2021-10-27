//
// Created by alireza on 1/7/21.
//
#include <stdio.h>
#include <stdlib.h>

#define HEEP
#define INPUT_LEN (23 * 1024)

#ifdef HEEP
#include "heep_riscv_sdk.h"
#endif

#ifdef PULP
    #include "rt/rt_api.h"
    #include "profile.h"
#else
    #define RT_L2_DATA
#endif
#include <stdint.h>
//#include "rt/rt_api.h"

//====== DEFINE DATA SOURCE =====//
#ifdef HEEP
#define DATA_ACQUISITION
	// #define FAST_DATA_ACQUISITION
#endif

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_DATA 10
#define NUM_FRACTION_CNV_FC 8
#define NUM_FRACTION_BN 5
#define NEG_INF (-(1<<14))

#define MUL(x, y, num) (short)(((int)(x)*(int)(y))>>(num))

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

#ifdef HEEP
    #ifndef DATA_ACQUISITION
        extern int16_t input_array[];
    #else
        #define SAMPLING_FREQ 256
        #define CAPTURE_IDLE_CYCLES (heep_kCpuFreq * sizeof(int32_t) /  sizeof(int16_t) \
                                      / (23 * SAMPLING_FREQ))
        int16_t input_array[INPUT_LEN] __attribute__((aligned(4))) = {5};
    #endif
#else
    extern int16_t input_array[INPUT_LEN];
#endif

// Printing output
#define RUN_PRINT_PROFILING
// #define PRINT_INPUT
// #define PRINT_PREDICTION
// #define PRINT_SUM
// #define PRINT_FC_OUT
// #define PRINT_OVERFLOW
// #define PRINT_BLOCK

extern int8_t* conv1d_w[3];
extern int8_t* conv1d_b[3];
extern int8_t* dense_w[2];
extern int8_t* dense_b[2];
extern int8_t* bn[12];

#endif //EPILEPSYGAN_GAN_H
