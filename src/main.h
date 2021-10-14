//
// Created by alireza on 1/7/21.
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "rt/rt_api.h"
#include "profile.h"
#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_DATA 10
#define NUM_FRACTION_CNV_FC 8
#define NUM_FRACTION_BN 5
#define NEG_INF (-(1<<14))

#define MUL(x, y, num) (short)(((int)(x)*(int)(y))>>(num))

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

// Printing output
#define RUN_PRINT_PROFILING
// #define PRINT_INPUT
// #define PRINT_PREDICTION
// #define PRINT_SUM
// #define PRINT_FC_OUT
// #define PRINT_CONV
// #define PRINT_BLOCK

extern int8_t* conv1d_w[3];
extern int8_t* conv1d_b[3];
extern int8_t* dense_w[2];
extern int8_t* dense_b[2];
extern int8_t* bn[12];
extern int16_t input_array[];

#endif //EPILEPSYGAN_GAN_H
