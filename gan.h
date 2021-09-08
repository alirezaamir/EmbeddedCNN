//
// Created by alireza on 1/7/21.
//
#include <stdio.h>
#include <stdlib.h>
#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_DATA 10
#define NUM_FRACTION_CNV_FC 8
#define NUM_FRACTION_BN 5
#define NEG_INF (-(1<<14))

#define MUL(x, y, num) (short)(((int)(x)*(int)(y))>>(num))

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

extern __int8_t* conv1d_w[3];
extern __int8_t* conv1d_b[3];
extern __int8_t* dense_w[2];
extern __int8_t* dense_b[2];
extern __int8_t* bn[12];
extern __int16_t input_array[];

#endif //EPILEPSYGAN_GAN_H
