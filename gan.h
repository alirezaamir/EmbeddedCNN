//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_BITS 26

#define MUL(x, y) (int)((((long)(x)*(long)(y)))>>NUM_FRACTION_BITS)

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

extern int* conv1d_w[3];
extern int* conv1d_b[3];
extern int* dense_w[2];
extern int* dense_b[2];
extern int* bn[12];
extern int input_array[];

#endif //EPILEPSYGAN_GAN_H
