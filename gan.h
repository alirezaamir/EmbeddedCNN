//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_BITS 10

#define MUL(x, y) (short)(((int)(x)*(int)(y))>>NUM_FRACTION_BITS)

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

extern short* conv1d_w[3];
extern short* conv1d_b[3];
extern short* dense_w[2];
extern short* dense_b[2];
extern short* bn[12];
extern short input_array[];

#endif //EPILEPSYGAN_GAN_H
