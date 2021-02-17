//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_BITS 30

#define MUL(x, y) (int)((((long)(x)*(long)(y)))>>NUM_FRACTION_BITS)

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

#define LEAKY_RATIO 322122547
#define INV_LEAKY_RATIO 3579139413

extern int Z_array[];
extern int* enc_w[8];
extern int* dec_w[8];
extern int* A_w[7];
extern int input_array[];

#endif //EPILEPSYGAN_GAN_H
