//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_BITS 30

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define MUL(x, y) ((((long)(x)*(long)(y)))>>NUM_FRACTION_BITS)

// data is a 2-dimensional matrix implemented as 1-dimensional array
// data[y][x] == data[ y * q + x ]

#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

#define LEAKY_RATIO 322122547 // 32bits: 1288490188 // 40 bits: 329853488332 // 30bits: 322122547
#define INV_LEAKY_RATIO 3579139413// 32bits: 14316557653 // 40 bits: 3665038758886 // 30 bits: 3579139413

extern long Z_array[];
extern long* enc_w[8];
extern long* dec_w[8];
extern long* A_w[7];
extern long input_array[];

#endif //EPILEPSYGAN_GAN_H
