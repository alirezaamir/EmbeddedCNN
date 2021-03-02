//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_BITS 28

#define MUL(x, y, shift) (int)((((long)(x)*(long)(y)))>>(shift))

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

#define LEAKY_RATIO 19660 // 314572 // 28: 80530636 // 30: 322122547

extern int Z_array[];
extern short* enc_w[8];
extern short* dec_w[8];
extern short* A_w[7];
extern short input_array[];

#endif //EPILEPSYGAN_GAN_H
