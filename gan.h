//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define NUM_FRACTION_BITS 14

#define MUL(x, y) (short)((((long)(x)*(long)(y)))>>NUM_FRACTION_BITS)

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[((n)*(filter_depth)+(k))*(filter_len)+(i)]

extern short* enc_w[9];
extern short* bias_w[9];
extern short* skip_w[4];
extern short* skip_bias[4];
extern short fc_weights_1[];
extern short fc_bias_1[];
extern short input_array[];

#endif //EPILEPSYGAN_GAN_H
