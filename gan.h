//
// Created by alireza on 1/7/21.
//

#ifndef EPILEPSYGAN_GAN_H
#define EPILEPSYGAN_GAN_H

#define mem2d(data,data_len,j,i)   data[((j)*(data_len))+(i)]
// data is a 2-dimensional matrix implemented as 1-dimensional array
// data[y][x] == data[ y * q + x ]

#define mem3d(filter,filter_len,filter_depth,n,k,i)   filter[(n*filter_depth+k)*filter_len+i]

float* get_z_array(void);
extern float Z_array[];

#endif //EPILEPSYGAN_GAN_H
