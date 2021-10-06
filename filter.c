/*
 * filter.c
 *
 * Copyright (c) 2018 Disi A
 * 
 * Author: Disi A
 * Email: adis@live.cn
 *  https://www.mathworks.com/matlabcentral/profile/authors/3734620-disi-a
 */
#include <stdlib.h>
#include <stdio.h>
#include "filter.h"
#include "gan.h"

BWBandPass* create_bw_band_pass_filter(int order){
    BWBandPass* filter = (BWBandPass *) malloc(sizeof(BWBandPass));
    filter -> n = order/4;
    filter -> A = (FTR_PRECISION *)malloc(filter -> n*sizeof(FTR_PRECISION));
    filter -> d1 = (FTR_PRECISION *)malloc(filter -> n*sizeof(FTR_PRECISION));
    filter -> d2 = (FTR_PRECISION *)malloc(filter -> n*sizeof(FTR_PRECISION));
    filter -> d3 = (FTR_PRECISION *)malloc(filter -> n*sizeof(FTR_PRECISION));
    filter -> d4 = (FTR_PRECISION *)malloc(filter -> n*sizeof(FTR_PRECISION));

    filter -> w0 = (FTR_PRECISION *)calloc(filter -> n, sizeof(FTR_PRECISION));
    filter -> w1 = (FTR_PRECISION *)calloc(filter -> n, sizeof(FTR_PRECISION));
    filter -> w2 = (FTR_PRECISION *)calloc(filter -> n, sizeof(FTR_PRECISION));
    filter -> w3 = (FTR_PRECISION *)calloc(filter -> n, sizeof(FTR_PRECISION));
    filter -> w4 = (FTR_PRECISION *)calloc(filter -> n, sizeof(FTR_PRECISION));

    filter->A[0] = 1116;
    filter->d1[0] = 22085;
    filter->d2[0] = -21870;
    filter->d3[0] = 10144;
    filter->d4[0] = -2168;
    return filter;
}

void free_bw_band_pass(BWBandPass* filter){
    free(filter -> A);
    free(filter -> d1);
    free(filter -> d2);
    free(filter -> d3);
    free(filter -> d4);
    free(filter -> w0);
    free(filter -> w1);
    free(filter -> w2);
    free(filter -> w3);
    free(filter -> w4);
    free(filter);
}
FTR_PRECISION bw_band_pass(BWBandPass* filter, FTR_PRECISION x, FTR_PRECISION y){
    int i;
    for(i=0; i<filter->n; ++i){
        filter->w0[i] = MUL_INT(filter->d1[i], filter->w1[i]) +
                MUL_INT(filter->d2[i], filter->w2[i]) +
                MUL_INT(filter->d3[i], filter->w3[i]) +
                MUL_INT(filter->d4[i], filter->w4[i]) + x;
        y = MUL_INT (filter->A[i], (filter->w0[i] - (filter->w2[i] << 1) + filter->w4[i]));
        filter->w4[i] = filter->w3[i];
        filter->w3[i] = filter->w2[i];
        filter->w2[i] = filter->w1[i];
        filter->w1[i] = filter->w0[i];
    }
    return y;
}