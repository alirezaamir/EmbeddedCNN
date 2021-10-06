/*
 * filter.h
 *
 * Copyright (c) 2018 Disi A
 * 
 * Author: Disi A
 * Email: adis@live.cn
 *  https://www.mathworks.com/matlabcentral/profile/authors/3734620-disi-a
 */
#ifndef filter_h
#define filter_h

#define MUL_INT(x, y) (int)((((long)(x)*(long)(y)))>>NUM_FRACTION_BITS)
#define FTR_PRECISION int

typedef struct {
    int n;
	FTR_PRECISION *A;
    FTR_PRECISION *d1;
    FTR_PRECISION *d2;
    FTR_PRECISION *d3;
    FTR_PRECISION *d4;
    FTR_PRECISION *w0;
    FTR_PRECISION *w1;
    FTR_PRECISION *w2;
    FTR_PRECISION *w3;
    FTR_PRECISION *w4;
} BWBandPass;

BWBandPass* create_bw_band_pass_filter(int order);

void free_bw_band_pass(BWBandPass* filter);


FTR_PRECISION bw_band_pass(BWBandPass* filter, FTR_PRECISION input, FTR_PRECISION output);

#if __cplusplus
}
#endif

#endif /* filter_h */