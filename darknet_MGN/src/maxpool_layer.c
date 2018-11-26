#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size_w, int size_h, int stride_w, int stride_h, int padding)
{
    maxpool_layer l;
    memset(&l, 0, sizeof(maxpool_layer));

    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + 2*padding - size_w)/stride_w + 1;
    l.out_h = (h + 2*padding - size_h)/stride_h + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size_w = size_w;
    l.size_h = size_h;
    l.stride_w = stride_w;
    l.stride_h = stride_h;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes  = (int *)calloc(output_size, sizeof(int));
    l.output   = (float *)calloc(output_size, sizeof(float));
    l.delta    = (float *)calloc(output_size, sizeof(float));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    #ifdef GPU
    l.forward_gpu  = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu  = cuda_make_int_array(0, output_size);
    l.output_gpu   = cuda_make_array(l.output, output_size);
    l.delta_gpu    = cuda_make_array(l.delta, output_size);
    #endif
    fprintf(stderr, "max          %dx%d / %dx%d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size_w, size_h, stride_w, stride_h, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + 2*l->pad - l->size_w)/l->stride_w + 1;
    l->out_h = (h + 2*l->pad - l->size_h)/l->stride_h + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = (int *)realloc(l->indexes, output_size * sizeof(int));
    l->output = (float *)realloc(l->output, output_size * sizeof(float));
    l->delta = (float *)realloc(l->delta, output_size * sizeof(float));

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size_h; ++n){
                        for(m = 0; m < l.size_w; ++m){
                            int cur_h = h_offset + i*l.stride_h + n;
                            int cur_w = w_offset + j*l.stride_w + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

