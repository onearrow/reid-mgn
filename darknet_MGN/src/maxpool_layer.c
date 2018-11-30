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
    for(int n = 0; n < l.batch; ++n){
        for(int c = 0; c < l.c; ++c){
            for(int i = 0; i < l.out_h; ++i){
                for(int j = 0; j < l.out_w; ++j){
                    int pool_index = n*l.c*l.out_h*l.out_w + c*l.out_h*l.out_w + i*l.out_w + j;
                    float max = -FLT_MAX;
                    int max_index = -1;
                    for(int h = 0; h < l.size_h; ++h){
                        for(int w = 0; w < l.size_w; ++w){
                            int h_offset = i*l.stride_h - l.pad + h;
                            int w_offset = j*l.stride_w - l.pad + w;
                            int index = n*l.c*l.h*l.w + c*l.h*l.w + h_offset*l.w + w_offset;
                            int valid = (h_offset >= 0 && h_offset < l.h && w_offset >= 0 && w_offset < l.w);
                            float val = (valid != 0) ? net.input[index] : -FLT_MAX;
                            if(val > max){
                                max = val;
                                max_index = index;
                            }
                        }
                    }
                    l.output[pool_index] = max;
                    l.indexes[pool_index] = max_index;
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

