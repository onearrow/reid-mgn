#include "slice_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

// slice层的功能：根据给定的参数将bottom分解成多个top
// axis--要进行分解的维度
// slice_num--切几刀
// slice_pos--切出来的第几块
layer make_slice_layer(int batch, int w, int h, int c, int slice_axis, int slice_num, int slice_pos)
{
    layer l = {0};

    // 获取本层参数
    l.type = SLICE;
    l.batch = batch;
    l.c = c;
    l.h = h;
    l.w = w;
    l.slice_axis = slice_axis;
    l.slice_num = slice_num;
    l.slice_pos = slice_pos;

    if (l.slice_axis == 1) {
    	l.out_c = c;
    	l.out_h = h;
    	l.out_w = w / (l.slice_num + 1);
    } else if (l.slice_axis == 2){
        l.out_c = c;
        l.out_h = h / (l.slice_num + 1);
    	l.out_w = w;
    } else {
    	l.out_c = c / (l.slice_num + 1);
        l.out_h = h;
        l.out_w = w;
    }

    l.outputs = l.out_c * l.out_h * l.out_w;   
    l.inputs = c * h * w;

    int output_size = l.outputs * batch;
    l.output = calloc(output_size, sizeof(float));
    // l.delta = calloc(output_size, sizeof(float));

    l.forward = forward_slice_layer;
    // l.backward = backward_slice_layer;

    #ifdef GPU
    // TODO:
    #endif
    fprintf(stderr, "slice\n");
    return l;
}

void resize_slice_layer(layer *l, int w, int h)
{
    fprintf(stderr, "resize_slice_layer okkkkkk\n");
    int c = l->c;
    l->h = h;
    l->w = w;

    if (l->slice_axis == 1) {
    	l->out_c = c;
    	l->out_h = h;
    	l->out_w = w / (l->slice_num + 1);
    } else if (l->slice_axis == 2){
        l->out_c = c;
        l->out_h = h / (l->slice_num + 1);
    	l->out_w = w;
    } else {
    	l->out_c = c / (l->slice_num + 1);
        l->out_h = h;
        l->out_w = w;
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = realloc(l->output, output_size * sizeof(float));
    // l->delta = realloc(l->delta, output_size * sizeof(float));
}

void slice_cpu(float *x, int w, int h, int c, int batch, int slice_axis, int slice_num, int slice_pos, float *out)
{
    int out_h = h / (slice_num + 1);
    for (int b=0; b<batch; ++b) {
        for (int k=0; k<c; ++k) {
            for (int j=0; j<h; ++j) {
                if ((j>=out_h*slice_pos) && (j<out_h*(slice_pos+1) && (slice_pos==0))) {
                    for (int i=0; i<w; ++i) {
                        int in_index  = i + w*(j + h*(k + c*b));
                        int out_index = i + w*(j + out_h*(k + c*b));
                        // fprintf(stderr, "slice %d %d %d %d %d %d\n", i, j, k, b, in_index, out_index);
                        out[out_index] = x[in_index];
                    }
                }else if ((j>=out_h*slice_pos) && (j<out_h*(slice_pos+1) && (slice_pos==1))) {
                    for (int i=0; i<w; ++i) {
                        int in_index  = i + w*(j + h*(k + c*b));
                        int out_index = i + w*(j + out_h*(k + c*b)) - 1;
                        // fprintf( stderr, "slice %d %d %d %d %d %d\n", i, j, k, b, in_index, out_index);
                        out[out_index] = x[in_index];
                    }
                }
            }
        }
    }
}

void forward_slice_layer(const layer l, network net)
{
    slice_cpu(net.input, l.w, l.h, l.c, l.batch, l.slice_axis, l.slice_num, l.slice_pos, l.output);
}

void backward_slice_layer(const layer l, network net)
{
	// TODO:
}


#ifdef GPU
void forward_slice_layer_gpu(const layer l, network net)
{
    // TODO:
}

void backward_slice_layer_gpu(const layer l, network net)
{
    // TODO:
}
#endif
