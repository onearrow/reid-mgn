#ifndef SLICE_LAYER_H
#define SLICE_LAYER_H

#include "network.h"
#include "layer.h"

layer make_slice_layer(int batch, int w, int h, int c, int slice_axis, int slice_num, int slice_pos);
void resize_slice_layer(layer *l, int w, int h);
void forward_slice_layer(const layer l, network net);
void backward_slice_layer(const layer l, network net);

#ifdef GPU
void forward_slice_layer_gpu(const layer l, network net);
void backward_slice_layer_gpu(const layer l, network net);
#endif

#endif
