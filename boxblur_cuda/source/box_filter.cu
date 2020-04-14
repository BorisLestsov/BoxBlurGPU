#include <iostream>
#include "common.h"
#include "helper_functions.cuh"

__global__
void boxblurker(uchar* d_ptr, uchar* d_res, int ker, int oh, int ow)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int h = oh - (ker - 1);
    int w = ow - (ker - 1);

    if (i >= h || j >= w) {
        return;
    }

    int sum_r = 0;
    for (int ki = i; ki < i+ker; ki++){
        for (int kj = j; kj < j+ker; kj++){
            uchar r = d_ptr[3*(ki*ow + kj) + c];
            sum_r += r;
        }
    }

	int ker_size = ker*ker;
    d_res[3*(i*w+j) + c] = sum_r/ker_size;
}

void box_blur_cuda(PPMImage* img, PPMImage* res, int ker){
	int h = res->x;
	int w = res->y;
	int oh = img->x;
	int ow = img->y;
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;

    uchar* d_ptr, *d_res;

    checkCudaErrors(cudaMalloc(&d_ptr, oh*ow*3*sizeof(uchar))); 
    checkCudaErrors(cudaMalloc(&d_res, h*w*3*sizeof(uchar)));

    checkCudaErrors(cudaMemcpy(d_ptr, ptr, oh*ow*3*sizeof(uchar), cudaMemcpyHostToDevice));

    int cell_size = 32;
    int num_blocks_x = h/cell_size + (h % cell_size != 0);
    int num_blocks_y = w/cell_size + (w % cell_size != 0);
    dim3 block_size = dim3(cell_size, cell_size);
    dim3 grid_size = dim3(num_blocks_x, num_blocks_y, 3);
    boxblurker<<<grid_size, block_size>>>(d_ptr, d_res, ker, oh, ow);

    checkCudaErrors(cudaMemcpy(dst, d_res, h*w*3*sizeof(uchar)*sizeof(uchar), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_ptr));
    checkCudaErrors(cudaFree(d_res));
}
