#include <iostream>
#include "common.h"
#include "helper_functions.cuh"

__global__
void boxblur_ker_sep_y(int* d_ptr, uchar* d_res, int ker, int oh, int ow)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int h = oh - (ker - 1);
    int w = ow;

    if (i >= h || j >= w) {
        return;
    }

    int sum_r = 0;
    int kj = j;
    for (int ki = i; ki < i+ker; ki++){
        int r = d_ptr[3*(ki*ow + kj) + c];
        sum_r += r;
    }

	int ker_size = ker*ker;
    d_res[3*(i*w+j) + c] = sum_r/ker_size;
}

__global__
void boxblur_ker_sep_x(uchar* d_ptr, int* d_res, int ker, int oh, int ow)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    int h = oh;
    int w = ow - (ker - 1);

    if (i >= h || j >= w) {
        return;
    }

    int sum_r = 0;
    int ki = i;
    for (int kj = j; kj < j+ker; kj++){
        uchar r = d_ptr[3*(ki*ow + kj) + c];
        sum_r += r;
    }

	// int ker_size = ker;
    d_res[3*(i*w+j) + c] = sum_r; ///ker_size;
}

void box_blur_cuda_sep(PPMImage* img, PPMImage* res, int ker){
	int h = res->x;
	int w = res->y;
	int oh = img->x;
	int ow = img->y;
	int th = oh;
	int tw = ow - (ker - 1);
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;

    uchar* d_ptr, *d_res;
    int* d_int;

    checkCudaErrors(cudaMalloc(&d_ptr, oh*ow*3*sizeof(uchar))); 
    checkCudaErrors(cudaMalloc(&d_int, th*tw*3*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_res, h*w*3*sizeof(uchar)));

    cudaEvent_t start_all, stop_all, start_comp, stop_comp;
    checkCudaErrors(cudaEventCreate(&start_all));
    checkCudaErrors(cudaEventCreate(&stop_all));
    checkCudaErrors(cudaEventCreate(&start_comp));
    checkCudaErrors(cudaEventCreate(&stop_comp));

    checkCudaErrors(cudaEventRecord(start_all));

    checkCudaErrors(cudaMemcpy(d_ptr, ptr, oh*ow*3*sizeof(uchar), cudaMemcpyHostToDevice));

    int cell_size = 32;
    int num_blocks_x;
    int num_blocks_y;
    dim3 block_size;
    dim3 grid_size;

    checkCudaErrors(cudaEventRecord(start_comp));

    num_blocks_x = th/cell_size + (th % cell_size != 0);
    num_blocks_y = tw/cell_size + (tw % cell_size != 0);
    block_size = dim3(cell_size, cell_size);
    grid_size = dim3(num_blocks_x, num_blocks_y, 3);
    boxblur_ker_sep_x<<<grid_size, block_size>>>(d_ptr, d_int, ker, oh, ow);

    num_blocks_x = h/cell_size + (h % cell_size != 0);
    num_blocks_y = w/cell_size + (w % cell_size != 0);
    block_size = dim3(cell_size, cell_size);
    grid_size = dim3(num_blocks_x, num_blocks_y, 3);
    boxblur_ker_sep_y<<<grid_size, block_size>>>(d_int, d_res, ker, th, tw);

    checkCudaErrors(cudaEventRecord(stop_comp));

    checkCudaErrors(cudaMemcpy(dst, d_res, h*w*3*sizeof(uchar)*sizeof(uchar), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stop_all));

    checkCudaErrors(cudaEventSynchronize(stop_all));
    float milliseconds_all=0, milliseconds_comp=0, milliseconds_data=0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds_all, start_all, stop_all));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds_comp, start_comp, stop_comp));
    milliseconds_data = milliseconds_all - milliseconds_comp;

    std::cout << "GPU COMPUTE TIME (microseconds):\t" << (int) (milliseconds_comp*1000) << std::endl;
    std::cout << "GPU TOTAL TIME (microseconds):\t" << (int) (milliseconds_all*1000) << std::endl;
    std::cout << "GPU DATA TIME (microseconds):\t" << (int) (milliseconds_data*1000) << std::endl;

    checkCudaErrors(cudaFree(d_ptr));
    checkCudaErrors(cudaFree(d_res));
}


__global__
void boxblur_ker(uchar* d_ptr, uchar* d_res, int ker, int oh, int ow)
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

    cudaEvent_t start_all, stop_all, start_comp, stop_comp;
    checkCudaErrors(cudaEventCreate(&start_all));
    checkCudaErrors(cudaEventCreate(&stop_all));
    checkCudaErrors(cudaEventCreate(&start_comp));
    checkCudaErrors(cudaEventCreate(&stop_comp));

    checkCudaErrors(cudaEventRecord(start_all));

    checkCudaErrors(cudaMemcpy(d_ptr, ptr, oh*ow*3*sizeof(uchar), cudaMemcpyHostToDevice));

    int cell_size = 32;
    int num_blocks_x = h/cell_size + (h % cell_size != 0);
    int num_blocks_y = w/cell_size + (w % cell_size != 0);
    dim3 block_size = dim3(cell_size, cell_size);
    dim3 grid_size = dim3(num_blocks_x, num_blocks_y, 3);

    checkCudaErrors(cudaEventRecord(start_comp));
    boxblur_ker<<<grid_size, block_size>>>(d_ptr, d_res, ker, oh, ow);
    checkCudaErrors(cudaEventRecord(stop_comp));

    checkCudaErrors(cudaMemcpy(dst, d_res, h*w*3*sizeof(uchar)*sizeof(uchar), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stop_all));

    checkCudaErrors(cudaEventSynchronize(stop_all));
    float milliseconds_all=0, milliseconds_comp=0, milliseconds_data=0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds_all, start_all, stop_all));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds_comp, start_comp, stop_comp));
    milliseconds_data = milliseconds_all - milliseconds_comp;

    std::cout << "GPU COMPUTE TIME (microseconds):\t" << (int) (milliseconds_comp*1000) << std::endl;
    std::cout << "GPU TOTAL TIME (microseconds):\t" << (int) (milliseconds_all*1000) << std::endl;
    std::cout << "GPU DATA TIME (microseconds):\t" << (int) (milliseconds_data*1000) << std::endl;

    checkCudaErrors(cudaFree(d_ptr));
    checkCudaErrors(cudaFree(d_res));
}

