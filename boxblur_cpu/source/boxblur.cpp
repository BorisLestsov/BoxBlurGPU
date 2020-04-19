#include <iostream>
#include <chrono>
#include "common.h"
#include "omp.h"

void box_blur_cpu_sep(PPMImage* img, PPMImage* res, int ker, int num_threads){
	int h = res->x;
	int w = res->y;
    int oh = img->x;
	int ow = img->y;
	int th = oh;
	int tw = ow - (ker - 1);
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;
	int ker_size = ker*ker;

	int* res2 = (int*) malloc(th * tw * 3*sizeof(int));

    if (num_threads == 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel for
	for (int i = 0; i < th; i++){
		for (int j = 0; j < tw; j++){

			int sum_r = 0;
			int sum_g = 0;
			int sum_b = 0;
            int ki = i;
            for (int kj = j; kj < j+ker; kj++){
                uchar r = ptr[3*(ki*ow + kj) + 0];
                uchar g = ptr[3*(ki*ow + kj) + 1];
                uchar b = ptr[3*(ki*ow + kj) + 2];
                sum_r += r;
                sum_g += g;
                sum_b += b;
            }

			res2[3*(i*tw+j) + 0] = sum_r;
			res2[3*(i*tw+j) + 1] = sum_g;
			res2[3*(i*tw+j) + 2] = sum_b;
		}
	}

#pragma omp parallel for
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){

			int sum_r = 0;
			int sum_g = 0;
			int sum_b = 0;
            int kj = j;
			for (int ki = i; ki < i+ker; ki++){
                int r = res2[3*(ki*tw + kj) + 0];
                int g = res2[3*(ki*tw + kj) + 1];
                int b = res2[3*(ki*tw + kj) + 2];
                sum_r += r;
                sum_g += g;
                sum_b += b;
			}

			dst[3*(i*w+j) + 0] = sum_r/ker_size;
			dst[3*(i*w+j) + 1] = sum_g/ker_size;
			dst[3*(i*w+j) + 2] = sum_b/ker_size;
		}
	}

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "CPU " << num_threads << " THREADS (ms.):\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

	free(res2);
}

void box_blur_cpu_acc(PPMImage* img, PPMImage* res, int ker, int num_threads){
	int h = res->x;
	int w = res->y;
    //int oh = img->x;
	int ow = img->y;
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;
	int ker_size = ker*ker;

    if (num_threads == 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel for
	for (int i = 0; i < h; i++){

        int left_sum_r = 0;
        int left_sum_g = 0;
        int left_sum_b = 0;
        int new_left_sum_r = 0;
        int new_left_sum_g = 0;
        int new_left_sum_b = 0;
        int right_sum_r = 0;
        int right_sum_g = 0;
        int right_sum_b = 0;
        int sum_r = 0;
        int sum_g = 0;
        int sum_b = 0;
        for (int ki = i; ki < i+ker; ki++){
            for (int kj = 0; kj < 0+ker; kj++){
                uchar r = ptr[3*(ki*ow + kj) + 0];
                uchar g = ptr[3*(ki*ow + kj) + 1];
                uchar b = ptr[3*(ki*ow + kj) + 2];
                sum_r += r;
                sum_g += g;
                sum_b += b;
                if (kj == 0){
                    left_sum_r += r;
                    left_sum_g += g;
                    left_sum_b += b;
                }
            }
        }
        dst[3*(i*w+0) + 0] = sum_r/ker_size;
        dst[3*(i*w+0) + 1] = sum_g/ker_size;
        dst[3*(i*w+0) + 2] = sum_b/ker_size;

		for (int j = 1; j < w; j++){

            int kj;
            uchar r;
            uchar g;
            uchar b;
            new_left_sum_r = 0;
            new_left_sum_g = 0;
            new_left_sum_b = 0;
            right_sum_r = 0;
            right_sum_g = 0;
            right_sum_b = 0;
			for (int ki = i; ki < i+ker; ki++){

                // right col
                kj = j+ker-1;
                r = ptr[3*(ki*ow + kj) + 0];
                g = ptr[3*(ki*ow + kj) + 1];
                b = ptr[3*(ki*ow + kj) + 2];
                right_sum_r += r;
                right_sum_g += g;
                right_sum_b += b;

                // left col
                kj = j;
                r = ptr[3*(ki*ow + kj) + 0];
                g = ptr[3*(ki*ow + kj) + 1];
                b = ptr[3*(ki*ow + kj) + 2];
                new_left_sum_r += r;
                new_left_sum_g += g;
                new_left_sum_b += b;
			}
            sum_r += right_sum_r - left_sum_r;
            sum_g += right_sum_g - left_sum_g;
            sum_b += right_sum_b - left_sum_b;
            left_sum_r = new_left_sum_r;
            left_sum_g = new_left_sum_g;
            left_sum_b = new_left_sum_b;

			dst[3*(i*w+j) + 0] = sum_r/ker_size;
			dst[3*(i*w+j) + 1] = sum_g/ker_size;
			dst[3*(i*w+j) + 2] = sum_b/ker_size;
		}
	}

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "CPU " << num_threads << " THREADS (ms.):\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
}

void box_blur_cpu(PPMImage* img, PPMImage* res, int ker, int num_threads){
	int h = res->x;
	int w = res->y;
	// int oh = img->x;
	int ow = img->y;
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;
	int ker_size = ker*ker;

    if (num_threads == 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#pragma omp parallel for
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){

			int sum_r = 0;
			int sum_g = 0;
			int sum_b = 0;
			for (int ki = i; ki < i+ker; ki++){
				for (int kj = j; kj < j+ker; kj++){
					uchar r = ptr[3*(ki*ow + kj) + 0];
					uchar g = ptr[3*(ki*ow + kj) + 1];
					uchar b = ptr[3*(ki*ow + kj) + 2];
					sum_r += r;
					sum_g += g;
					sum_b += b;
				}
			}

			dst[3*(i*w+j) + 0] = sum_r/ker_size;
			dst[3*(i*w+j) + 1] = sum_g/ker_size;
			dst[3*(i*w+j) + 2] = sum_b/ker_size;
		}
	}

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "CPU " << num_threads << " THREADS (ms.):\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
}
