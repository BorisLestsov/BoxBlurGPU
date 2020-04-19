#include <iostream>
#include "common.h"
#include "omp.h"


void box_blur_cpu_opt(PPMImage* img, PPMImage* res, int ker){
	int h = res->x;
	int w = res->y;
    //int oh = img->x;
	int ow = img->y;
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;
	int ker_size = ker*ker;


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
}

void box_blur_cpu(PPMImage* img, PPMImage* res, int ker){
	int h = res->x;
	int w = res->y;
	// int oh = img->x;
	int ow = img->y;
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;
	int ker_size = ker*ker;

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
}
