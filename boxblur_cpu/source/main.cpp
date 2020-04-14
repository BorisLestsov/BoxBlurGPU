#include <iostream>
#include "common.h"
#include "omp.h"

void box_blur_cpu(PPMImage* img, PPMImage* res, int ker){
	int h = res->x;
	int w = res->y;
	int oh = img->x;
	int ow = img->y;
	uchar* ptr = (uchar*) img->data;
	uchar* dst = (uchar*) res->data;
	int ker_size = ker*ker;

#pragma omp parallel for
	for (int i = 0; i < res->x; i++){
		for (int j = 0; j < res->y; j++){

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

int main(int argc, char** argv)
{
	PPMImage* img, *res;
	img = readPPM(argv[1]);
	int ker = 10;

	res = new PPMImage;
	res->x = img->x - (ker - 1);
	res->y = img->y - (ker - 1);
    res->data = (PPMPixel*)malloc(res->x * res->y * sizeof(PPMPixel));

	box_blur_cpu(img, res, ker);

	writePPM("res.ppm", res);
	delete res;
	delete img;
}
