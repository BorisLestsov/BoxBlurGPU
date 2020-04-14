#include <iostream>
#include "common.h"
#include "box_filter_cuda.h"


int main(int argc, char** argv)
{
	PPMImage* img, *res;
	img = readPPM(argv[1]);
	int ker = 10;

	res = new PPMImage;
	res->x = img->x - (ker - 1);
	res->y = img->y - (ker - 1);
    res->data = (PPMPixel*)malloc(res->x * res->y * sizeof(PPMPixel));

    box_blur_cuda(img, res, ker);

	writePPM("res.ppm", res);
    free(img->data);
    free(res->data);
	free(res);
	free(img);
}


