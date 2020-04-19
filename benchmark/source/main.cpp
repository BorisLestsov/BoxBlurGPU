#include <iostream>
#include <string>
#include "box_filter_cpu.h"
#include "box_filter_cuda.h"


int main(int argc, char** argv)
{
	PPMImage* img, *res;
	img = readPPM(argv[1]);
	int ker = std::atoi(argv[3]);
    std::string mode = std::string(argv[4]);

	res = new PPMImage;
	res->x = img->x - (ker - 1);
	res->y = img->y - (ker - 1);
    res->data = (PPMPixel*)malloc(res->x * res->y * sizeof(PPMPixel));

    if (mode == "cpu"){
        box_blur_cpu(img, res, ker);
    } else if (mode == "cuda") {
        box_blur_cuda(img, res, ker);
    } else {
        throw std::string("Unknown mode!");
    }

	writePPM(argv[2], res);
    free(img->data);
    free(res->data);
	free(res);
	free(img);
}


