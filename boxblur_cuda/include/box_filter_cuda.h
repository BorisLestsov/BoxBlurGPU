#include "common.h"

void box_blur_cuda(PPMImage* img, PPMImage* res, int ker);
void box_blur_cuda_sep(PPMImage* img, PPMImage* res, int ker);
