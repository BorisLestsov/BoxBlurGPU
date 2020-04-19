#include "common.h"

void box_blur_cpu(PPMImage* img, PPMImage* res, int ker, int num_threads=0);
void box_blur_cpu_acc(PPMImage* img, PPMImage* res, int ker, int num_threads=0);
void box_blur_cpu_sep(PPMImage* img, PPMImage* res, int ker, int num_threads=0);
