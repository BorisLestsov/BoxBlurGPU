#include <stdio.h>
#include "common.h"


int main(int argc, char** argv)
{
	PPMImage* img;
	img = readPPM(argv[1]);
	writePPM("res.ppm", img);
}
