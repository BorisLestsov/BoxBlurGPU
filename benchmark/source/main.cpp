#include <iostream>
#include <string>
#include <exception>
#include "box_filter_cpu.h"
#include "box_filter_cuda.h"
#include "argparse.hpp"


int main(int argc, const char** argv)
{
    argparse::ArgumentParser parser;
    try {
        parser.addArgument("--input", 1, false);
        parser.addArgument("--output", 1, false);
        parser.addArgument("--ker", 1, false);
        parser.addArgument("--cpumode", 1, false);
        parser.addArgument("--cudamode", 1, false);

        parser.parse(argc, argv);
    } catch (std::exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }


    try {

        PPMImage* img, *res;
        img = readPPM(parser.retrieve<std::string>("input").c_str());
        int ker = parser.retrieve<int>("ker");
        std::string cpu_mode = parser.retrieve<std::string>("cpumode");
        std::string cuda_mode = parser.retrieve<std::string>("cudamode");

        res = new PPMImage;
        res->x = img->x - (ker - 1);
        res->y = img->y - (ker - 1);
        res->data = (PPMPixel*)malloc(res->x * res->y * sizeof(PPMPixel));

        if (cpu_mode == "conv"){
            box_blur_cpu(img, res, ker, 1);
            box_blur_cpu(img, res, ker, 4);
        } else if (cpu_mode == "sep") {
            box_blur_cpu_sep(img, res, ker, 1);
            box_blur_cpu_sep(img, res, ker, 4);
        } else if (cpu_mode == "acc") {
            box_blur_cpu_acc(img, res, ker, 1);
            box_blur_cpu_acc(img, res, ker, 4);
        } else {
            throw std::string("Unknown cpu mode!");
        }

        if (cpu_mode == "conv"){
            box_blur_cuda(img, res, ker);
        } else if (cpu_mode == "sep") {
            box_blur_cuda_sep(img, res, ker);
        } else {
            throw std::string("Unknown cuda mode!");
        }

        writePPM(parser.retrieve<std::string>("output").c_str(), res);
        free(img->data);
        free(res->data);
        free(res);
        free(img);
    } catch (std::exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
    } catch (std::string e) {
        std::cout << "Exception: " << e << std::endl;
    }
}


