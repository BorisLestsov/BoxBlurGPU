#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include "argparse.hpp"
#include "box_filter_cpu.h"
#include "box_filter_cuda.h"


int main(int argc, const char** argv)
{
    argparse::ArgumentParser parser;
    try {
        parser.addArgument("--input", 1, false);
        parser.addArgument("--output", 1, false);
        parser.addArgument("--ker", 1, false);
        parser.addArgument("--mode1", 1, false);
        parser.addArgument("--mode2", 1, false);

        parser.parse(argc, argv);
    } catch (std::exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }


    try {

        PPMImage* img, *res;
        img = readPPM(parser.retrieve<std::string>("input").c_str());
        int ker = parser.retrieve<int>("ker");
        std::string mode1 = parser.retrieve<std::string>("mode1");
        std::string mode2 = parser.retrieve<std::string>("mode2");
        std::vector<std::string> modes = {mode1, mode2};

        res = new PPMImage;
        res->x = img->x - (ker - 1);
        res->y = img->y - (ker - 1);
        res->data = (PPMPixel*)malloc(res->x * res->y * sizeof(PPMPixel));

        for (auto& mode: modes){
            if (mode == "cpu_conv"){
                box_blur_cpu(img, res, ker, 1);
                box_blur_cpu(img, res, ker, 4);
            } else if (mode == "cpu_sep") {
                box_blur_cpu_sep(img, res, ker, 1);
                box_blur_cpu_sep(img, res, ker, 4);
            } else if (mode == "cpu_acc") {
                box_blur_cpu_acc(img, res, ker, 1);
                box_blur_cpu_acc(img, res, ker, 4);
            } else if (mode == "cuda_conv"){
                box_blur_cuda(img, res, ker);
            } else if (mode == "cuda_sep") {
                box_blur_cuda_sep(img, res, ker);
            } else {
                throw std::string("Unknown mode!: " + mode);
            }
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


