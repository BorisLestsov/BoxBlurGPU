#include <iostream>
#include <string>
#include <exception>
#include "argparse.hpp"
#include "box_filter_cuda.h"


int main(int argc, const char** argv)
{

    argparse::ArgumentParser parser;
    try {
        parser.addArgument("--input", 1, false);
        parser.addArgument("--output", 1, false);
        parser.addArgument("--ker", 1, false);
        parser.addArgument("--mode", 1, false);

        parser.parse(argc, argv);
    } catch (std::exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }

    
    try {
        PPMImage* img, *res;
        img = readPPM(parser.retrieve<std::string>("input").c_str());
        int ker = parser.retrieve<int>("ker");
        std::string mode = parser.retrieve<std::string>("mode");

        res = new PPMImage;
        res->x = img->x - (ker - 1);
        res->y = img->y - (ker - 1);
        res->data = (PPMPixel*)malloc(res->x * res->y * sizeof(PPMPixel));

        if (mode == "conv"){
            box_blur_cuda(img, res, ker);
        } else if (mode == "sep") {
            box_blur_cuda_sep(img, res, ker);
        } else if (mode == "check") {
            PPMImage* res2 = new PPMImage;
            res2->x = img->x - (ker - 1);
            res2->y = img->y - (ker - 1);
            res2->data = (PPMPixel*)malloc(res2->x * res2->y * sizeof(PPMPixel));

            box_blur_cuda(img, res, ker);
            box_blur_cuda_sep(img, res2, ker);

            uchar* res_ptr = (uchar*) res->data;
            uchar* res2_ptr = (uchar*) res2->data;

            double s = 0;
            for (int i = 0; i < res->x; i++){
                for (int j = 0; j < res->y; j++){
                    int ind = i*res->y + j;
                    s += std::abs(res_ptr[ind+0] - res2_ptr[ind+0]);
                    s += std::abs(res_ptr[ind+1] - res2_ptr[ind+1]);
                    s += std::abs(res_ptr[ind+2] - res2_ptr[ind+2]);
                }
            }
            s /= res->x * res->y * 3;
            std::cout << "MAE: " << s << std::endl;

            free(res2->data);
            free(res2);
        } else {
            throw std::string("Unknown mode!");
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


