#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "cnn.h"

#include "linAlgebra.h"
#include "cnn.h"
#include "Matrix.h"  

using namespace std;

extern default_random_engine random_engine;

struct Shape {
    int rows;
    int columns;
};

namespace func {
    double relu(double x);
    double sigmoid(double x);
    double tanh(double x);
    double relu_gradient(double x);
    double sigmoid_gradient(double x);
    double tanh_gradient(double x);
    double softmax(double x);
}
namespace cnn{

    std::unique_ptr<Matrix> convolution(const std::unique_ptr<Matrix>& input, const std::unique_ptr<Matrix>& kernel);
    std::unique_ptr<Matrix> max_pool(const std::unique_ptr<Matrix>& input, int pool_size, int stride);
    std::unique_ptr<std::vector<double>> pooling_flatten(const std::unique_ptr<Matrix>& input);

}

namespace pre_process {
    // Assuming Matrix is defined elsewhere or forward declared
    int process_GTSRB_image(const char* path, vector<unique_ptr<Matrix>> &X_train, vector<unique_ptr<vector<Matrix>>> &Y_train, unsigned int nr_images = 100);
    int process_GTSRB_csv(const char* filename, vector<vector<double>> &X_train, vector<vector<double>>);
    void process_GTSRB(const char* filename);

}

#endif // UTILS_H
