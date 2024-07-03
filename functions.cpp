#include <random>
#include<cmath>
#include<functions.h>
#include<fstream>
#include<time.h>
#include<float.h>
#include "linAlgebra.h"



// using namespace std;


namespace func{
    double relu(double x){
        if(x > 0) return x;
        else return 0;
    }

    double sigmoid(double x){
        return 1.0/(1.0+exp(-x));
    }

    double tanh(double x){
        return tanh(x);
    }

    double relu_gradient(double x){
        if (x>0) return (double)1;
        else return (double)0.2;

    }

    double sigmoid_gradient(double x){
        return x*(1-x);
    }

    double tan_gradient(double x){
        return (1-(x*x));
    }

    double softmax(double x){
        if(isnan(x)) return 0;
        return exp(x);
    }


}
namespace cnn{

    std::unique_ptr<Matrix> convolution(const std::unique_ptr<Matrix>& input, const std::unique_ptr<Matrix>& kernel)
{
    int output_rows = input->getRows() - kernel->getRows() + 1;
    int output_cols = input->getColumns() - kernel->getColumns() + 1;
    std::unique_ptr<Matrix> conv_result(new Matrix(output_rows, output_cols, false));

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            double dot_product = np::multiply(kernel, input, i, j);
            conv_result->set(i, j, dot_product);
        }
    }

    return conv_result;
}

std::unique_ptr<Matrix> max_pool(const std::unique_ptr<Matrix>& input, int pool_size, int stride)
{
    int output_rows = input->getRows() / pool_size;
    int output_cols = input->getColumns() / pool_size;
    std::unique_ptr<Matrix> pooled(new Matrix(output_rows, output_cols, false));

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_cols; ++j) {
            double max_val = np::maximum(input, i * pool_size, j * pool_size, Shape{pool_size, pool_size});
            pooled->set(i, j, max_val);
        }
    }

    return pooled;
}

std::unique_ptr<std::vector<double>> pooling_flatten(const std::unique_ptr<Matrix>& input)
{
    int size = input->getRows() * input->getColumns();
    std::unique_ptr<std::vector<double>> flattened(new std::vector<double>(size));

    for (int i = 0; i < input->getRows(); ++i) {
        for (int j = 0; j < input->getColumns(); ++j) {
            (*flattened)[i * input->getColumns() + j] = input->get(i, j);
        }
    }

    return flattened;
}

}

// namespace pre_process{
//     int process_GTSRB__images(const char* path, vector<unique_ptr<Matrix>> &X_train , vector<unique_ptr<double>> &Y_train, unsigned int nr_images){
//         string str(path);
//         const int width = 30;
//         const int hight = 30;
//         const int LABELS = 43;

//         for(unsigned int i=0; i<nr_images;i++){
//             vector<cv::String> files;       
            
            
//              }
//     }
// }


