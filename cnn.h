#ifndef CNN_H
#define CNN_H

#include <iostream>
#include <vector>
#include <memory>
#include "functions.h"
#include "linAlgebra.h"
#include "Matrix.h" // Assume Matrix class is defined

// Shape struct to represent dimensions


class CNN {
public:
    CNN(Shape input_dim, Shape kernel_size, Shape pool_size, unsigned int hidden_layer_nodes, unsigned int output_dim);

    void forward_pass(const std::unique_ptr<Matrix>& input);
    void backward_pass(const std::vector<double>& expected_output, double learning_rate);
    void info();
    std::vector<double> get_output() const;
    double cross_entropy(std::unique_ptr<std::vector<double> > &ypred, 
					std::unique_ptr<std::vector<double> > &ytrue);

private:
    Shape input_dim;
    Shape kernel_size;
    Shape pool_size;
    std::unique_ptr<Matrix> kernel;
    std::vector<std::unique_ptr<Matrix>> weights;

    std::vector<std::unique_ptr<Matrix>> conv_activations;
    std::vector<std::unique_ptr<std::vector<double>>> activations;



    double cross_entropy_loss(const std::vector<double>& predicted, const std::vector<double>& expected);
    double softmax(const std::vector<double>& logits, int index);

    void update_kernel(const std::unique_ptr<Matrix>& delta_conv, const std::unique_ptr<Matrix>& input);
};

#endif // CNN_H
