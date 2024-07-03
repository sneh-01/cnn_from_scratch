#include "cnn.h"
#include "functions.h"
#include "linAlgebra.h"
#include<random>
#include<vector>
#include<memory>
#include "Matrix.h" 
#include <cmath>

CNN::CNN(Shape input_dim, Shape kernel_size, Shape pool_size, unsigned int hidden_layer_nodes, unsigned int output_dim)
    : input_dim(input_dim), kernel_size(kernel_size), pool_size(pool_size)
{
    // Initialize kernel
    kernel = std::make_unique<Matrix>(kernel_size.rows, kernel_size.columns, true);

    // Initialize weights
    std::unique_ptr<Matrix> W0(new Matrix((input_dim.rows - kernel_size.rows + 1) / pool_size.rows * (input_dim.columns - kernel_size.columns + 1) / pool_size.columns + 1, hidden_layer_nodes, true));
    this->weights.push_back(std::move(W0));
    std::unique_ptr<Matrix> W1(new Matrix(hidden_layer_nodes + 1, output_dim, true));
    this->weights.push_back(std::move(W1));
}





void CNN::forward_pass(const std::unique_ptr<Matrix>& input)
{
    // Convolution 1
    auto conv1 = cnn::convolution(input, kernel);
    conv1 = np::applyFunction(conv1, func::relu);
    conv_activations.push_back(std::move(conv1));

    // Max Pooling 1
    auto pool1 = cnn::max_pool(conv_activations.back(), pool_size.rows, 2);
    conv_activations.push_back(std::move(pool1));

    // Convolution 2
    auto conv2 = cnn::convolution(conv_activations.back(), kernel);
    conv2 = np::applyFunction(conv2, func::relu);
    conv_activations.push_back(std::move(conv2));

    // Max Pooling 2
    auto pool2 = cnn::max_pool(conv_activations.back(), pool_size.rows, 2);
    conv_activations.push_back(std::move(pool2));

    // Pool Flattening
    auto flatten = cnn::pooling_flatten(conv_activations.back());
    activations.push_back(std::move(flatten));

    // Append bias for hidden layer
    activations.back()->push_back(1.0);

    // Hidden Layer
    auto W0 = np::transpose(weights[0]);
    auto hidden = np::dot(W0, activations.back());
    hidden = np::applyFunction(hidden, func::relu);
    activations.push_back(std::move(hidden));

    // Append bias for output layer
    activations.back()->push_back(1.0);

    // Output Layer (softmax is applied in cross_entropy_loss)
    auto W1 = np::transpose(weights[1]);
    auto output = np::dot(W1, activations.back());
    activations.push_back(std::move(output));
}

void CNN::backward_pass(const std::vector<double>& expected_output, double learning_rate)
{
    // Remove bias from activations
    activations[1]->pop_back();
    activations[0]->pop_back();

    // Compute loss derivative (using softmax and cross entropy derivative combined)
    std::vector<double> delta_L = np::subtract(*activations.back(), expected_output);

    // Backpropagation through output layer
    auto dW1 = np::dot(activations[1], delta_L);
    dW1 = np::multiply(dW1, learning_rate);
    weights[1] = np::subtract(weights[1], dW1);

    // Delta calculation for hidden layer
    auto delta_h = np::dot(weights[1], delta_L);
    auto active = np::applyFunction(activations[1], func::relu_gradient);
    delta_h = np::multiply(delta_h, active);

    // Backpropagation through hidden layer
    auto dW0 = np::dot(activations[0], delta_h, 1);
    dW0 = np::multiply(dW0, learning_rate);
    weights[0] = np::subtract(weights[0], dW0);

    // Backpropagation through convolutional layers
    auto delta_x = np::dot(weights[0], delta_h);
    active = np::applyFunction(activations[0], func::relu_gradient);

    std::unique_ptr<Matrix> delta_conv(new Matrix(conv_activations[0]->getRows(), conv_activations[0]->getColumns(), false));
    unsigned int counter = 0;
    for (unsigned int r = 0; r < conv_activations[0]->getRows(); ++r) {
        for (unsigned int c = 0; c < conv_activations[0]->getColumns(); ++c) {
            if (conv_activations[0]->get(r, c) == 1.0) {
                delta_conv->set(r, c, (*delta_x)[counter++]);
            }
        }
    }

    // Update kernel
    update_kernel(delta_conv, conv_activations[0]);
}






void CNN::update_kernel(const std::unique_ptr<Matrix>& delta_conv, const std::unique_ptr<Matrix>& input)
{
    for (int i = 0; i < kernel->getRows(); ++i) {
        for (int j = 0; j < kernel->getColumns(); ++j) {
            double delta = np::multiply(delta_conv, input, i, j);
            kernel->set(i, j, delta);
        }
    }
}

