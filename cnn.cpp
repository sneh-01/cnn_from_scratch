#include "cnn.h"
#include "functions.h"
#include "linAlgebra.h"
#include <random>
#include <vector>
#include <memory>
#include "Matrix.h"
#include <cmath>
#include <assert.h>

using namespace std;

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

void CNN::forward_pass(const std::unique_ptr<Matrix> &input)
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
    auto delta_output = np::applyFunction(*activations[1], func::relu_gradient);
    std::vector<double> delta_h = np::multiply(delta_L, delta_output);

    // Update weights and biases for output layer
    cnn::update_weights(activations[1], delta_h, weights[1], learning_rate);
    cnn::update_bias(delta_h, biases[1], learning_rate);

    // Backpropagation through hidden layer
    auto delta_hidden = np::applyFunction(*activations[0], func::relu_gradient);
    delta_h = np::multiply(np::dot(weights[1], delta_h), delta_hidden);

    // Update weights and biases for hidden layer
    cnn::update_weights(activations[0], delta_h, weights[0], learning_rate);
    cnn::update_bias(delta_h, biases[0], learning_rate);

    // Backpropagation through second convolutional layer
    auto delta_conv2 = np::applyFunction(*conv_activations[3], func::relu_gradient);
    update_kernel(delta_conv2, conv_activations[2], kernels[1], learning_rate);

    // Backpropagation through first convolutional layer
    auto delta_conv1 = np::applyFunction(*conv_activations[1], func::relu_gradient);
    update_kernel(delta_conv1, conv_activations[0], kernels[0], learning_rate);
}

void CNN::info(){
    	cout << "Kernel size: (" << kernel->getRows() << "," << kernel->getColumns() << ")" << endl;
        for(unsigned int i = 0; i < weights.size(); i++){
            cout << "Weight "<< i << " size: (" << weights[i]->getRows() << "," << weights[i]->getColumns() << ")" << endl;
        }
}

std::vector<double> CNN::get_output() const {
    return *activations.back();
}

double CNN::cross_entropy(std::unique_ptr<std::vector<double> > &ypred, 
					std::unique_ptr<std::vector<double> > &ytrue){
	
	assert(ypred->size() == ytrue->size());
	std::unique_ptr<std::vector<double> > z = np::applyFunction(ypred,log);
	z = np::multiply(z,ytrue);
	double error = np::element_sum(z);
	return (-error);
}



