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


// int CNN::forward_propagate(std::unique_ptr<Matrix> &input,
// 	std::vector<std::unique_ptr<Matrix> > &conv_activations, 
// 	std::vector<std::unique_ptr<std::vector<double> > > &activations){
	
// 	assert(weights.size() == 2); 
// 	std::unique_ptr<Matrix> conv = std::make_unique<Matrix>(input->getRows() - kernel->getRows() + 1,
// 							input->getColumns() - kernel->getColumns() + 1, true);	

// 	for(unsigned int i = 0; i < conv->getRows(); i++){
// 		for(unsigned int j = 0; j < conv->getColumns(); j++){
// 			conv->set(i,j,np::multiply(kernel, input, i, j));				
// 		}
// 	}
// 	conv = np::applyFunction(conv, fns::relu);

// 	unsigned int x = (conv->getRows()/pool_window.rows);
// 	unsigned int y = (conv->getColumns()/pool_window.columns);
	
// 	std::unique_ptr<Matrix>	pool = std::make_unique<Matrix>(conv->getRows(), conv->getColumns(), false);
// 	std::unique_ptr<std::vector<double> > pool_flatten = std::make_unique<std::vector<double> >();

// 	unsigned int xptr=0, yptr=0;
// 	auto max_index = std::make_unique<Shape>(Shape{0,0});
// 	for(unsigned int i=0; i < x; i++){
// 		xptr = (i * pool_window.rows);
// 		for(unsigned int j=0; j < y; j++){
// 			yptr = (j * pool_window.columns);
// 			double max = np::maximum(conv, xptr, yptr, pool_window, max_index);
// 			pool_flatten->push_back(max);
// 			pool->set(max_index->rows, max_index->columns, 1);
// 		}
// 	}
	
// 	conv_activations[0] = std::move(pool);

// 	pool_flatten->push_back(1);


// 	std::unique_ptr<Matrix> W0 = np::transpose(weights[0]);
// 	std::unique_ptr<std::vector<double> > hidden = np::dot(W0, pool_flatten);
// 	hidden = np::applyFunction(hidden, fns::relu);
// 	hidden->push_back(1);

// 	activations[0] = std::move(pool_flatten);

// 	std::unique_ptr<Matrix> W1 = np::transpose(weights[1]);
// 	std::unique_ptr<std::vector<double> > output = np::dot(W1, hidden);
// 	output = np::applyFunction(output, fns::softmax);
// 	output = np::normalize(output); 
	
// 	activations[1] = std::move(hidden);
// 	activations[2] = std::move(output);
// 	return 0;
// }


