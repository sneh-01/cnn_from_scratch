#include "train.h"
#include <iostream>

void train(CNN& cnn, const std::vector<std::unique_ptr<Matrix>>& X_train, 
           const std::vector<std::vector<double>>& Y_train, unsigned int epochs, double learning_rate) {
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < X_train.size(); ++i) {
            cnn.forward_pass(X_train[i]);
            cnn.backward_pass(Y_train[i], learning_rate);
            total_loss += cnn.cross_entropy(CNN::get_output(), Ytrain[i]);
        }
        std::cout << "Epoch " << epoch + 1 << " / " << epochs << " - Loss: " << total_loss / X_train.size() << std::endl;
    }
}
