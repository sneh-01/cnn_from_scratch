#ifndef TRAIN_H
#define TRAIN_H

#include "cnn.h"
#include "Matrix.h"
#include <vector>
#include <memory>

void train(CNN& cnn, const std::vector<std::unique_ptr<Matrix>>& X_train, 
           const std::vector<std::vector<double>>& Y_train, unsigned int epochs, double learning_rate);

#endif // TRAIN_H
