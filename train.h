
#ifndef TRAIN_H
#define TRAIN_H

#include "CNN.h"
#include "Matrix.h"
#include "linAlgebra.h"
#include  "functions.h"
#include "cnn.h"


class Trainer {
public:
    Trainer(CNN& cnn, double learning_rate, unsigned int epochs)
        : cnn(cnn), learning_rate(learning_rate), epochs(epochs) {}

    void train(std::vector<std::unique_ptr<Matrix>>& Xtrain, std::vector<std::unique_ptr<std::vector<double>>>& Ytrain);

private:
    CNN& cnn;
    double learning_rate;
    unsigned int epochs;
};

#endif // TRAIN_H
