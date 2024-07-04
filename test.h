#include "Matrix.h"
#include "functions.h"
#include "linAlgebra.h"
#include "cnn.h"
#include "train.h"

#ifndef TEST_H
#define TEST_H

#include "cnn.h"
#include "Matrix.h"
#include <vector>
#include <memory>

double test(CNN& cnn, const std::vector<std::unique_ptr<Matrix>>& X_test, 
            const std::vector<std::vector<double>>& Y_test);

#endif // TEST_H
