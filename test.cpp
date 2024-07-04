#include "functions.h"
#include "Matrix.h"
#include "cnn.h"
#include "test.h"


#include "test.h"
#include <iostream>

double test(CNN& cnn, const std::vector<std::unique_ptr<Matrix>>& X_test, 
            const std::vector<std::vector<double>>& Y_test) {
    double correct_predictions = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        cnn.forward_pass(X_test[i]);
        const auto& prediction = cnn.get_output();
        auto max_it = std::max_element(prediction.begin(), prediction.end());
        int predicted_label = std::distance(prediction.begin(), max_it);
        auto expected_label = std::distance(Y_test[i].begin(), std::max_element(Y_test[i].begin(), Y_test[i].end()));
        if (predicted_label == expected_label) {
            correct_predictions += 1.0;
        }
    }
    return correct_predictions / X_test.size();
}
