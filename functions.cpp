#include <random>
#include<cmath>
#include<functions.h>
#include<fstream>
#include<time.h>
#include<float.h>
#include "linAlgebra.h"
#include<opencv2/opencv.h>







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


    void update_kernel(const std::unique_ptr<Matrix>& delta_conv, const std::unique_ptr<Matrix>& input, std::unique_ptr<Matrix>& kernel, double learning_rate)
    {
        for (int i = 0; i < kernel->getRows(); ++i) {
            for (int j = 0; j < kernel->getColumns(); ++j) {
                double delta = 0.0;
                for (int r = 0; r < delta_conv->getRows(); ++r) {
                    for (int c = 0; c < delta_conv->getColumns(); ++c) {
                        delta += delta_conv->get(r, c) * input->get(r + i, c + j);
                    }
                }
                kernel->set(i, j, kernel->get(i, j) - learning_rate * delta);
            }
        }
    }

    void update_weights(const std::unique_ptr<Matrix>& activations, const std::vector<double>& delta, std::unique_ptr<Matrix>& weights, double learning_rate)
    {
        for (int i = 0; i < weights->getRows(); ++i) {
            for (int j = 0; j < weights->getColumns(); ++j) {
                weights->set(i, j, weights->get(i, j) - learning_rate * activations->get(i) * delta[j]);
            }
        }
    }

    void update_bias(const std::vector<double>& delta, std::vector<double>& bias, double learning_rate)
    {
        for (size_t i = 0; i < bias.size(); ++i) {
            bias[i] -= learning_rate * delta[i];
        }
    }





}


namespace pre_process{

    int process_GTSRB_image(const char* path, std::vector<std::unique_ptr<Matrix>>& Xtrain, 
                        std::vector<std::unique_ptr<std::vector<double>>>& Ytrain, unsigned int nr_images) {
    std::string str(path);  // convert char* to string
    const int width = 28;
    const int height = 28;
    const int LABELS = 43;

    for(unsigned int i=0 ; i< LABELS; i++){
        std::vector<cv::string> files;
        cv::glob(path + std::to_string(i), files, true);

        for(unsigned int k=0; k < (nr_images / LABELS); k++){
        cv::Mat img = cv::imread(files[k]);
        if(img.empty()) continue;
        
        unique_ptr<Matrix> image = make_unique<Matrix>(width, height , true);
        for(unsigned int h =0 ;h<height;h++){
            for(unsigned int w=0 ; w<width ; w++){
                image->set(h,w,(double)(img.at<uchar>(h,w)/255.0));
            }
            Xtrain.emplace_back(std::move(image));
            unique_ptr<vector<double>> vr = make_unique<vector<double>>(LABELS, 0);
            (*vr)[i] = 1.0;
            Ytrain.emplace_back(std::move(vr));  

            }


        }
        return 0;

    }

    int process_GTSRB_csv(const char* filename, std::vector<std::vector<double> > &Xtrain, 
		std::vector<std::vector<double> > &Ytrain)
        {
		std::string data(filename);
		ifstream in(data.c_str());

		if(!in.is_open()) return 1;

		typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
		std::vector<std::string> svec;
		std::string line;

		while(getline(in, line)){
			Tokenizer tok(line);
			auto it = tok.begin();
			int label = std::stoi(*it);
			std::vector<double> labels(10, 0.0);
			labels[label] = 1.0;


			svec.assign(std::next(it, 1), tok.end());

			std::vector<double> dvec(svec.size());
			std::transform(svec.begin(), svec.end(), dvec.begin(), [](const std::string& val)
			{
				return (std::stod(val)/255); // divide by 255 for normalization, since each pixel is 8 bit
			});

			Xtrain.push_back(dvec);
			Ytrain.push_back(labels);
		}
		cout << "processed the input file" << endl;
		return 0;
	}

    void process_image(const char* filename){
        std::vector<double> image;
        cv::Mat img = cv :: imread(filename);
        if(img.empty()){
            std::cout <<"No Image" << endl;

        }
        else{
            if(img.isCountinuous()){
                image.assign(img.datastart, img.dataend);
                for(unsigned int j =0 ; j< image.size(); j++){
                    cout << endl <<image.size();
                }
            }
            else{
                std::cout <<"Image is not continuous" << endl;
            }
        }

    }



    
}





   


