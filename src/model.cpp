#include "model.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"
#include "kl_error.hpp"

KLError Model::init(const char *model_path) {

}

float Model::inference(const char *img_path) {
    float *model_input = interpreter_->typed_input_tensor<float>(0);

}

void Model::convert_image(const cv::Mat &src, float *dest) {

}
