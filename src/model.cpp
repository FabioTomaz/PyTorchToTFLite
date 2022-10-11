#include "model.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"
#include "kl_error.hpp"
#include "logger.hpp"


#define TFLITE_MINIMAL_CHECK(x, msg)        \
  if (!(x)) {                               \
    Logger::error(msg);                     \
    return KLError::MODEL_LOAD_ERROR;       \
  }                                         


KLError Model::init(const char *model_path) {
    Logger::info("Loading model...");

    // Load model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path);
    TFLITE_MINIMAL_CHECK(model_ != nullptr, "Failed to load model from file");

    // Build and set up the interpreter with the InterpreterBuilder
    // to allocate memory and read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    TFLITE_MINIMAL_CHECK(interpreter_ != nullptr, "Failed to build interpreter");

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk, "Failed to allocate tensors");

    Logger::info("Finished loading model.");
    return KLError::NONE;
}

float Model::inference(const char *img_path) {
    float *input = interpreter_->typed_input_tensor<float>(0);
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    convert_image(img, input);
    interpreter_->Invoke();
    float *output = interpreter_->typed_output_tensor<float>(0);
    return output[1];
}

void Model::convert_image(const cv::Mat &src, float *dest) {
    int i = 0;
    for (int channel = 0; channel < src.channels(); channel++)
    {
        for (int row = 0; row < src.rows; row++)
        {
            for (int col = 0; col < src.cols; col++)
            {
                dest[i++] = src.at<cv::Vec3b>(row, col)[channel];
            }
        }
    }
}
