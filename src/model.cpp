#include <sys/time.h>   
#include <string>

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

/**
 * @brief Converts timeval to milliseconds
 */
int to_millis(struct timeval time) { return (time.tv_sec * (uint64_t)1000) + (time.tv_usec / 1000); }

/**
 * @brief Returns max element index
 */
int argmax(float *output, int classes) {
    float max = 0;
    int max_index = 0;
    for (int class_id=0; class_id<classes; class_id++) {
        if (output[class_id]>max) {
            max = output[class_id];
            max_index = class_id;
        }
    }

    return max_index;
}

/**
 * @brief Loads the model into memory
 * 
 * @param model_path relative path to input model
 * @return KLError loading status
 */
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

/**
 * @brief Performs model inference on an image
 * 
 * @param img_path relative path to input image
 * @return float genuine score
 */
float Model::inference(const char *img_path) {
    if(model_==nullptr)
    { 
        Logger::error("Model has not been properly looaded.");
        throw KLError::MODEL_INFERENCE_ERROR;
    }

    float *input = interpreter_->typed_input_tensor<float>(0);

    // Read image
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if(img.empty())
    { 
        Logger::error("Failed to load image.");
        throw KLError::MODEL_INFERENCE_ERROR;
    }

    // Pre-process input 
    cv::resize(img, img, cv::Size(model_height, model_width), 0, 0, CV_INTER_LINEAR);
    convert_image(img, input);

    // Inference
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    if (interpreter_->Invoke() != kTfLiteOk) {
        Logger::error("Failed to invoke tflite!");
        throw KLError::MODEL_INFERENCE_ERROR;
    }
    gettimeofday(&stop_time, nullptr);
    Logger::debug("Inference time: " +  std::to_string(to_millis(stop_time) - to_millis(start_time)) + "ms");


    float *output = interpreter_->typed_output_tensor<float>(0);
    std::string label = argmax(output, 3) == 1 ? "Real" : "Fake";
    Logger::info("Image is " + label + " face");

    return output[1];
}

/**
 * @brief Converts OpenCV image to input tensor arrray
 * 
 * @param src OpenCV input image 
 * @param dest destination float array
 */
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

