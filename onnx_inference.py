import tensorflow as tf
import numpy as np
import cv2

if __name__ == "__main__":
    img = cv2.imread("test/fixtures/image_F1.bmp").astype(np.float32)
    # img = cv2.resize(((img/255.0)-0.5)*2.0, (224, 224))
    # img = img[:,:,::-1]
    img_t = np.transpose(img, (2,0,1))

    mobilePath = "assets/models/2.7_80x80_MiniFASNetV2_bhoke.tflite"
    model = tf.lite.Interpreter(model_path=mobilePath)
    model.allocate_tensors()

    inputLayer = model.get_input_details()[0]["index"]
    outputLayer = model.get_output_details()[0]["index"]

    model.set_tensor(inputLayer, [img_t])
    model.invoke()
    preds = model.get_tensor(outputLayer)[0]
    print(preds)