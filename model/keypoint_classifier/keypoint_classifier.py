#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # local/dev fallback
    from tensorflow.lite.python.interpreter import Interpreter


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = Interpreter(model_path=model_path,
                                       num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        probabilities = np.squeeze(result)
        result_index = int(np.argmax(probabilities))
        confidence_score = float(probabilities[result_index])

        return result_index, confidence_score
