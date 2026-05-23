import torch
import numpy as np
import onnxruntime
import onnx
import urllib.request
import time
import json
import glob
import os


import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def download_model_and_data():
    onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz"
    imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

    urllib.request.urlretrieve(onnx_model_url, filename="resnet50v2.tar.gz")
    urllib.request.urlretrieve(imagenet_labels_url, filename="imagenet-simple-labels.json")

test_data_num = 3
inputs = []
ref_outputs = []
def get_input_and_inf_output():
    test_data_dir = "./data/resnet50v2/test_data_set"
    


    for i in range(test_data_num):
        input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            inputs.append(onnx.numpy_helper.to_array(tensor))
    
    print("loaded {} inputs successfully".format(test_data_num))


    for i in range(test_data_num):
        output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            ref_outputs.append(onnx.numpy_helper.to_array(tensor))
    
    print('loaded {} reference outputs sucessfully'.format(test_data_num))
            

def infer_by_onnx_runtime():
    session = onnxruntime.InferenceSession('model/resnet50v2.onnx', None)
    input_name = session.get_inputs()[0].name
    print('Input Name', input_name)

    out_puts = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]
    print('Predicted {} results'.format(len(out_puts)))

    for ref_o, o in zip(ref_outputs, out_puts):
        np.testing.assert_almost_equal(ref_o, o, 4)
    print('ONNX Runtime outputs are similar to reference outputs!')


if __name__ == "__main__":
    #download_model_and_data()
    get_input_and_inf_output()
    print(len(inputs), len(ref_outputs))

    infer_by_onnx_runtime()

