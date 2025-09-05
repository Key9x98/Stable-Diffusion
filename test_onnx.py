import onnxruntime as ort

session = ort.InferenceSession("onnx_model/unet/model.onnx", providers=["CUDAExecutionProvider"])
print("Inputs:", [inp.name for inp in session.get_inputs()])
print("Outputs:", [out.name for out in session.get_outputs()])

import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("GPU compute capability:", torch.cuda.get_device_capability())
