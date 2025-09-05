import onnx

models = {
    "unet": "onnx_model/unet/model.onnx",
    "text_encoder": "onnx_model/text_encoder/model.onnx",
    "vae_encoder": "onnx_model/vae_encoder/model.onnx",
    "vae_decoder": "onnx_model/vae_decoder/model.onnx",
}

for name, path in models.items():
    model = onnx.load(path)
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    print(f"\n=== {name} ===")
    print("Inputs :", inputs)
    print("Outputs:", outputs)
