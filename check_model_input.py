import onnx

def check_model_inputs(onnx_path):
    """Kiểm tra input shapes của ONNX model"""
    model = onnx.load(onnx_path)
    print(f"\n=== {onnx_path} ===")
    
    for input_tensor in model.graph.input:
        print(f"Input: {input_tensor.name}")
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        print(f"  Shape: {' x '.join(shape)}")
        print()

# Kiểm tra tất cả models
models = [
    "onnx_model/unet/model.onnx",
    "onnx_model/text_encoder/model.onnx", 
    "onnx_model/vae_encoder/model.onnx",
    "onnx_model/vae_decoder/model.onnx"
]

for model_path in models:
    try:
        check_model_inputs(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")