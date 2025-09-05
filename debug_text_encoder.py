import numpy as np
import tensorrt as trt
from transformers import CLIPTokenizer
from sd_trt_infer import load_engine, infer # Import các hàm helper

print("--- Bắt đầu Debug Text Encoder ---")

# Tải engine Text Encoder FP16
text_encoder_engine = load_engine("trt_models/text_encoder_fp16.plan")

# Tải "ground truth" hidden_states từ PyTorch
pytorch_hidden_states = np.load("debug_hidden_states.npy")

# Tạo lại input_ids giống hệt như trong script test
prompt = "a cute cat, high quality"
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.numpy().astype(np.int32)

# Chạy inference với engine TensorRT
trt_hidden_states, _ = infer(text_encoder_engine, [input_ids])
np.save("debug_trt_hidden_states.npy", trt_hidden_states)
print("✅ Đã chạy và lưu hidden_states từ TensorRT Text Encoder.")

# So sánh kết quả
if np.isnan(trt_hidden_states).any() or np.isinf(trt_hidden_states).any():
    print("❌ LỖI NGHIÊM TRỌNG: Output của Text Encoder TensorRT chứa NaN hoặc Inf!")
else:
    is_close = np.allclose(pytorch_hidden_states, trt_hidden_states, atol=1e-2)
    if is_close:
        print("✅ THÀNH CÔNG: Output của Text Encoder TensorRT khớp với PyTorch!")
    else:
        print("❌ LỖI: Output của Text Encoder TensorRT KHÔNG khớp với PyTorch.")
        diff = np.abs(pytorch_hidden_states - trt_hidden_states).mean()
        print(f"   - Sai số trung bình tuyệt đối: {diff}")