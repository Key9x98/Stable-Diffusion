import numpy as np
import tensorrt as trt
import torch
from sd_trt_infer import load_engine, infer # Import các hàm helper từ script chính của bạn

print("--- Bắt đầu Debug UNet ---")
# Tải engine UNet FP16
unet_engine = load_engine("trt_merge_lora_model/unet_fp16.plan")

# Tải các input "ground truth"
hidden_states = np.load("debug_hidden_states.npy")
latents = np.load("debug_initial_latents.npy")
timestep = np.array([999], dtype=np.int32)

# Chạy inference với engine TensorRT
trt_noise_pred = infer(unet_engine, [latents, timestep, hidden_states])[0]
np.save("debug_trt_noise_pred.npy", trt_noise_pred)
print("✅ Đã chạy và lưu noise_pred từ TensorRT UNet.")

# Tải output của PyTorch để so sánh
pytorch_noise_pred = np.load("debug_pytorch_noise_pred.npy")

# So sánh kết quả
if np.isnan(trt_noise_pred).any() or np.isinf(trt_noise_pred).any():
    print("❌ LỖI NGHIÊM TRỌNG: Output của UNet TensorRT chứa NaN hoặc Inf!")
else:
    # atol (absolute tolerance) cao hơn một chút cho FP16
    is_close = np.allclose(pytorch_noise_pred, trt_noise_pred, atol=1e-2) 
    if is_close:
        print("✅ THÀNH CÔNG: Output của UNet TensorRT khớp với PyTorch!")
    else:
        print("❌ LỖI: Output của UNet TensorRT KHÔNG khớp với PyTorch.")
        diff = np.abs(pytorch_noise_pred - trt_noise_pred).mean()
        print(f"   - Sai số trung bình tuyệt đối: {diff}")
