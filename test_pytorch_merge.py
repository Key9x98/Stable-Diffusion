import torch
import numpy as np
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
from PIL import Image

# --- Cấu hình ---
base_model_id = "runwayml/stable-diffusion-v1-5"
merged_model_dir = "./merged_model"
device = "cuda"
dtype = torch.float16

print("--- Bắt đầu tạo Ground Truth từ PyTorch ---")

# --- 1. Tải các thành phần đã được hợp nhất ---
print("Tải Tokenizer...")
tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")

print("Tải Text Encoder và áp dụng LoRA...")
text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder = PeftModel.from_pretrained(text_encoder, f"{merged_model_dir}/text_encoder").to(device)
text_encoder = text_encoder.merge_and_unload() # Hợp nhất thành model thuần túy

print("Tải UNet đã hợp nhất...")
unet = UNet2DConditionModel.from_pretrained(f"{merged_model_dir}/unet", torch_dtype=dtype, low_cpu_mem_usage=False).to(device)

print("Tải VAE...")
vae = AutoencoderKL.from_pretrained(f"{merged_model_dir}/vae", torch_dtype=dtype).to(device)

print("Tải Scheduler...")
scheduler = EulerDiscreteScheduler.from_pretrained(base_model_id, subfolder="scheduler")

# --- 2. Tạo các file debug .npy ---
prompt = "a cute cat, high quality"
input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)

print("Tạo debug_hidden_states.npy...")
with torch.no_grad():
    hidden_states = text_encoder(input_ids)[0].cpu().numpy()
np.save("debug_hidden_states.npy", hidden_states)

print("Tạo debug_initial_latents.npy và debug_pytorch_noise_pred.npy...")
# Tạo latents ban đầu có thể tái tạo được để so sánh chính xác
generator = torch.Generator(device=device).manual_seed(0)
latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype)
np.save("debug_initial_latents.npy", latents.cpu().numpy())

t = torch.tensor([999], dtype=torch.int64).to(device)
with torch.no_grad():
    noise_pred = unet(latents, t, encoder_hidden_states=torch.from_numpy(hidden_states).to(device)).sample.cpu().numpy()
np.save("debug_pytorch_noise_pred.npy", noise_pred)

print("✅ Đã tạo xong các file .npy.")

# --- 3. Tạo ảnh mẫu để kiểm tra bằng mắt thường ---
print("Tạo ảnh mẫu pytorch_merged_output.png...")
num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps)
latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=dtype) # Tái tạo lại latents

with torch.no_grad():
    for t in scheduler.timesteps:
        latent_model_input = latents
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=torch.from_numpy(hidden_states).to(device)).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Decode
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)

image.save("pytorch_merged_output.png")
print("✅ Đã lưu ảnh kết quả từ PyTorch: pytorch_merged_output.png")
print("\n🎉 Hoàn tất! Bây giờ bạn có thể chạy script debug_unet.py.")

'''
❌ LỖI NGHIÊM TRỌNG: Output của UNet TensorRT chứa NaN hoặc Inf!
Đây là bằng chứng không thể chối cãi.
Diễn giải kết quả
Script test_pytorch_merge.py đã thành công: Nó đã tạo ra các file debug_*.npy và một ảnh đẹp (pytorch_merged_output.png). Điều này chứng tỏ model PyTorch sau khi tích hợp LoRA hoạt động hoàn hảo.
Script debug_unet.py đã thất bại thảm hại: Khi bạn đưa chính xác cùng một đầu vào (latents, timestep, hidden_states) vào engine UNet FP16, đầu ra của nó không phải là một dự đoán nhiễu hợp lệ, mà là các giá trị vô nghĩa (NaN - Not a Number, Inf - Infinity).
Kết luận cuối cùng: Vấn đề nằm ở quá trình build engine UNet với cờ --fp16. Các trọng số của UNet (kể cả sau khi đã tích hợp LoRA) không tương thích với việc giảm độ chính xác xuống 16-bit. Các phép tính bên trong engine đã bị sụp đổ, gây ra lỗi số học và tạo ra dữ liệu "rác".
Đây chính là nguyên nhân gốc rễ của việc bạn nhận được ảnh màu xám.

-> thử chuyển unet sang fp32
'''