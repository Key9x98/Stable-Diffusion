import torch
from diffusers import StableDiffusionPipeline

# --- Cấu hình ---
base_model_id = "runwayml/stable-diffusion-v1-5"
lora_path = r"D:\workspace\Stable-Diffusion\lora\cat_20230627113759.safetensors"
output_dir = "merged_model" # Thư mục để lưu model đã tích hợp

# --- Tải pipeline gốc ---
print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_type=torch.float16)

# --- Tải và áp dụng LoRA ---
print(f"Loading and fusing LoRA weights from: {lora_path}")
pipe.load_lora_weights(lora_path)
pipe.fuse_lora() # Hợp nhất vĩnh viễn LoRA vào các lớp của UNet và Text Encoder

# --- Lưu các thành phần đã được hợp nhất ---
print(f"Saving merged components to: {output_dir}")
pipe.unet.save_pretrained(f"{output_dir}/unet")
pipe.text_encoder.save_pretrained(f"{output_dir}/text_encoder")
pipe.vae.save_pretrained(f"{output_dir}/vae") # VAE thường không bị ảnh hưởng bởi LoRA nhưng cứ lưu lại cho đủ bộ

print("✅ Merging complete!")