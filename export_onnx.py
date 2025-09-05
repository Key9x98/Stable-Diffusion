import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from peft import PeftModel
import os

# --- Cấu hình ---
merged_model_dir = "merged_model"
onnx_output_dir = "onnx_merge_lora_model"
base_model_id = "runwayml/stable-diffusion-v1-5" # Cần để tải Text Encoder gốc
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print(f"Sử dụng thiết bị: {device}")

# --- 1. Export UNet ---
print("\n--- Bắt đầu export UNet ---")
unet_dir = os.path.join(onnx_output_dir, "unet")
os.makedirs(unet_dir, exist_ok=True)

unet = UNet2DConditionModel.from_pretrained(
    os.path.join(merged_model_dir, "unet"), 
    torch_dtype=dtype,
    # ================ THÊM DÒNG NÀY ================
    low_cpu_mem_usage=False 
).to(device)

unet_input = (
    torch.randn(1, 4, 64, 64, dtype=dtype).to(device),              # sample
    # Thay thế torch.randn bằng một tensor số nguyên cố định
    torch.tensor([999], dtype=torch.int64).to(device),             # timestep
    torch.randn(1, 77, 768, dtype=dtype).to(device)                 # encoder_hidden_states
)
torch.onnx.export(
    unet,
    unet_input,
    os.path.join(unet_dir, "model.onnx"),
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
    dynamic_axes={
        "sample": {0: "batch_size"},
        "timestep": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
    },
    opset_version=17,
)
print("✅ UNet đã được export sang ONNX.")


# --- 2. Export Text Encoder ---
print("\n--- Bắt đầu export Text Encoder ---")
text_encoder_dir = os.path.join(onnx_output_dir, "text_encoder")
os.makedirs(text_encoder_dir, exist_ok=True)

# Tải Text Encoder gốc
text_encoder = CLIPTextModel.from_pretrained(
    base_model_id, subfolder="text_encoder", torch_dtype=dtype
)

# Áp dụng các trọng số LoRA (adapter) lên model gốc
text_encoder = PeftModel.from_pretrained(
    text_encoder, os.path.join(merged_model_dir, "text_encoder")
).to(device)

# Hợp nhất và dỡ bỏ wrapper của PEFT để có một model PyTorch thuần túy
text_encoder = text_encoder.merge_and_unload()

text_input = torch.randint(1, 1000, (1, 77)).to(device)
torch.onnx.export(
    text_encoder,
    text_input,
    os.path.join(text_encoder_dir, "model.onnx"),
    input_names=["input_ids"],
    output_names=["hidden_states", "pooler_output"],
    dynamic_axes={"input_ids": {0: "batch_size"}},
    opset_version=17,
)
print("✅ Text Encoder đã được export sang ONNX.")


# --- 3. Export VAE Decoder ---
print("\n--- Bắt đầu export VAE Decoder ---")
vae_decoder_dir = os.path.join(onnx_output_dir, "vae_decoder")
os.makedirs(vae_decoder_dir, exist_ok=True)

vae = AutoencoderKL.from_pretrained(
    os.path.join(merged_model_dir, "vae"), torch_dtype=dtype
).to(device)

vae_input = torch.randn(1, 4, 64, 64, dtype=dtype).to(device)
# Chúng ta chỉ cần decoder, nên sẽ trace qua decoder
torch.onnx.export(
    vae.decoder,
    vae_input,
    os.path.join(vae_decoder_dir, "model.onnx"),
    input_names=["latent_sample"],
    output_names=["sample"],
    dynamic_axes={"latent_sample": {0: "batch_size"}},
    opset_version=17,
)
print("✅ VAE Decoder đã được export sang ONNX.")
print("\n🎉 Hoàn tất quá trình export ONNX!")