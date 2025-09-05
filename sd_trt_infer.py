import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from PIL import Image
import time, os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ENGINE_DIR = "trt_models"
PROMPT = "a futuristic city at sunset, ultra realistic"
DEVICE = "cuda"

# ====== Helper: Load engine ======
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# ====== Helper: Allocate buffers ======
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        shape = engine.get_tensor_shape(binding)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        
        # Bọc kết quả bằng int() để đảm bảo nó là số nguyên
        size = int(torch.prod(torch.tensor(shape)).item())

        # Allocate host & device
        host_mem = np.empty(size, dtype=dtype).reshape(shape)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((binding, host_mem, device_mem))
        else:
            outputs.append((binding, host_mem, device_mem))
    return inputs, outputs, bindings, stream


# ====== Run inference ======
def infer(engine, inputs_data):
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Copy inputs
    for (name, host_mem, device_mem), inp in zip(inputs, inputs_data):
        np.copyto(host_mem, inp)
        cuda.memcpy_htod(device_mem, host_mem)
        context.set_tensor_address(name, int(device_mem))

    # Set outputs
    for name, host_mem, device_mem in outputs:
        context.set_tensor_address(name, int(device_mem))

    context.execute_async_v3(stream.handle)
    stream.synchronize()

    results = []
    for name, host_mem, device_mem in outputs:
        cuda.memcpy_dtoh(host_mem, device_mem)
        results.append(np.copy(host_mem))
    return results

# ====== Main Pipeline ======
def main():
    print("Loading pipeline components (scheduler, tokenizer, etc.)...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_type=torch.float16)
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer

    # Load TRT engines
    text_encoder_engine = load_engine(os.path.join(ENGINE_DIR, "text_encoder_fp32.plan"))
    unet_engine = load_engine(os.path.join(ENGINE_DIR, "unet_fp32.plan"))
    vae_decoder_engine = load_engine(os.path.join(ENGINE_DIR, "vae_decoder_fp16.plan"))

    # Encode prompt
    print("Encoding prompt...")
    text_inputs = tokenizer(PROMPT, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
    input_ids = text_inputs.input_ids.numpy().astype(np.int32)
    hidden_states, _ = infer(text_encoder_engine, [input_ids])  # lấy hidden states

    hidden_states = hidden_states.astype(np.float32) # Chuyển 1 lần duy nhất

    # Diffusion loop
    print("Running diffusion with UNet TRT...")
    # TẠO LATENTS LÀ FP32 NGAY TỪ ĐẦU
    latents = torch.randn((1, 4, 64, 64), device=DEVICE, dtype=torch.float32).cpu().numpy()
    scheduler.set_timesteps(50)
    timesteps = scheduler.timesteps.numpy()

    for t in timesteps:
        timestep = np.array([t], dtype=np.int32)
        noise_pred = infer(unet_engine, [latents, timestep, hidden_states])[0]
        latents = scheduler.step(
            torch.from_numpy(noise_pred),
            torch.tensor(t),
            torch.from_numpy(latents)
        ).prev_sample.numpy()

    # Decode image
    print("Decoding with VAE decoder TRT...")
    # Chuyển latents cuối cùng về FP16 cho VAE FP16
    images = infer(vae_decoder_engine, [latents.astype(np.float16)])[0]
    # Lấy ra ảnh đầu tiên từ batch
    image = images[0]
    
    # 1. Chuyển đổi khoảng giá trị từ [-1, 1] về [0, 255]
    image = np.clip(image / 2 + 0.5, 0, 1)
    image = (image * 255).round().astype(np.uint8)

    # 2. Thay đổi trật tự các chiều từ (C, H, W) sang (H, W, C) mà Pillow cần
    #    Output của VAE là (3, 512, 512) -> Cần chuyển thành (512, 512, 3)
    image = image.transpose((1, 2, 0))

    # 3. Tạo ảnh từ mảng đã xử lý
    image = Image.fromarray(image)

    out_path = "trt_output.png"
    image.save(out_path)
    print(f"✅ Saved generated image at {out_path}")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time()-start:.2f}s")
