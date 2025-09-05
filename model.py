from diffusers import StableDiffusionPipeline
import torch, time

device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_type=torch.float16, device = device)

prompt = "a photo of a futuristic city at sunset, ultra realistic"
start = time.time()
image = pipe(prompt).images[0]
end = time.time()

print(f"PyTorch Inference time: {end-start:.2f}s")
image.save("sd_original.png")
