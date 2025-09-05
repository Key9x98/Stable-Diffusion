
# Chuyển Stable Diffusion sang TensorRT

## Link tài nguyên
- Google Drive chứa model: [Link](https://drive.google.com/drive/folders/1pDcYanTodx5XeG8bebcS91jU5iS1g4DV?usp=sharing)  
- Báo cáo: [Link](https://drive.google.com/file/d/1OReWEegMR0NEJiR1BRsTZpT_E2Yf0Yq_/view?usp=sharing)  

## Export ONNX
```bash
python -m optimum.exporters.onnx --model runwayml/stable-diffusion-v1-5 onnx_model/
````
hoặc chạy file export_onnx.py
## Chuyển ONNX sang TensorRT

### UNet

```bash
trtexec --onnx=onnx_model/unet/model.onnx --saveEngine=trt_models/unet_fp16.plan --fp16 --shapes=sample:1x4x64x64,timestep:scalar,encoder_hidden_states:1x77x768
```

### Text Encoder

```bash
trtexec --onnx=onnx_model/text_encoder/model.onnx --saveEngine=trt_models/text_encoder_fp16.plan --fp16 --shapes=input_ids:1x77
```

### VAE Encoder

```bash
trtexec --onnx=onnx_model/vae_encoder/model.onnx --saveEngine=trt_models/vae_encoder_fp16.plan --fp16 --shapes=sample:1x3x512x512
```

### VAE Decoder

```bash
trtexec --onnx=onnx_model/vae_decoder/model.onnx --saveEngine=trt_models/vae_decoder_fp16.plan --fp16 --shapes=latent_sample:1x4x64x64
```
