import os
import subprocess

# Thư mục chứa ONNX model
onnx_dir = "onnx_model"

# Thư mục lưu TensorRT engine
trt_dir = "trt_models"
os.makedirs(trt_dir, exist_ok=True)

# Thư mục lưu log
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Danh sách model cần convert
models = [
    {
        "name": "unet",
        "onnx": f"{onnx_dir}/unet/model.onnx",
        "engine": f"{trt_dir}/unet_fp16.plan",
        "shapes": "sample:1x4x64x64,timestep:1,encoder_hidden_states:1x77x768"
    },
    {
        "name": "text_encoder",
        "onnx": f"{onnx_dir}/text_encoder/model.onnx",
        "engine": f"{trt_dir}/text_encoder_fp16.plan",
        "shapes": "input_ids:1x77"
    },
    {
        "name": "vae_encoder",
        "onnx": f"{onnx_dir}/vae_encoder/model.onnx",
        "engine": f"{trt_dir}/vae_encoder_fp16.plan",
        "shapes": "sample:1x3x512x512"
    },
    {
        "name": "vae_decoder",
        "onnx": f"{onnx_dir}/vae_decoder/model.onnx",
        "engine": f"{trt_dir}/vae_decoder_fp16.plan",
        "shapes": "latent_sample:1x4x64x64"
    }
]


def run_trtexec(model):
    log_file = os.path.join(log_dir, f"{model['name']}.log")

    cmd = [
        "trtexec",
        f"--onnx={model['onnx']}",
        f"--saveEngine={model['engine']}",
        "--fp16",
        "--workspace=4096",
        f"--shapes={model['shapes']}"
    ]

    print(f"\n=== Converting {model['name']} ===")
    print(" ".join(cmd))
    print(f"📄 Log: {log_file}")

    with open(log_file, "w") as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=f, shell=True, check=True)
            print(f"✅ {model['name']} converted successfully.")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to convert {model['name']}. Check log at {log_file}")


if __name__ == "__main__":
    for m in models:
        run_trtexec(m)
    print("\n🎉 Done! Check trt_models/ for engines and logs/ for build logs.")
