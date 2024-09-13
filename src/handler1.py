""" Handler file for image inpainting using MI-GAN model. """

import runpod
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import json

# 设置模型文件名
MODEL_PATH = 'migan_traced.pt'

# 检查是否支持 MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 设备")
else:
    device = torch.device("cuda")
    print("MPS 不可用，使用 CPU")

# 加载模型
try:
    model = torch.jit.load(MODEL_PATH, map_location="cpu").to(device)
    print(f"成功加载 TorchScript 模型到 {device}")
except Exception as e:
    print(f"无法加载模型: {e}")
    exit()

model.eval()

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[-1])  # 移除 data URL 前缀
    return Image.open(BytesIO(img_data))

def preprocess_image(image_base64, mask_base64):
    # 从 base64 加载原图和 mask
    image = base64_to_image(image_base64).convert('RGB')
    mask = base64_to_image(mask_base64).convert('L')
    
    # 调整图像和mask大小为512x512
    image = image.resize((512, 512), Image.LANCZOS)
    mask = mask.resize((512, 512), Image.LANCZOS)
    
    # 转换为numpy数组
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # 对mask进行二值化处理
    mask_np = (mask_np > 120).astype(np.uint8) * 255
    
    # 归一化处理
    image_norm = norm_img(image_np)
    mask_norm = norm_img(mask_np)
    
    # 将图像从 [0, 1] 转换为 [-1, 1]
    image_norm = image_norm * 2 - 1
    
    # 转换为张量并移动到适当的设备
    image_tensor = torch.from_numpy(image_norm).float().to(device)
    mask_tensor = torch.from_numpy(mask_norm).float().to(device)

    # 创建擦除的图像
    erased_img = image_tensor * (1 - mask_tensor)

    # 组合输入
    input_tensor = torch.cat([0.5 - mask_tensor, erased_img], dim=0)

    return input_tensor.unsqueeze(0).to(device)  # 添加batch维度并确保在正确的设备上

def process_images(image_base64, mask_base64):
    # 准备输入数据
    input_data = preprocess_image(image_base64, mask_base64)

    # 进行推理
    with torch.no_grad():
        output = model(input_data)

    # 将输出移回 CPU 进行后处理
    output = output.cpu()

    # 处理输出
    output = (output.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).numpy()

    # 转换为BGR
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # 将结果转换为 base64
    _, buffer = cv2.imencode('.png', output_bgr)
    result_base64 = base64.b64encode(buffer).decode('utf-8')

    return result_base64

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    # 从输入中获取图像和掩码的 base64 编码
    image_base64 = job_input.get('image')
    mask_base64 = job_input.get('mask')

    if not image_base64 or not mask_base64:
        return {"error": "Missing image or mask data"}

    try:
        # 处理图像
        result_base64 = process_images(image_base64, mask_base64)
        
        # 返回结果
        return {"result": result_base64}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})