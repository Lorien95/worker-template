import torch
import numpy as np
import cv2
import io
import asyncio
import base64
import gc
from io import BytesIO
from PIL import Image, ImageOps, PngImagePlugin
import json
from typing import List, Optional, Dict, Tuple

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

# 设置模型文件名
MODEL_PATH = 'migan_traced.pt'

# 检查是否支持 MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 设备")
else:
    device = torch.device("cpu")
    print("MPS 不可用，使用 CPU")

# 加载模型
try:
    model = torch.jit.load(MODEL_PATH, map_location="cpu").to(device)
    print(f"成功加载 TorchScript 模型到 {device}")
except Exception as e:
    print(f"无法加载模型: {e}")
    exit()

model.eval()

def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[-1])  # 移除 data URL 前缀
    return Image.open(BytesIO(img_data))


def decode_base64_to_image(
    encoding: str, gray=False
) -> Tuple[np.array, Optional[np.array], Dict]:
    if encoding.startswith("data:image/") or encoding.startswith(
        "data:application/octet-stream;base64,"
    ):
        encoding = encoding.split(";")[1].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoding)))

    alpha_channel = None
    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass
    # exif_transpose will remove exif rotate info，we must call image.info after exif_transpose
    infos = image.info

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    return np_img, alpha_channel, infos


# 图像预处理函数
def preprocess_image(image_base64, mask_base64):
    # 解码 base64 图像 分离通道
    image, alpha_channel, infos = decode_base64_to_image(image_base64)
    mask, _, _ = decode_base64_to_image(mask_base64, gray=True)
    #二值化
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    rgb_np_img = process_image_new(image, mask)
    image_bgr2 = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('saved_image22.png', image_bgr2)
    # # 从 base64 加载原图和 mask
    # image = base64_to_image(image_base64).convert('RGB')
    # mask = base64_to_image(mask_base64).convert('L')
    torch_gc()

    rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    rgb_res = concat_alpha_channel(rgb_np_img, alpha_channel)

    image_bgr24 = cv2.cvtColor(rgb_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite('saved_image24.png', image_bgr24)
    ext = "png"
    res_img_bytes = pil_to_bytes(
        Image.fromarray(rgb_res),
        ext=ext,
        quality=95,
        infos=infos,
    )
    return res_img_bytes

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def concat_alpha_channel(rgb_np_img, alpha_channel) -> np.ndarray:
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != rgb_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(rgb_np_img.shape[1], rgb_np_img.shape[0])
            )
        rgb_np_img = np.concatenate(
            (rgb_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )
    return rgb_np_img


def pil_to_bytes(pil_img, ext: str, quality: int = 95, infos={}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in infos.items() if v is not None}
        if ext == "jpg":
            ext = "jpeg"
        if "png" == ext.lower() and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(output, format=ext, quality=quality, **kwargs)
        image_bytes = output.getvalue()
    return image_bytes

@torch.no_grad()
def process_image_new(image, mask):
    """
    images: [H, W, C] RGB, not normalized
    masks: [H, W]
    return: BGR IMAGE
    """
    print("Hello, World1!")
    image_bgr4 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('saved_image4.png', image_bgr4)
    if image.shape[0] == 512 and image.shape[1] == 512:
        return forward(image, mask)

    boxes = boxes_from_mask(mask)
    crop_result = []
    for box in boxes:
        crop_image, crop_mask, crop_box = _crop_box(image, mask, box)
        origin_size = crop_image.shape[:2]
        resize_image = resize_max_size(crop_image, size_limit=512)
        resize_mask = resize_max_size(crop_mask, size_limit=512)
        print("Input image shape:", resize_image.shape)
        print("Input image dtype:", resize_image.dtype)
        inpaint_result = _pad_forward(resize_image, resize_mask)

        # only paste masked area result
        inpaint_result = cv2.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        original_pixel_indices = crop_mask < 127
        inpaint_result[original_pixel_indices] = crop_image[:, :, ::-1][
            original_pixel_indices
        ]

        crop_result.append((inpaint_result, crop_box))

    inpaint_result = image[:, :, ::-1].copy()
    for crop_image, crop_box in crop_result:
        x1, y1, x2, y2 = crop_box
        inpaint_result[y1:y2, x1:x2, :] = crop_image

    image_bgr8 = cv2.cvtColor(inpaint_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('saved_image8.png', image_bgr8)

    return inpaint_result

pad_mod=512
pad_to_square=True
min_size=512
def _pad_forward(image, mask):
    print("Hello, World13!")

    origin_height, origin_width = image.shape[:2]
    pad_image = pad_img_to_modulo(
        image, mod=pad_mod, square=pad_to_square, min_size=min_size
    )
    pad_mask = pad_img_to_modulo(
        mask, mod=pad_mod, square=pad_to_square, min_size=min_size
    )

    # logger.info(f"final forward pad size: {pad_image.shape}")

    # image, mask = forward_pre_process(image, mask, config)

    result = forward(pad_image, pad_mask)
    result = result[0:origin_height, 0:origin_width, :]

    # result, image, mask = forward_post_process(result, image, mask, config)

    if True:
        mask = mask[:, :, np.newaxis]
        result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))
    return result
    
def pad_img_to_modulo(
    img: np.ndarray, mod: int, square: bool = False, min_size: Optional[int] = None
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def resize_max_size(
    np_img, size_limit: int, interpolation=cv2.INTER_CUBIC
) -> np.ndarray:
    # Resize image's longer size to size_limit if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(np_img, dsize=(new_w, new_h), interpolation=interpolation)
    else:
        return np_img

def boxes_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    Args:
        mask: (h, w, 1)  0~255

    Returns:

    """
    height, width = mask.shape[:2]
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box = np.array([x, y, x + w, y + h]).astype(int)

        box[::2] = np.clip(box[::2], 0, width)
        box[1::2] = np.clip(box[1::2], 0, height)
        boxes.append(box)

    return boxes



def _crop_box(image, mask, box):
    """

    Args:
        image: [H, W, C] RGB
        mask: [H, W, 1]
        box: [left,top,right,bottom]

    Returns:
        BGR IMAGE, (l, r, r, b)
    """
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    img_h, img_w = image.shape[:2]

    w = box_w + 256
    h = box_h + 256

    _l = cx - w // 2
    _r = cx + w // 2
    _t = cy - h // 2
    _b = cy + h // 2

    l = max(_l, 0)
    r = min(_r, img_w)
    t = max(_t, 0)
    b = min(_b, img_h)

    # try to get more context when crop around image edge
    if _l < 0:
        r += abs(_l)
    if _r > img_w:
        l -= _r - img_w
    if _t < 0:
        b += abs(_t)
    if _b > img_h:
        t -= _b - img_h

    l = max(l, 0)
    r = min(r, img_w)
    t = max(t, 0)
    b = min(b, img_h)

    crop_img = image[t:b, l:r, :]
    crop_mask = mask[t:b, l:r]

    # logger.info(f"box size: ({box_h},{box_w}) crop size: {crop_img.shape}")

    return crop_img, crop_mask, [l, t, r, b]



def forward( image, mask):
    """Input images and output images have same size
    images: [H, W, C] RGB
    masks: [H, W] mask area == 255
    return: BGR IMAGE
    """
    print("Hello, World2!")

    print("Input image shape:", image.shape)
    print("Input image dtype:", image.dtype)
    image_bgr4 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('saved_image5.png', image_bgr4)


    image = norm_img(image)  # [0, 1]
    image = image * 2 - 1  # [0, 1] -> [-1, 1]



    mask = (mask > 120) * 255
    mask = norm_img(mask)

    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    erased_img = image * (1 - mask)
    input_image = torch.cat([0.5 - mask, erased_img], dim=1)


    output = model(input_image)


    output = (
        (output.permute(0, 2, 3, 1) * 127.5 + 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    output = output[0].cpu().numpy()
    cur_res = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    image_bgr6 = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite('saved_image6.png', image_bgr6)

    return cur_res




def process_images(image_base64, mask_base64):
    # 准备输入数据
    result = preprocess_image(image_base64, mask_base64)
    # 将字节数据转换为 base64 编码的字符串
    base64_encoded = base64.b64encode(result).decode('utf-8')
    # # 进行推理
    # with torch.no_grad():
    #     output = model(input_data)

    # # 将输出移回 CPU 进行后处理
    # output = output.cpu()

    # # 处理输出
    # output = (output.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).numpy()

    # # 转换为BGR
    # output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # # 将结果转换为 base64
    # _, buffer = cv2.imencode('.png', output_bgr)
    # result_base64 = base64.b64encode(buffer).decode('utf-8')

    return base64_encoded

# 读取本地 JSON 文件
with open('test.json', 'r') as f:
    data = json.load(f)

# 从 JSON 中获取 base64 编码的图像和掩码
image_base64 = data['image']
mask_base64 = data['mask']

# 处理图像
result_base64 = process_images(image_base64, mask_base64)

# 将结果保存到文件
with open('result.json', 'w') as f:
    json.dump({'result': result_base64}, f)

print("处理完成，结果已保存为 result.json")