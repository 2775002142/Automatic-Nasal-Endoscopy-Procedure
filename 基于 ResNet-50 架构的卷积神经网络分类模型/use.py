# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

def create_model(model_name="resnet50", num_classes=3):
    if model_name == "resnet50":
        model = models.resnet50(weights=None) 
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    return model

CLASS_NAMES = ["囊肿", "正常鼻咽", "鼻咽癌"] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(model_name="resnet50", num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"错误：文件不存在 - {img_path}")
        return None

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    class_idx = predicted.item()
    confidence = probabilities[0][class_idx].item()
    pred_class = CLASS_NAMES[class_idx]

    print(f"\n图片: {img_path.name}")
    print(f"预测类别: {pred_class} (索引: {class_idx})")
    print(f"置信度: {confidence:.4f}")
    print("各类别概率:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {probabilities[0][i]:.4f}")

    return pred_class, confidence

if __name__ == "__main__":
    test_img1 = r"D:\Trae CN\projects\Classification\图像数据集\鼻咽癌\AE240314_彭强生_143972\AEE3ELT8H6LM1.JPG"
    predict(test_img1)
    test_img2 = r"D:\Trae CN\projects\Classification\图像数据集\囊肿\AE211209_赖奕珊_67432\AEBC9DQOHJVF6.JPG"
    predict(test_img2)
    test_img3 = r"D:\Trae CN\projects\Classification\图像数据集\正常鼻咽\AE210105_陈美华_76358\515N5F3T.JPG"
    predict(test_img3)