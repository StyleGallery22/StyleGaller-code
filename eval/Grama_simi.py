import os
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_image(image_path, size=None):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping invalid or unreadable image: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size:
        image = cv2.resize(image, size)
    image = image / 255.0
    return image

def extract_features(image, model):
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        features = model(image)
    return features

def gram_matrix(features):
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    g_matrix = torch.bmm(features, features.transpose(1, 2))
    return g_matrix / (c * h * w)

def split_image(image, block_size):
    h, w, _ = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                blocks.append(block)
    return blocks

def calculate_style_loss(parent_folder, block_size):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)

    total_average_loss = 0.0
    folder_count = 0

    for root, dirs, files in os.walk(parent_folder):
        content_image_path = os.path.join(root, 'RE-SD.png')
        if not os.path.isfile(content_image_path):
            continue

        content_image = load_image(content_image_path)
        if content_image is None:
            continue
        style_image_paths = [
            os.path.join(root, f)
            for f in files
            if f.startswith('style') and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and ':' not in f  
        ]
        if not style_image_paths:
            continue
        content_blocks = split_image(content_image, block_size)
        if not content_blocks:
            continue

        content_grams = []
        for block in content_blocks:
            features = extract_features(block, model)
            g = gram_matrix(features)
            content_grams.append(g)

        total_loss = 0.0

        style_grams = []
        for style_image_path in style_image_paths:
            style_image = load_image(style_image_path)
            if style_image is None:
                continue
            style_blocks = split_image(style_image, block_size)
            for block in style_blocks:
                features = extract_features(block, model)
                g = gram_matrix(features)
                style_grams.append(g)

        if not style_grams:
            continue

        for c_gram in content_grams:
            distances = [torch.norm(c_gram - s_gram, p=1).item() for s_gram in style_grams]
            min_distance = min(distances)
            total_loss += min_distance

        average_loss = total_loss / len(content_grams)
        total_average_loss += average_loss
        folder_count += 1

        print(f"📁 Folder: {root}, Average Style Loss: {average_loss:.4f}")
    final_average_loss = total_average_loss / folder_count if folder_count > 0 else 0
    return final_average_loss

if __name__ == "__main__":
    parent_folder = 'eval_data'  
    block_size = 128          

    average_loss = calculate_style_loss(parent_folder, block_size)
    print(f"\n✅ Final Average Style Loss: {average_loss:.4f}")
