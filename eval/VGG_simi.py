import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
vgg19 = models.vgg19(pretrained=True).features.to(device)
vgg19.eval()

feature_layers = {
    'texture': 5,
    'color': 28,
    'beginner_style': 10,
    'intermediate_style': 19,
}

def get_preprocessor():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def split_image_into_blocks(image, block_size):
    width, height = image.size
    blocks, positions = [], []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            box = (x, y, min(x + block_size, width), min(y + block_size, height))
            block = image.crop(box)
            if block.size != (block_size, block_size):
                block = block.resize((block_size, block_size), Image.LANCZOS)
            blocks.append(block)
            positions.append((x, y))
    grid_width = (width + block_size - 1) // block_size
    grid_height = (height + block_size - 1) // block_size
    grid_shape = (grid_height, grid_width)
    return blocks, positions, grid_shape

def preprocess_all_blocks(blocks):
    preprocessor = get_preprocessor()
    tensors = []
    for block in blocks:
        tensor = preprocessor(block)
        tensors.append(tensor)
    return torch.stack(tensors).to(device)

def extract_features(batch_tensor, layer_name, batch_size):
    target_layer = feature_layers[layer_name]
    features = []
    for i in range(0, batch_tensor.size(0), batch_size):
        batch = batch_tensor[i:i + batch_size]
        with torch.no_grad():
            x = batch
            for idx, layer in enumerate(vgg19):
                x = layer(x)
                if idx == target_layer:
                    features.append(x)
                    break
    return torch.cat(features, dim=0)

def process_single_image(img_path, block_size=32, batch_size=32):
    print(f"Processing single image: {img_path}")
    try:
        img = Image.open(img_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"❌ Skipping invalid image: {img_path}")
        return None, None, None, None, None
    print(f"Image size: {img.size}")
    blocks, positions, grid_shape = split_image_into_blocks(img, block_size)
    batch_tensor = preprocess_all_blocks(blocks)
    texture_features = extract_features(batch_tensor, 'texture', batch_size)
    color_features = extract_features(batch_tensor, 'color', batch_size)
    beginner_style_features = extract_features(batch_tensor, 'beginner_style', batch_size)
    intermediate_style_features = extract_features(batch_tensor, 'intermediate_style', batch_size)
    return texture_features, color_features, beginner_style_features, intermediate_style_features, blocks

def process_multiple_images(img_paths, block_size=32, batch_size=32):
    all_blocks = []
    print(f"Processing {len(img_paths)} images...")
    for img_idx, img_path in enumerate(img_paths):
        if ':' in img_path: 
            print(f"⚠️ Skipping invalid file (colon found): {img_path}")
            continue
        print(f"\nProcessing image {img_idx + 1}/{len(img_paths)}: {img_path}")
        try:
            img = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"❌ Skipping invalid image: {img_path}")
            continue
        blocks, positions, grid_shape = split_image_into_blocks(img, block_size)
        all_blocks.extend(blocks)
    if not all_blocks:
        raise RuntimeError("No valid image blocks found in the provided paths.")
    batch_tensor = preprocess_all_blocks(all_blocks)
    texture_features = extract_features(batch_tensor, 'texture', batch_size)
    color_features = extract_features(batch_tensor, 'color', batch_size)
    beginner_style_features = extract_features(batch_tensor, 'beginner_style', batch_size)
    intermediate_style_features = extract_features(batch_tensor, 'intermediate_style', batch_size)
    return texture_features, color_features, beginner_style_features, intermediate_style_features, all_blocks

def calculate_cosine_similarity_double_chunked(features1, features2, chunk_size1=100, chunk_size2=100):
    print(f"Calculating cosine similarities using double chunking...")
    features1_flat = features1.view(features1.size(0), -1)
    features2_flat = features2.view(features2.size(0), -1)
    features1_norm = F.normalize(features1_flat, p=2, dim=1)
    features2_norm = F.normalize(features2_flat, p=2, dim=1)
    N1, N2 = features1_norm.size(0), features2_norm.size(0)
    max_similarities = torch.full((N1,), float('-inf'), device=device)
    closest_indices = torch.zeros(N1, dtype=torch.long, device=device)
    for i in range(0, N1, chunk_size1):
        end_i = min(i + chunk_size1, N1)
        chunk1 = features1_norm[i:end_i]
        chunk1_max_similarities = torch.full((chunk1.size(0),), float('-inf'), device=device)
        chunk1_closest_indices = torch.zeros(chunk1.size(0), dtype=torch.long, device=device)
        for j in range(0, N2, chunk_size2):
            end_j = min(j + chunk_size2, N2)
            chunk2 = features2_norm[j:end_j]
            with torch.no_grad():
                cosine_similarities = torch.mm(chunk1, chunk2.t())
                chunk_max_similarities, chunk_max_indices = torch.max(cosine_similarities, dim=1)
                update_mask = chunk_max_similarities > chunk1_max_similarities
                chunk1_closest_indices[update_mask] = chunk_max_indices[update_mask] + j
                chunk1_max_similarities = torch.max(chunk1_max_similarities, chunk_max_similarities)
        max_similarities[i:end_i] = chunk1_max_similarities
        closest_indices[i:end_i] = chunk1_closest_indices
    return max_similarities, closest_indices

def find_color_cosine_matches(group1_features, group2_features, chunk_size1=100, chunk_size2=100):
    print(f"Finding closest matches using cosine similarity for color features...")
    max_similarities, closest_indices = calculate_cosine_similarity_double_chunked(
        group1_features, group2_features, chunk_size1, chunk_size2
    )
    return max_similarities, closest_indices

def save_matching_blocks(group1_blocks, group2_blocks, closest_indices, output_folder, indices_to_save):
    os.makedirs(output_folder, exist_ok=True)
    for i in indices_to_save:
        if i < len(group1_blocks):
            group1_block = group1_blocks[i]
            group2_block_index = closest_indices[i].item()
            group2_block = group2_blocks[group2_block_index]
            combined_image = Image.new('RGB', (group1_block.width + group2_block.width, max(group1_block.height, group2_block.height)))
            combined_image.paste(group1_block, (0, 0))
            combined_image.paste(group2_block, (group1_block.width, 0))
            combined_image.save(os.path.join(output_folder, f"combined_block_{i:03d}.png"))
    print(f"Saved matching blocks to '{output_folder}'.")

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU Memory - Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


