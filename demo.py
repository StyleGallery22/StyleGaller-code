import os
import time
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from torchvision.utils import save_image


import utils
from pipeline import StyleGallery
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_folder = "outputs"

model_name = "pretrained_models/runwayml_stable-diffusion-v1-5"
vae = ""

seed = 42
start_layer, end_layer = 10, 16
noise_steps = 15

content_image = "content/animals/0008.png"
content_mask_path = None
style_image = ["style/Romanticism/Romanticism007.png"]

os.makedirs(os.path.join(output_folder, "content"), exist_ok=True)

scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler", device=device)
controller = utils.Controller(self_layers=(start_layer, end_layer))

pipe = StyleGallery.from_pretrained(
    model_name, scheduler=scheduler, safety_checker=None
).to(device)

pipe.extractor = pipe.unet

torch.manual_seed(seed)

style_image_path = style_image[0]
print(style_image_path)
depth = pipe.get_depth(
    style_image_path
)



content_original_mask = None
if content_mask_path is not None:
    content_original_mask = utils.convert_mask_to_array(content_mask_path)

cluster_matches, content_dict, style_dict = pipe.cluster_match(
    content_image,
    style_image,
    steps=noise_steps,
    content_original_mask=content_original_mask,
    use_depth=True
)

result = {}
for content_cluster, match_info in cluster_matches.items():
    result[content_cluster] = {
        'style_idx': match_info['style_dict_index'],
        'style_cluster': match_info['style_cluster'],
        'similarity': match_info['similarity']
    }
print(result)
result = pipe.style_transfer(
    content_dict=content_dict,
    style_dict=style_dict,
    controller=controller,
    cluster_matches=cluster_matches,
    num_optimize_steps=150
)
save_image(result, "result.png")
