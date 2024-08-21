import torch
import requests
from PIL import Image
from src.models.vision_transformer import vit_huge
from src.transforms import make_transforms

model = vit_huge(patch_size=14)

# print model parameters
# for name, param in model.named_parameters():
#     print(name, param.shape)

# load pre-trained weights
state_dict = torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
    map_location="cpu",
)

new_state_dict = {}
for key, value in state_dict["encoder"].items():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

# load sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# preprocess image
transform = make_transforms()
pixel_values = transform(image).unsqueeze(0)

output = model(pixel_values)

print(f"Output shape: {output.shape}")
print(f"Output: {output}")
