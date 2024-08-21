from src.models import vit_base

model = vit_base()

for name, param in model.named_parameters():
    print(name, param.shape)
